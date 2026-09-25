"""Resumable root-only Classical finetuning and play qualification.

Example::

    python3 -m nnue.classical_train --run-dir /tmp/classical-finetune \
        --teacher-bin /path/to/frozen-v26-classical-engine \
        --nnue-bin /path/to/nnue --selfplay-bin /path/to/nnue-selfplay

The teacher generates all samples and is also the match opponent. Every epoch
saved by the trainer is screened at 5 ms from the exact Classical root. Only a
positive independent 5,000-game 5 ms gate permits a 10,000-game 50 ms gate.
Raw corpora and generated binaries remain in --run-dir, never in the release.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from nnue.classical_corpus import assemble


REPO = Path(__file__).resolve().parent.parent
OPENING = REPO / "data/positions/classical.fen"
TRAIN_SAMPLES = 500_000
VALIDATION_SAMPLES = 10_000
PEAK_LEARNING_RATE = 1.5e-5
MIN_LEARNING_RATE = 3e-6


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(parts: list[str | Path], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        subprocess.run(
            [str(part) for part in parts], cwd=REPO, env=env,
            stdout=output, stderr=subprocess.STDOUT, check=True,
        )


def corpus_count(nnue_bin: Path, shards: Path, work: Path, maximum: int) -> int:
    result = subprocess.run(
        [
            str(nnue_bin), "corpus-build", "--shards", str(shards),
            "--work-dir", str(work), "--cycle", "1",
            "--validation-samples", str(VALIDATION_SAMPLES),
            "--max-samples", str(maximum), "--max-abs-score", "3500",
            "--feature-set", "steinbeisser_nnue_features",
            "--input-count", "130", "--max-active-features", "28",
        ], cwd=REPO, check=True, capture_output=True, text=True,
    )
    return int(json.loads(result.stdout)["samples"])


def generate_stream(args: argparse.Namespace, label: str, target: int, seed: int) -> Path:
    shards = args.run_dir / f"{label}-shards"
    work = args.run_dir / f"{label}-work"
    shards.mkdir(parents=True, exist_ok=True)
    for number in range(1, 41):
        count = corpus_count(args.nnue_bin, shards, work, target)
        print(f"generation={label} unique={count}/{target}", flush=True)
        if count >= target:
            return work / "cycle1_fen"
        output = shards / f"{label}-{number:03}.sbin"
        if output.is_file():
            continue
        pending = output.with_suffix(".sbin.pending")
        pending.unlink(missing_ok=True)
        command(
            [
                args.selfplay_bin, "generate", "--repo", REPO,
                "--local-bin", args.teacher_bin, "--openings", OPENING,
                "--games-out", pending, "--target-samples", "62000",
                "--parallel-games", "15", "--max-abs-score", "3500",
                "--time", "100", "--seed", str(seed + number),
            ], args.run_dir / "logs" / f"generate-{label}-{number:03}.log",
        )
        pending.replace(output)
    raise RuntimeError(f"{label}: insufficient unique samples after 40 shards")


def prepare_corpus(args: argparse.Namespace) -> Path:
    output = args.run_dir / "corpus"
    audit_path = output / "assembly.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
    else:
        train = generate_stream(args, "train", TRAIN_SAMPLES + VALIDATION_SAMPLES, 20271000)
        heldout = generate_stream(args, "heldout", 60_000, 20272000)
        audit = assemble(train, heldout, output, VALIDATION_SAMPLES)
    if (
        audit["train_samples"] != TRAIN_SAMPLES
        or audit["validation_samples"] != VALIDATION_SAMPLES
        or audit["shared_exact_keys"] != 0
        or digest(output / "train.sbin") != audit["train_sha256"]
        or digest(output / "val.sbin") != audit["val_sha256"]
        or digest(output / "manifest.json") != audit["manifest_sha256"]
    ):
        raise ValueError("Classical corpus count, integrity, or leakage check failed")
    return output


def train(args: argparse.Namespace, corpus: Path) -> list[dict]:
    output = args.run_dir / "trainer"
    metrics_path = output / "metrics.json"
    identity_path = output / "inputs.json"
    identity = {
        "base_model_sha256": digest(args.base_model),
        "teacher_bin_sha256": digest(args.teacher_bin),
        "train_sha256": digest(corpus / "train.sbin"),
        "val_sha256": digest(corpus / "val.sbin"),
        "manifest_sha256": digest(corpus / "manifest.json"),
        "opening_sha256": digest(OPENING),
        "peak_learning_rate": PEAK_LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE,
    }
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("trainer inputs changed; use a separate --run-dir")
    elif metrics_path.is_file():
        raise ValueError("existing trainer metrics lack an input identity")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)
    if not metrics_path.is_file():
        env = os.environ.copy()
        env.update({
            "STEINBEISSER_NNUE_TRAIN_PATH": str(corpus / "train.sbin"),
            "STEINBEISSER_NNUE_VAL_PATH": str(corpus / "val.sbin"),
            "STEINBEISSER_NNUE_MANIFEST_PATH": str(corpus / "manifest.json"),
            "STEINBEISSER_NNUE_OUTPUT_DIR": str(output),
            "STEINBEISSER_NNUE_FEATURE_SET": "steinbeisser_nnue_features",
            "STEINBEISSER_NNUE_ARCHITECTURE": "130,84,50,1",
            "STEINBEISSER_NNUE_DATASET_CACHE_DIR": str(args.run_dir / "dataset-cache"),
            "STEINBEISSER_NNUE_LEARNING_RATE": str(PEAK_LEARNING_RATE),
            "STEINBEISSER_NNUE_MIN_LEARNING_RATE": str(MIN_LEARNING_RATE),
            "STEINBEISSER_NNUE_EPOCHS": "100",
            "STEINBEISSER_NNUE_PATIENCE": "10",
            "STEINBEISSER_TRAIN_SCREEN_CHECKPOINTS": "100",
            "STEINBEISSER_NNUE_INITIAL_MODEL": str(args.base_model),
            "STEINBEISSER_NNUE_CLI": str(args.nnue_bin),
            "STEINBEISSER_NNUE_THREADS": "1",
            "STEINBEISSER_NNUE_LOADER_WORKERS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        })
        command(
            [args.python, "-m", "nnue.fit"],
            args.run_dir / "logs" / "train.log", env=env,
        )
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("initial_model_sha256") != digest(args.base_model):
        raise ValueError("trainer did not initialize from the selected base model")
    if (
        int(metrics.get("train_dataset_size", -1)) != TRAIN_SAMPLES
        or int(metrics.get("val_dataset_size", -1)) != VALIDATION_SAMPLES
        or metrics.get("dataset_path") != str(corpus / "train.sbin")
        or metrics.get("val_dataset_path") != str(corpus / "val.sbin")
    ):
        raise ValueError("trainer metrics do not match the assembled corpus")
    history = metrics["history"]
    if not history or any(not row.get("screen_checkpoint_file") for row in history):
        raise ValueError("not every trained epoch has a playtest checkpoint")
    return history


def materialize(args: argparse.Namespace, epoch: int, model: Path) -> Path:
    key = f"epoch-{epoch:03}-{digest(model)[:12]}"
    target = args.run_dir / "models" / key
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        command(
            [
                args.nnue_bin, "materialize-candidate", "--repo", REPO,
                "--reference-ref", args.reference_ref, "--model", model,
                "--source-dir", args.run_dir / "sources" / key,
                "--target", target, "--target-dir", args.run_dir / "cargo-target" / key,
            ], args.run_dir / "logs" / f"materialize-{key}.log",
        )
    return target


def read_match(log: Path, games: int) -> dict:
    fields = {}
    for line in log.read_text().splitlines():
        if "," in line and not line.startswith("{"):
            key, value = line.split(",", 1)
            fields[key] = value
    errors = {
        key: int(fields[key])
        for key in ("github_illegal", "github_errors", "local_illegal", "local_errors")
    }
    wins, draws, losses = (int(value) for value in fields["w_d_l"].split("-"))
    if int(fields["games"]) != games or wins + draws + losses != games:
        raise ValueError(f"incomplete match: {log}")
    if fields.get("early_stopped") != "false" or any(errors.values()):
        raise ValueError(f"early stop or engine error: {log}: {errors}")
    return {
        "games": games, "wins": wins, "draws": draws, "losses": losses,
        "elo": float(fields["elo"]), "elo_95_ci": fields["elo_95_ci"],
        "errors": errors, "log": str(log), "log_sha256": digest(log),
        "candidate_avg_depth": float(fields["local_avg_depth"]) if "local_avg_depth" in fields else None,
        "candidate_avg_nps": float(fields["local_avg_nps"]) if "local_avg_nps" in fields else None,
        "search": {
            actor: {
                metric: int(fields[f"{actor}_{metric}"])
                for metric in ("moves", "nodes", "engine_ms", "depth_sum")
            }
            for actor in ("local", "github")
            if all(f"{actor}_{metric}" in fields for metric in ("moves", "nodes", "engine_ms", "depth_sum"))
        },
    }


def play(args: argparse.Namespace, name: str, candidate: Path, games: int, ms: int, seed: int) -> dict:
    log = args.run_dir / "logs" / f"match-{name}-{games}x{ms}.log"
    if log.is_file():
        try:
            return read_match(log, games)
        except (KeyError, ValueError):
            pass
    command(
        [
            args.selfplay_bin, "match", "--repo", REPO,
            "--github-bin", args.teacher_bin, "--local-bin", candidate,
            "--openings", OPENING, "--pairs", str(games // 2),
            "--parallel-games", "15", "--time", str(ms),
            "--seed", str(seed), "--progress-every-pairs", str(max(50, games // 20)),
        ], log,
    )
    return read_match(log, games)


def qualify(args: argparse.Namespace, history: list[dict]) -> dict:
    report_path = args.run_dir / "results.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {
        "protocol": "root-only Classical, every trained epoch playtested",
        "base_model_sha256": digest(args.base_model),
        "teacher_bin_sha256": digest(args.teacher_bin),
        "classical_opening_sha256": digest(OPENING),
        "corpus_train_sha256": digest(args.run_dir / "corpus/train.sbin"),
        "corpus_val_sha256": digest(args.run_dir / "corpus/val.sbin"),
        "reference_ref": args.reference_ref,
        "screens": [],
    }
    if (
        report.get("base_model_sha256") != digest(args.base_model)
        or report.get("teacher_bin_sha256") != digest(args.teacher_bin)
        or report.get("classical_opening_sha256") != digest(OPENING)
        or report.get("corpus_train_sha256") != digest(args.run_dir / "corpus/train.sbin")
        or report.get("corpus_val_sha256") != digest(args.run_dir / "corpus/val.sbin")
        or report.get("reference_ref") != args.reference_ref
    ):
        raise ValueError("run identity changed; use a separate --run-dir")
    rows = {int(row["epoch"]): row for row in report["screens"]}
    for checkpoint in history:
        epoch = int(checkpoint["epoch"])
        model = Path(checkpoint["screen_checkpoint_file"])
        if epoch in rows and rows[epoch]["model_sha256"] == digest(model):
            continue
        candidate = materialize(args, epoch, model)
        screen = play(
            args, f"epoch-{epoch:03}-{digest(model)[:12]}-screen",
            candidate, 1_000, 5, 20273000,
        )
        rows[epoch] = {
            "epoch": epoch, "qval_loss": checkpoint["quantized_val_loss"],
            "model": str(model), "model_sha256": digest(model),
            "candidate": str(candidate), "screen_1000x5": screen,
        }
        report["screens"] = [rows[key] for key in sorted(rows)]
        write_json(report_path, report)
        print(f"epoch={epoch} screen_elo={screen['elo']:+.2f}", flush=True)
    if len(rows) != len(history):
        raise ValueError("not all epochs were screened")
    winner = max(rows.values(), key=lambda row: (row["screen_1000x5"]["elo"], -row["qval_loss"]))
    report["selected_epoch"] = winner["epoch"]
    report["selected_model_sha256"] = winner["model_sha256"]
    candidate = materialize(args, winner["epoch"], Path(winner["model"]))
    key = winner["model_sha256"][:12]
    report["gate_5000x5"] = play(args, f"independent-gate-{key}", candidate, 5_000, 5, 20274000)
    write_json(report_path, report)
    if report["gate_5000x5"]["wins"] <= report["gate_5000x5"]["losses"]:
        report["fifty_ms_status"] = "not_eligible_after_5ms_gate"
        write_json(report_path, report)
        return report
    report["gate_10000x50"] = play(args, f"final-gate-{key}", candidate, 10_000, 50, 20275000)
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--teacher-bin", type=Path, required=True)
    parser.add_argument("--nnue-bin", type=Path, required=True)
    parser.add_argument("--selfplay-bin", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=REPO / "data/models/v27/general.nnq")
    parser.add_argument("--reference-ref", default="v2.6.1")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    for name in ("teacher_bin", "nnue_bin", "selfplay_bin", "base_model"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
        setattr(args, name, path)
    args.run_dir = args.run_dir.resolve()
    if args.plan:
        print(json.dumps({
            "train_samples": TRAIN_SAMPLES, "validation_samples": VALIDATION_SAMPLES,
            "teacher_sha256": digest(args.teacher_bin),
            "base_sha256": digest(args.base_model),
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "screen": "every epoch, 1000 games at 5 ms, exact Classical root",
            "gate": "5000 games at 5 ms, then 10000 at 50 ms only if positive",
        }, indent=2))
        return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "controller.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another Classical trainer owns {args.run_dir}") from error
        corpus = prepare_corpus(args)
        report = qualify(args, train(args, corpus))
    print(json.dumps({"selected_epoch": report["selected_epoch"],
                      "gate_5000x5": report["gate_5000x5"],
                      "gate_10000x50": report.get("gate_10000x50")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
