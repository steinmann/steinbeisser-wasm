"""Random-only v2.7 base-model tournament and independent release gate.

The 15 candidates are chosen by their 1,000-game random-opening screen Elo
against the frozen release. These 16 entrants play 500 color-paired games per
matchup (7,500 games per player). The leading candidate then plays a
separate 10,000-game 50 ms gate against the release on unused random starts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil

from nnue.classical_train import command, digest, read_match, write_json


CANDIDATES = 15
GAMES_PER_MATCHUP = 500
TOURNAMENT_MS = 5
GATE_GAMES = 10_000
GATE_MS = 50


def elo(points: float, games: int) -> float:
    score = min(max((points + 0.5) / (games + 1), 0.001), 0.999)
    return 400.0 * math.log10(score / (1.0 - score))


def select(screened: list[dict], reference_ref: str) -> list[dict]:
    ranked = sorted(
        (row for row in screened if row.get("reference_ref") == reference_ref),
        key=lambda row: (
            -float(row["random_elo_vs_release"]),
            float(row.get("qval_loss", math.inf)),
            str(row["id"]),
        ),
    )
    selected = []
    seen = set()
    for row in ranked:
        model = Path(str(row["model"]))
        binary = Path(str(row["source_bin"]))
        model_hash = str(row["model_sha256"])
        if (
            digest(model) != model_hash
            or digest(binary) != str(row["source_bin_sha256"])
        ):
            raise ValueError(f"screened candidate changed: {row['id']}")
        if model_hash in seen:
            continue
        seen.add(model_hash)
        selected.append(row)
        if len(selected) == CANDIDATES:
            return selected
    raise ValueError(f"only {len(selected)} distinct screened models; need {CANDIDATES}")


def match(
    repo: Path, selfplay_bin: Path, reference: Path, candidate: Path,
    openings: Path, games: int, ms: int, parallel: int, seed: int, log: Path,
) -> dict:
    if log.is_file():
        try:
            return read_match(log, games)
        except (KeyError, ValueError):
            pass
    command(
        [
            selfplay_bin, "match", "--repo", repo,
            "--github-bin", reference, "--local-bin", candidate,
            "--openings", openings, "--pairs", str(games // 2),
            "--parallel-games", str(parallel), "--time", str(ms),
            "--seed", str(seed),
        ], log,
    )
    return read_match(log, games)


def add_score(card: dict, wins: int, draws: int, losses: int, search: dict) -> None:
    card["wins"] += wins
    card["draws"] += draws
    card["losses"] += losses
    for metric in ("moves", "nodes", "engine_ms", "depth_sum"):
        card[metric] += int(search[metric])


def summarize(selected: list[dict], reference_ref: str, matches: list[dict]) -> list[dict]:
    players = [str(row["id"]) for row in selected] + [f"reference:{reference_ref}"]
    screened = {str(row["id"]): row for row in selected}
    cards = {
        player: {key: 0 for key in (
            "wins", "draws", "losses", "moves", "nodes", "engine_ms", "depth_sum"
        )} for player in players
    }
    for row in matches:
        result = row["result"]
        if "local" not in result["search"] or "github" not in result["search"]:
            raise ValueError("match is missing raw NPS/depth totals")
        add_score(cards[row["left"]], result["wins"], result["draws"], result["losses"], result["search"]["local"])
        add_score(cards[row["right"]], result["losses"], result["draws"], result["wins"], result["search"]["github"])
    def player_elo(card: dict) -> float:
        games = card["wins"] + card["draws"] + card["losses"]
        if games != CANDIDATES * GAMES_PER_MATCHUP:
            raise ValueError(f"player has {games} games; expected {CANDIDATES * GAMES_PER_MATCHUP}")
        return elo(card["wins"] + 0.5 * card["draws"], games)
    reference_elo = player_elo(cards[players[-1]])
    standings = []
    for player in players:
        card = cards[player]
        standings.append({
            "player": player,
            "elo_vs_release": player_elo(card) - reference_elo,
            "games": card["wins"] + card["draws"] + card["losses"],
            "wins": card["wins"], "draws": card["draws"], "losses": card["losses"],
            "avg_depth": card["depth_sum"] / max(1, card["moves"]),
            "avg_nps": card["nodes"] * 1000 / max(1, card["engine_ms"]),
            "screen_elo": float(screened.get(player, {}).get("random_elo_vs_release", 0.0)),
            "qval_loss": screened.get(player, {}).get("qval_loss"),
        })
    return sorted(standings, key=lambda row: (-row["elo_vs_release"], row["player"]))


def qualify(
    screened: list[dict], *, repo: Path, run_root: Path, reference_ref: str,
    reference_bin: Path, selfplay_bin: Path, tournament_openings: Path,
    gate_openings: Path, parallel_matches: int = 15,
) -> dict:
    selected = select(screened, reference_ref)
    if not reference_bin.is_file():
        raise ValueError("missing release binary")
    players = [
        {"id": str(row["id"]), "binary": Path(str(row["source_bin"])), "model": Path(str(row["model"]))}
        for row in selected
    ] + [{"id": f"reference:{reference_ref}", "binary": reference_bin}]
    identity = {
        "reference_ref": reference_ref,
        "reference_bin_sha256": digest(reference_bin),
        "selected": [
            [row["id"], row["model_sha256"], row["source_bin_sha256"]]
            for row in selected
        ],
        "tournament_openings_sha256": digest(tournament_openings),
        "gate_openings_sha256": digest(gate_openings),
        "games_per_matchup": GAMES_PER_MATCHUP,
        "time_ms": TOURNAMENT_MS,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = run_root / f"general-qualification-{fingerprint}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity.json", identity)
    jobs = [
        (number, left, right)
        for number, (left, right) in enumerate(
            ((left, right) for index, left in enumerate(players) for right in players[index + 1:]),
            start=1,
        )
    ]

    def play_pair(job: tuple[int, dict, dict]) -> dict:
        number, left, right = job
        result = match(
            repo, selfplay_bin, right["binary"], left["binary"],
            tournament_openings, GAMES_PER_MATCHUP, TOURNAMENT_MS,
            1, 20276000 + number, output / f"pair-{number:03}.log",
        )
        return {"pair": number, "left": left["id"], "right": right["id"], "result": result}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(15, parallel_matches))) as executor:
        for future in as_completed([executor.submit(play_pair, job) for job in jobs]):
            rows.append(future.result())
            write_json(output / "progress.json", {
                "identity": identity, "matches_completed": len(rows),
                "matches_requested": len(jobs),
            })
    rows.sort(key=lambda row: row["pair"])
    standings = summarize(selected, reference_ref, rows)
    winner = next(row for row in standings if row["player"] != f"reference:{reference_ref}")
    report = {
        "identity": identity, "standings": standings,
        "pairings": len(rows), "winner": winner["player"],
        "winner_model": str(next(row["model"] for row in selected if row["id"] == winner["player"])),
        "output_dir": str(output),
        "tournament_matches": rows,
    }
    write_json(output / "results.json", report)
    winner_record = next(row for row in selected if row["id"] == winner["player"])
    gate = match(
        repo, selfplay_bin, reference_bin, Path(str(winner_record["source_bin"])),
        gate_openings, GATE_GAMES, GATE_MS, 15, 20277000,
        output / "release-gate-10000x50.log",
    )
    report["release_gate"] = gate
    report["release_gate_passed"] = gate["wins"] > gate["losses"]
    if report["release_gate_passed"]:
        shutil.copyfile(Path(str(winner_record["model"])), output / "winner.nnq")
        report["winner_model_sha256"] = digest(output / "winner.nnq")
    write_json(output / "results.json", report)
    return report
