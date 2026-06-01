#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

NNUE_MODULE_DIR = Path(__file__).resolve().parent
if str(NNUE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(NNUE_MODULE_DIR))

from fit import (
    current_feature_schema,
    dataset_cache_format,
    dataset_cache_mmap_mode,
    read_json,
    trainer_main,
    write_json,
)


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int
    elo: float
    elo_lower: float
    elo_upper: float


@dataclass(frozen=True)
class TrainConfig:
    validation_samples: int
    min_train_increment: int
    initial_train_samples: int
    max_train_increment: int
    max_cycles: int
    target_train_samples: int

    def validate(self) -> None:
        if self.validation_samples <= 0:
            raise SystemExit("STEINBEISSER_TRAIN_VALIDATION_SAMPLES must be positive")
        if self.initial_train_samples <= 0:
            raise SystemExit("STEINBEISSER_TRAIN_INITIAL_SAMPLES must be positive")
        if self.min_train_increment <= 0:
            raise SystemExit("STEINBEISSER_TRAIN_MIN_INCREMENT must be positive")
        if self.max_train_increment < 0:
            raise SystemExit("STEINBEISSER_TRAIN_MAX_INCREMENT must be non-negative")
        if self.max_cycles < 0:
            raise SystemExit("STEINBEISSER_TRAIN_MAX_CYCLES must be non-negative")
        if self.target_train_samples < 0:
            raise SystemExit("STEINBEISSER_TRAIN_TARGET_SAMPLES must be non-negative")

    def train_increment(self, previous_train_samples: int) -> int:
        increment = self.min_train_increment
        if self.max_train_increment > 0:
            increment = min(increment, self.max_train_increment)
        return max(1, increment)

    def required_train_samples(self, previous_train_samples: int) -> int:
        return max(
            self.initial_train_samples,
            previous_train_samples + self.train_increment(previous_train_samples),
        )

    def complete(self, completed_cycle: int, train_samples: int) -> bool:
        cycle_done = self.max_cycles > 0 and completed_cycle >= self.max_cycles
        samples_done = self.target_train_samples > 0 and train_samples >= self.target_train_samples
        if self.max_cycles > 0 and self.target_train_samples > 0:
            return cycle_done and samples_done
        if self.max_cycles > 0:
            return cycle_done
        if self.target_train_samples > 0:
            return samples_done
        return False


@dataclass
class RunState:
    cycle: int = 0
    model: str | None = None
    source_bin: str | None = None
    train_samples: int = 0
    reference_ref: str = ""
    screened_candidates: list[dict[str, object]] = field(default_factory=list)
    tournament_completed: bool = False
    tournament_summary: dict[str, object] | None = None
    tournament_results: object = None
    training_data_export: object = None
    last_error: str | None = None
    last_error_time: str | None = None

    @classmethod
    def load(cls) -> "RunState":
        if not STATE_PATH.is_file():
            return cls(reference_ref=REFERENCE_REF)
        raw = read_json(STATE_PATH)
        if not isinstance(raw, dict):
            raise SystemExit(f"{STATE_PATH} must contain a JSON object")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "RunState":
        cycle = int(raw.get("cycle", 0))
        if cycle > 0 and not raw.get("model"):
            raise SystemExit(
                f"{STATE_PATH} is not compatible with the current JAX-only trainer; run "
                "`python3 nnue/train.py --clean` "
                "or remove that state before starting fresh"
            )
        raw_ref = raw.get("reference_ref")
        if cycle > 0 and not raw_ref:
            raise SystemExit(
                f"{STATE_PATH} does not record its release reference; run "
                "`python3 nnue/train.py --clean` "
                "to start a fresh release track"
            )
        reference_ref = str(raw_ref or REFERENCE_REF)
        if cycle > 0 and reference_ref != REFERENCE_REF:
            raise SystemExit(
                f"{STATE_PATH} belongs to reference {reference_ref}, but train resolved {REFERENCE_REF}; "
                "run `python3 nnue/train.py --clean` to start a fresh release track"
            )
        screened = raw.get("screened_candidates")
        summary = raw.get("tournament_summary")
        return cls(
            cycle=cycle,
            model=str(raw["model"]) if raw.get("model") is not None else None,
            source_bin=str(raw["source_bin"]) if raw.get("source_bin") is not None else None,
            train_samples=int(raw.get("train_samples", 0)),
            reference_ref=reference_ref,
            screened_candidates=[
                candidate for candidate in screened if isinstance(candidate, dict)
            ] if isinstance(screened, list) else [],
            tournament_completed=bool(raw.get("tournament_completed")),
            tournament_summary=summary if isinstance(summary, dict) else None,
            tournament_results=raw.get("tournament_results"),
            training_data_export=raw.get("training_data_export"),
            last_error=str(raw["last_error"]) if raw.get("last_error") else None,
            last_error_time=str(raw["last_error_time"]) if raw.get("last_error_time") else None,
        )

    @property
    def next_cycle(self) -> int:
        return self.cycle + 1

    def complete(self, config: TrainConfig) -> bool:
        return config.complete(self.cycle, self.train_samples)

    def to_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "cycle": self.cycle,
            "model": self.model,
            "source_bin": self.source_bin,
            "train_samples": self.train_samples,
            "reference_ref": self.reference_ref,
            "screened_candidates": self.screened_candidates,
            "tournament_completed": self.tournament_completed,
            "tournament_summary": self.tournament_summary,
            "tournament_results": self.tournament_results,
            "training_data_export": self.training_data_export,
        }
        if self.last_error:
            state["last_error"] = self.last_error
        if self.last_error_time:
            state["last_error_time"] = self.last_error_time
        return state

    def save(self) -> None:
        write_json(STATE_PATH, self.to_dict())

    def record_cycle(self, result: "CycleResult") -> None:
        self.cycle = result.cycle
        self.model = str(result.model)
        self.source_bin = str(result.source_bin)
        self.train_samples = result.train_samples
        self.reference_ref = REFERENCE_REF
        self.screened_candidates.extend(result.positive_records)
        self.tournament_completed = False
        self.tournament_summary = None
        self.tournament_results = None
        self.training_data_export = None


@dataclass
class Scorecard:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: float = 0.0
    games: int = 0

    def add(self, wins: int, draws: int, losses: int) -> None:
        self.wins += wins
        self.draws += draws
        self.losses += losses
        self.points += wins + 0.5 * draws
        self.games += wins + draws + losses

    @property
    def score_pct(self) -> float:
        return self.points / self.games * 100.0 if self.games else 0.0

    @property
    def elo(self) -> float:
        return elo_from_points(self.points, self.games)


@dataclass(frozen=True)
class OpeningBookConfig:
    open_book: Path
    match_openings: Path
    match_openings_override: str | None
    single_openings_dir: Path
    random_openings: Path
    match_single_count: int
    match_random_count: int
    tournament_openings: Path
    tournament_openings_explicit: bool


@dataclass(frozen=True)
class GeneratorConfig:
    selfplay_bin: Path
    repo: Path
    openings: Path
    shard_dir: Path
    log_path: Path
    chunk_samples: int
    workers: int
    selfplay_ms: int
    max_abs_score: int
    backlog_samples: int
    progress_interval_seconds: float


StatusEmitter = Callable[[str], None]
CommandFormatter = Callable[[list[object]], str]


class ContinuousGenerator:
    def __init__(
        self,
        engine: Path,
        config: GeneratorConfig,
        emit_status: StatusEmitter,
        command_text: CommandFormatter,
    ) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.error: str | None = None
        self.engine = engine
        self.config = config
        self.emit_status = emit_status
        self.command_text = command_text
        self.target_unique_samples = 0
        self.last_unique_samples = 0
        self.generated_since_snapshot = 0
        self.last_backlog_emit = 0.0

    def start(self) -> None:
        if self.thread is None:
            self.emit_status(
                "generator=start "
                f"engine={self.engine} "
                f"chunk_samples={self.config.chunk_samples} "
                f"parallel_games={self.config.workers} "
                f"time_ms={self.config.selfplay_ms}"
            )
            self.thread = threading.Thread(target=self.run, name="selfplay-generator", daemon=True)
            self.thread.start()

    def check(self) -> None:
        with self.lock:
            error = self.error
        if error is not None:
            raise SystemExit(error)

    def set_target(self, target_unique_samples: int) -> None:
        with self.lock:
            self.target_unique_samples = max(0, int(target_unique_samples))

    def note_unique_samples(self, unique_samples: int) -> None:
        with self.lock:
            self.last_unique_samples = max(self.last_unique_samples, int(unique_samples))
            self.generated_since_snapshot = 0

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def run(self) -> None:
        while not self.stop_event.is_set():
            if not self.wait_for_backlog_slot():
                return
            if not self.run_one_shard():
                return

    def wait_for_backlog_slot(self) -> bool:
        while not self.stop_event.is_set():
            with self.lock:
                target = self.target_unique_samples
                estimated_unique_samples = self.last_unique_samples + self.generated_since_snapshot
            if (
                target <= 0
                or self.config.backlog_samples <= 0
                or estimated_unique_samples < target + self.config.backlog_samples
            ):
                return True
            now = time.monotonic()
            if now - self.last_backlog_emit >= self.config.progress_interval_seconds:
                self.emit_status(
                    "generator=backlog_wait "
                    f"estimated_unique_samples={estimated_unique_samples} "
                    f"target_unique_samples={target} "
                    f"backlog_cap={self.config.backlog_samples}"
                )
                self.last_backlog_emit = now
            time.sleep(1.0)
        return False

    def run_one_shard(self) -> bool:
        config = self.config
        final = config.shard_dir / f"selfplay-bg-{time.time_ns()}.sbin"
        tmp = final.with_suffix(".sbin.tmp")
        started = time.monotonic()
        cmd = [
            config.selfplay_bin,
            "generate",
            "--repo",
            config.repo,
            "--openings",
            config.openings,
            "--games-out",
            tmp,
            "--target-samples",
            str(config.chunk_samples),
            "--parallel-games",
            str(config.workers),
            "--time",
            str(config.selfplay_ms),
            "--max-abs-score",
            str(config.max_abs_score),
            "--seed",
            str(time.time_ns() & ((1 << 63) - 1)),
            "--local-bin",
            self.engine,
        ]
        self.emit_status(f"generator=shard_start file={final.name}")
        with config.log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {self.command_text(cmd)}\n")
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=config.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self.lock:
                self.proc = proc
            while proc.poll() is None:
                if self.stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    tmp.unlink(missing_ok=True)
                    with self.lock:
                        self.proc = None
                    return False
                time.sleep(1)
            with self.lock:
                self.proc = None
            if self.stop_event.is_set():
                tmp.unlink(missing_ok=True)
                return False
            if proc.returncode != 0:
                tmp.unlink(missing_ok=True)
                error = f"selfplay generator failed with status {proc.returncode}; see {config.log_path}"
                with self.lock:
                    self.error = error
                log.write(f"# {error}\n")
                return False
            tmp.replace(final)
            with self.lock:
                self.generated_since_snapshot += config.chunk_samples
            log.write(f"# generated_shard {final}\n")
        elapsed = time.monotonic() - started
        self.emit_status(
            "generator=shard_done "
            f"file={final.name} "
            f"bytes={final.stat().st_size} "
            f"elapsed_s={elapsed:.1f}"
        )
        return True


def elo_from_points(points: float, games: int) -> float:
    if games <= 0:
        return 0.0
    score = min(max((points + 0.5) / (games + 1.0), 0.001), 0.999)
    return -400.0 * math.log10((1.0 / score) - 1.0)


def match_list_fields(matches: list[MatchResult], label: str) -> str:
    if not matches:
        return f"wdl_{label}=NA elo_{label}=NA"
    wdl = ",".join(f"{match.wins}-{match.draws}-{match.losses}" for match in matches)
    elos = ",".join(
        f"{match.elo:+.2f}[{match.elo_lower:+.2f},{match.elo_upper:+.2f}]"
        for match in matches
    )
    return f"wdl_{label}={wdl} elo_{label}={elos}"


def row_epochs(rows: list[dict[str, object]]) -> str:
    return ",".join(str(int(row["epoch"])) for row in rows)


def row_qvals(rows: list[dict[str, object]]) -> str:
    return ",".join(f"{float(row['quantized_val_loss']):.6f}" for row in rows)


def tournament_table_lines(standings: list[dict[str, object]]) -> list[str]:
    headers = [
        "Rank",
        "Player",
        "Elo vs latest",
        "Games",
        "Score",
        "WDL",
        "Screen Elo",
        "QVal",
    ]
    rows: list[list[str]] = []
    for rank, row in enumerate(standings, start=1):
        qval = row.get("qval_loss")
        qval_text = "ref" if qval is None else f"{float(qval):.6f}"
        rows.append(
            [
                str(rank),
                markdown_table_cell(row.get("player", "")),
                f"{float(row.get('elo_vs_latest', 0.0)):+.1f}",
                str(int(row.get("games", 0))),
                f"{float(row.get('score_pct', 0.0)):.2f}%",
                f"{int(row.get('wins', 0))}-{int(row.get('draws', 0))}-{int(row.get('losses', 0))}",
                f"{float(row.get('screen_elo', 0.0)):+.1f}",
                qval_text,
            ]
        )
    return ["## Final positive-hit round robin"] + compact_markdown_table(headers, rows)


def markdown_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def compact_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def format_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return [format_row(headers), format_row(["---"] * len(headers))] + [
        format_row(row) for row in rows
    ]


PLOT_DARK_GRAY = "#575757"
PLOT_LIGHT_GRAY = "#bcbcbc"
PLOT_RED = "#d62728"
PLOT_AXIS = "#262626"
PLOT_ZERO = "#777777"
PLOT_GRID = "#e6e6e6"
PLOT_ERROR = "#171717"


def screening_plot_stems() -> list[Path]:
    return [OUTPUT_DIR / "elo_screen"]


SCREENING_MATPLOTLIB_CODE = r"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rcParams

try:
    import seaborn as sns
    sns.set_style("white")
except Exception:
    pass

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.unicode_minus": True,
    "svg.fonttype": "none",
})

payload = json.loads(Path(sys.argv[1]).read_text())
cycles = payload["cycles"]
reference_ref = payload["reference_ref"]
dark = payload["dark_gray"]
light = payload["light_gray"]
red = payload["red"]
axis = payload["axis"]
zero = payload["zero"]

xs = [float(row["positions_m"]) for row in cycles]
qvals = [float(row["qval"]) for row in cycles]
pos_x, pos_y, neg_x, neg_y = [], [], [], []
for row in cycles:
    x = float(row["positions_m"])
    for elo in row["elos"]:
        elo = float(elo)
        if elo >= 0:
            pos_x.append(x)
            pos_y.append(elo)
        else:
            neg_x.append(x)
            neg_y.append(elo)

all_elos = pos_y + neg_y
x_max = max(15.2, max(xs) + 0.2)
y_min = min(-225.0, min(all_elos) - 12.0)
y_max = max(65.0, max(all_elos) + 12.0)
q_min = min(qvals) - 0.00004
q_max = max(qvals) + 0.00004

fig_w = 656.910625 / 72.0
fig_h = 423.356875 / 72.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
fig.subplots_adjust(
    left=99.478125 / 656.910625,
    right=537.765625 / 656.910625,
    top=1.0 - 7.2 / 423.356875,
    bottom=1.0 - 350.4 / 423.356875,
)
ax2 = ax.twinx()

ax.scatter(neg_x, neg_y, s=50, color=light, linewidths=0, zorder=3)
ax.scatter(pos_x, pos_y, s=50, color=dark, linewidths=0, zorder=4)
ax2.plot(xs, qvals, color=red, linewidth=2.8, solid_capstyle="round", zorder=2)
ax.axhline(0, color=zero, linewidth=1.2, zorder=1)

ax.set_xlim(0, x_max)
ax.set_ylim(y_min, y_max)
ax2.set_ylim(q_min, q_max)
ax.set_xticks(list(range(0, int(math.floor(x_max)) + 1, 2)))
ax.set_yticks([-200, -150, -100, -50, 0, 50])

q_step = 0.0002
q_start = math.ceil((min(qvals) - 1e-12) / q_step) * q_step
q_end = math.floor((max(qvals) + 1e-12) / q_step) * q_step
q_ticks = []
value = q_start
while value <= q_end + 1e-12:
    q_ticks.append(round(value, 4))
    value += q_step
ax2.set_yticks(q_ticks)

ax.set_xlabel("Number of training positions in million", fontsize=22, fontweight="bold", labelpad=12.5)
ax.set_ylabel(f"Elo vs {reference_ref}", fontsize=22, fontweight="bold", labelpad=10.5)
ax2.set_ylabel("QVal loss", fontsize=22, fontweight="bold", color=red, labelpad=13.5)

ax.tick_params(axis="both", labelsize=18, colors=axis, width=1.1, length=6, pad=7)
ax2.tick_params(axis="y", labelsize=18, colors=red, labelcolor=red, width=1.1, length=6, pad=7)
for label in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(False)
ax2.grid(False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(axis)
ax.spines["bottom"].set_color(axis)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_color(red)
ax2.spines["right"].set_linewidth(1.2)

for stem in payload["stems"]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=240)
plt.close(fig)
"""


TOURNAMENT_BARPLOT_MATPLOTLIB_CODE = r"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rcParams

try:
    import seaborn as sns
    sns.set_style("white")
except Exception:
    pass

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.unicode_minus": True,
    "svg.fonttype": "none",
})

payload = json.loads(Path(sys.argv[1]).read_text())
rows = payload["rows"]
reference_ref = payload["reference_ref"]
dark = payload["dark_gray"]
light = payload["light_gray"]
axis = payload["axis"]
grid = payload["grid"]
error_color = payload["error"]

players = [str(row["player"]) for row in rows]
elos = [float(row["elo"]) for row in rows]
errors = [float(row["error"]) for row in rows]

left_edge = min(elo - err for elo, err in zip(elos, errors))
right_edge = max(elo + err for elo, err in zip(elos, errors))
pad = max(2.5, 0.06 * (right_edge - left_edge))
x_min = left_edge - pad
x_max = right_edge + pad

view_width = 657.507578
left = 135.257578
right = 650.307578
top = 7.2
bottom_margin = 56.578125
row_step = 28.260584
row_pad = 32.245326
plot_h = max(120.0, 2.0 * row_pad + row_step * max(0, len(rows) - 1))
bottom = top + plot_h
view_height = bottom + bottom_margin

fig, ax = plt.subplots(figsize=(view_width / 72.0, view_height / 72.0), dpi=300)
fig.subplots_adjust(
    left=left / view_width,
    right=right / view_width,
    top=1.0 - top / view_height,
    bottom=1.0 - bottom / view_height,
)

y = list(range(len(rows)))
colors = [dark if elo > 0 else light if elo < 0 else "#ffffff" for elo in elos]
edges = ["#303030" if elo > 0 else "#777777" if elo < 0 else "#ffffff" for elo in elos]
ax.barh(
    y,
    elos,
    height=0.62,
    color=colors,
    edgecolor=edges,
    linewidth=0.65,
    xerr=errors,
    error_kw={"ecolor": error_color, "elinewidth": 1.05, "capsize": 3.5, "capthick": 1.05},
)
ax.set_yticks(y)
ax.set_yticklabels(players)
ax.invert_yaxis()
ax.set_xlim(x_min, x_max)

tick_start = math.ceil(x_min / 10.0) * 10
tick_end = math.floor(x_max / 10.0) * 10
ax.set_xticks(list(range(int(tick_start), int(tick_end) + 1, 10)))
ax.set_xlabel(f"Elo vs {reference_ref}", fontsize=17, fontweight="bold", labelpad=8)

ax.xaxis.grid(True, color=grid, linewidth=0.9, linestyle=(0, (3.33, 1.44)))
ax.yaxis.grid(False)
ax.axvline(0, color=axis, linewidth=1.05)
ax.tick_params(axis="x", labelsize=13, colors=axis, width=1.0, length=4.8, pad=7)
ax.tick_params(axis="y", labelsize=11.5, colors=axis, length=0, pad=3.5)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

for stem in payload["stems"]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=300)
plt.close(fig)
"""


def plotting_python() -> Path | None:
    def has_plotting(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        probe = subprocess.run(
            [str(candidate), "-c", "import matplotlib, seaborn"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return probe.returncode == 0

    candidates = [
        Path(os.environ["STEINBEISSER_PLOT_PYTHON"])
        for _ in [None]
        if os.environ.get("STEINBEISSER_PLOT_PYTHON")
    ]
    candidates += [Path("/tmp/steinbeisser-plot-venv/bin/python"), Path(sys.executable)]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_plotting(candidate):
            return candidate
    venv_python = Path("/tmp/steinbeisser-plot-venv/bin/python")
    venv_root = venv_python.parent.parent
    try:
        if not venv_python.is_file():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                check=True,
            )
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "matplotlib",
                "seaborn",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=True,
        )
    except Exception as error:
        append_log(f"# plot dependency install failed: {error}")
        return None
    if has_plotting(venv_python):
        return venv_python
    return None


def run_plotter(name: str, script: str, payload: dict[str, object]) -> bool:
    python = plotting_python()
    if python is None:
        append_log(f"# {name} plot skipped: matplotlib/seaborn is unavailable")
        return False
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    payload_path = WORK_DIR / f"{name}-plot-payload.json"
    write_json(payload_path, payload)
    proc = subprocess.run(
        [str(python), "-c", script, str(payload_path)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode == 0:
        return True
    append_log(f"# {name} plot skipped: {proc.stdout.strip()}")
    return False


def parse_public_cycle_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    cycle_pattern = re.compile(
        r"cycle=(\d+)\s+train=(\d+)\s+val=(\d+)\s+best_epoch=(\d+)\s+"
        r"best_qval_loss=([0-9.]+).*?\belo_[^=]+=([^\n]+)"
    )
    elo_pattern = re.compile(r"([+-]?[0-9]+(?:\.[0-9]+)?)\[")
    cycles: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = cycle_pattern.search(line)
        if not match:
            continue
        cycle, train, _val, _epoch, qval, elo_blob = match.groups()
        elos = [float(value) for value in elo_pattern.findall(elo_blob)]
        if not elos:
            continue
        cycles.append(
            {
                "cycle": int(cycle),
                "train": int(train),
                "positions_m": int(train) / 1_000_000.0,
                "qval": float(qval),
                "elos": elos,
            }
        )
    return cycles


def write_screening_plot(reference_ref: str) -> None:
    try:
        cycles = parse_public_cycle_lines(SCREEN_LOG_PATH)
        if not cycles:
            return
        run_plotter(
            "screening",
            SCREENING_MATPLOTLIB_CODE,
            {
                "cycles": cycles,
                "reference_ref": reference_ref,
                "stems": [str(path) for path in screening_plot_stems()],
                "dark_gray": PLOT_DARK_GRAY,
                "light_gray": PLOT_LIGHT_GRAY,
                "red": PLOT_RED,
                "axis": PLOT_AXIS,
                "zero": PLOT_ZERO,
            },
        )
    except Exception as error:
        append_log(f"# screening plot skipped: {error}")


def performance_elo_se(row: dict[str, object]) -> float:
    games = int(row.get("games", 0))
    if games <= 0:
        return 0.0
    wins = float(row.get("wins", 0))
    draws = float(row.get("draws", 0))
    p = min(max((wins + 0.5 * draws) / games, 0.000001), 0.999999)
    ex2 = (wins + 0.25 * draws) / games
    se_p = math.sqrt(max(0.0, ex2 - p * p) / games)
    return 400.0 * se_p / (math.log(10.0) * p * (1.0 - p))


def write_tournament_barplot(results_path: Path) -> None:
    try:
        if not results_path.is_file():
            return
        summary = read_json(results_path)
        if not isinstance(summary, dict) or summary.get("status") != "completed":
            return
        raw_rows = summary.get("standings")
        if not isinstance(raw_rows, list):
            return
        rows = sorted(
            [row for row in raw_rows if isinstance(row, dict)],
            key=lambda row: float(row.get("elo_vs_latest", 0.0)),
            reverse=True,
        )
        if not rows:
            return
        reference_ref = str(summary.get("reference_ref") or REFERENCE_REF)
        elos = [float(row.get("elo_vs_latest", 0.0)) for row in rows]
        errors = [
            0.0 if bool(row.get("is_reference")) else 1.96 * performance_elo_se(row)
            for row in rows
        ]
        stems = [TOURNAMENT_RESULT_DIR / "elo_tournament"]
        run_plotter(
            "tournament-barplot",
            TOURNAMENT_BARPLOT_MATPLOTLIB_CODE,
            {
                "reference_ref": reference_ref,
                "rows": [
                    {
                        "player": str(row.get("player", "")),
                        "elo": elo,
                        "error": error,
                    }
                    for row, elo, error in zip(rows, elos, errors)
                ],
                "stems": [str(path) for path in stems],
                "dark_gray": PLOT_DARK_GRAY,
                "light_gray": PLOT_LIGHT_GRAY,
                "axis": PLOT_AXIS,
                "grid": PLOT_GRID,
                "error": PLOT_ERROR,
            },
        )
    except Exception as error:
        append_log(f"# tournament barplot skipped: {error}")


def read_fen_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def prepare_match_openings(config: OpeningBookConfig) -> None:
    if config.match_openings_override:
        if not config.match_openings.is_file():
            raise SystemExit(f"missing match openings override: {config.match_openings}")
        return
    openings = load_single_position_openings(config) + load_random_book_openings(config)
    config.match_openings.parent.mkdir(parents=True, exist_ok=True)
    config.match_openings.write_text("\n".join(openings) + "\n", encoding="utf-8")


def write_opening_file(path: Path, openings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(openings) + "\n", encoding="utf-8")


def require_unique_openings(openings: list[str], label: str) -> None:
    seen: set[str] = set()
    for index, opening in enumerate(openings, start=1):
        if opening in seen:
            raise SystemExit(f"{label} contains duplicate opening at line {index}")
        seen.add(opening)


def prepare_tournament_openings(
    config: OpeningBookConfig,
    paired_openings_per_pairing: int,
) -> tuple[Path, list[str]]:
    required = paired_openings_per_pairing
    if config.tournament_openings_explicit:
        if not config.tournament_openings.is_file():
            raise SystemExit(
                f"missing tournament openings override: {config.tournament_openings}"
            )
        openings = [
            normalize_material_scores(line)
            for line in read_fen_lines(config.tournament_openings)
        ]
        if len(openings) < required:
            raise SystemExit(
                f"{config.tournament_openings} contains {len(openings)} tournament openings; "
                f"need {required} so no opening repeats within a pairing"
            )
        openings = openings[:required]
        require_unique_openings(openings, "tournament opening override")
        return config.tournament_openings, openings
    openings = load_unique_book_openings(
        config.random_openings,
        required,
        "tournament",
    )
    write_opening_file(config.tournament_openings, openings)
    return config.tournament_openings, openings


def load_single_position_openings(config: OpeningBookConfig) -> list[str]:
    if not config.single_openings_dir.is_dir():
        raise SystemExit(f"missing single-position opening dir: {config.single_openings_dir}")
    openings: list[str] = []
    for path in sorted(config.single_openings_dir.glob("*.fen")):
        lines = read_fen_lines(path)
        if len(lines) != 1:
            raise SystemExit(f"{path} must contain exactly one non-comment FEN line")
        openings.append(normalize_material_scores(lines[0]))
    if len(openings) < config.match_single_count:
        raise SystemExit(
            f"{config.single_openings_dir} contains {len(openings)} single-position FENs; "
            f"need {config.match_single_count}"
        )
    return openings[: config.match_single_count]


def load_random_book_openings(config: OpeningBookConfig) -> list[str]:
    return load_book_openings(
        config.random_openings,
        config.match_random_count,
        "random match",
    )


def load_book_openings(path: Path, count: int, label: str) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"missing {label} opening book: {path}")
    openings = [normalize_material_scores(line) for line in read_fen_lines(path)]
    if len(openings) < count:
        raise SystemExit(f"{path} contains {len(openings)} FENs; need {count} for {label}")
    return openings[:count]


def load_unique_book_openings(path: Path, count: int, label: str) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"missing {label} opening book: {path}")
    openings: list[str] = []
    seen: set[str] = set()
    for line in read_fen_lines(path):
        opening = normalize_material_scores(line)
        if opening in seen:
            continue
        seen.add(opening)
        openings.append(opening)
        if len(openings) == count:
            return openings
    raise SystemExit(f"{path} contains {len(openings)} unique FENs; need {count} for {label}")


def normalize_material_scores(fen: str) -> str:
    fields = fen.split()
    if len(fields) < 6:
        raise SystemExit(f"FEN needs at least 6 fields: {fen}")
    black_pieces = fields[0].count("S")
    white_pieces = fields[0].count("s")
    fields[1] = str(max(0, 14 - white_pieces))
    fields[2] = str(max(0, 14 - black_pieces))
    return " ".join(fields)

# Training orchestration -----------------------------------------------------

NNUE_DIR = Path(__file__).resolve().parent
REPO = NNUE_DIR.parent
WORKSPACE = REPO
RUN_ROOT = Path(os.environ.get("STEINBEISSER_TRAIN_RUN_DIR", "/tmp/steinbeisser-train"))
BIN_DIR = RUN_ROOT / "bin"
SHARD_DIR = RUN_ROOT / "shards"
WORK_DIR = RUN_ROOT / "work"
CORPUS_DATA_DIR = WORK_DIR / "corpus-data"
SOURCE_BIN_DIR = RUN_ROOT / "source-bin"
REFERENCE_SOURCE_DIR = RUN_ROOT / "reference-sources"
CARGO_TARGET_DIR = RUN_ROOT / "cargo-target"
TRAINER_ARTIFACT_ROOT = RUN_ROOT / "trainer-artifacts"
DATASET_CACHE_DIR = Path(os.environ.get("STEINBEISSER_TRAIN_DATASET_CACHE_DIR", RUN_ROOT / "dataset-cache"))
CANONICAL_TRAINING_DIR = Path()
OUTPUT_DIR = Path()
POSITIVE_NET_DIR = Path()
TOURNAMENT_RESULT_DIR = Path()
RUN_OUTPUT_DIRS: tuple[Path, ...] = ()
LOG_PATH = RUN_ROOT / "train.log"
REPORT_PATH = RUN_ROOT / "cycles.log"
STATE_PATH = RUN_ROOT / "state.json"
VALIDATION_KEYS_PATH = WORK_DIR / "validation_keys.json"
SCREEN_LOG_PATH = NNUE_DIR / "train.log"

OPENINGS = Path(os.environ.get("STEINBEISSER_TRAIN_OPENINGS", REPO / "data/random100K.fen"))
MATCH_OPENINGS_OVERRIDE = os.environ.get("STEINBEISSER_TRAIN_MATCH_OPENINGS")
MATCH_OPENINGS = (
    Path(MATCH_OPENINGS_OVERRIDE)
    if MATCH_OPENINGS_OVERRIDE
    else WORK_DIR / "screening_openings_120.fen"
)
MATCH_SINGLE_OPENINGS_DIR = Path(
    os.environ.get("STEINBEISSER_TRAIN_MATCH_SINGLE_OPENINGS_DIR", REPO / "data/positions")
)
MATCH_RANDOM_OPENINGS = Path(os.environ.get("STEINBEISSER_TRAIN_MATCH_RANDOM_OPENINGS", OPENINGS))
NNUE_MANIFEST = NNUE_DIR / "Cargo.toml"
SELFPLAY_SOURCE = NNUE_DIR / "src/bin/selfplay.rs"
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
SELFPLAY_BIN = CARGO_TARGET_DIR / "release" / f"nnue-selfplay{EXE_SUFFIX}"
NNUE_BIN = CARGO_TARGET_DIR / "release" / f"nnue{EXE_SUFFIX}"
REFERENCE_REF = ""
REFERENCE_BIN = Path()


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


VALIDATION_SAMPLES = env_int("STEINBEISSER_TRAIN_VALIDATION_SAMPLES", 10_000)
MIN_TRAIN_INCREMENT = env_int(
    "STEINBEISSER_TRAIN_MIN_INCREMENT",
    env_int("STEINBEISSER_TRAIN_CYCLE_SAMPLES", 500_000),
)
INITIAL_TRAIN_SAMPLES = env_int("STEINBEISSER_TRAIN_INITIAL_SAMPLES", MIN_TRAIN_INCREMENT)
GENERATION_WORKERS = env_int("STEINBEISSER_TRAIN_PARALLEL_GAMES", 15)
CORE_BUDGET = env_int("STEINBEISSER_TRAIN_CORES", GENERATION_WORKERS)
TOURNAMENT_PARALLEL_MATCHES = env_int(
    "STEINBEISSER_TRAIN_TOURNAMENT_PARALLEL_MATCHES",
    max(1, CORE_BUDGET),
)
NNUE_TRAIN_THREADS = env_int("STEINBEISSER_NNUE_THREADS", env_int("STEINBEISSER_TRAIN_THREADS", 1))
NNUE_LOADER_WORKERS = env_int(
    "STEINBEISSER_NNUE_LOADER_WORKERS",
    env_int("STEINBEISSER_LOADER_WORKERS", max(1, min(CORE_BUDGET, os.cpu_count() or CORE_BUDGET))),
)
SELFPLAY_MS = env_int("STEINBEISSER_TRAIN_SELFPLAY_MS", 100)
MAX_ABS_SCORE = env_int("STEINBEISSER_TRAIN_MAX_ABS_SCORE", 3500)
MATCH_MS = env_int("STEINBEISSER_TRAIN_MATCH_MS", 5)
MATCH_SINGLE_POSITION_COUNT = env_int("STEINBEISSER_TRAIN_MATCH_SINGLE_POSITIONS", 60)
MATCH_RANDOM_POSITION_COUNT = env_int("STEINBEISSER_TRAIN_MATCH_RANDOM_POSITIONS", 60)
MATCH_GAMES = env_int(
    "STEINBEISSER_TRAIN_MATCH_GAMES",
    2 * (MATCH_SINGLE_POSITION_COUNT + MATCH_RANDOM_POSITION_COUNT),
)
SCREEN_CHECKPOINTS = env_int("STEINBEISSER_TRAIN_SCREEN_CHECKPOINTS", 3)
TARGET_TRAIN_SAMPLES = env_int("STEINBEISSER_TRAIN_TARGET_SAMPLES", 15_000_000)
TOURNAMENT_GAMES_PER_ENGINE = env_int(
    "STEINBEISSER_TRAIN_TOURNAMENT_GAMES_PER_ENGINE",
    10_000,
)
TOURNAMENT_GAMES_PER_PAIRING_OVERRIDE = env_int(
    "STEINBEISSER_TRAIN_TOURNAMENT_GAMES_PER_PAIRING",
    0,
)
TOURNAMENT_OPENINGS = Path(
    os.environ.get("STEINBEISSER_TRAIN_TOURNAMENT_OPENINGS", WORK_DIR / "tournament_openings.fen")
)
OPENING_CONFIG = OpeningBookConfig(
    open_book=OPENINGS,
    match_openings=MATCH_OPENINGS,
    match_openings_override=MATCH_OPENINGS_OVERRIDE,
    single_openings_dir=MATCH_SINGLE_OPENINGS_DIR,
    random_openings=MATCH_RANDOM_OPENINGS,
    match_single_count=MATCH_SINGLE_POSITION_COUNT,
    match_random_count=MATCH_RANDOM_POSITION_COUNT,
    tournament_openings=TOURNAMENT_OPENINGS,
    tournament_openings_explicit=bool(os.environ.get("STEINBEISSER_TRAIN_TOURNAMENT_OPENINGS")),
)
MAX_CYCLES = env_int("STEINBEISSER_TRAIN_MAX_CYCLES", 30)
GENERATION_CHUNK_SAMPLES = env_int(
    "STEINBEISSER_TRAIN_GENERATION_CHUNK_SAMPLES",
    min(MIN_TRAIN_INCREMENT, 50_000),
)
MAX_TRAIN_INCREMENT = env_int("STEINBEISSER_TRAIN_MAX_INCREMENT", 0)
GENERATION_BACKLOG_SAMPLES = env_int(
    "STEINBEISSER_TRAIN_GENERATION_BACKLOG_SAMPLES",
    max(GENERATION_CHUNK_SAMPLES, MIN_TRAIN_INCREMENT),
)
SNAPSHOT_POLL_SECONDS = env_float("STEINBEISSER_TRAIN_POLL_SECONDS", 10.0)
PROGRESS_INTERVAL_SECONDS = env_float("STEINBEISSER_TRAIN_PROGRESS_SECONDS", 60.0)
STATUS_OUTPUT = env_int("STEINBEISSER_TRAIN_STATUS_OUTPUT", 0) > 0
TRAIN_EPOCH_COUNT = env_int("STEINBEISSER_TRAIN_EPOCHS", 100)
TRAIN_CONFIG = TrainConfig(
    validation_samples=VALIDATION_SAMPLES,
    min_train_increment=MIN_TRAIN_INCREMENT,
    initial_train_samples=INITIAL_TRAIN_SAMPLES,
    max_train_increment=MAX_TRAIN_INCREMENT,
    max_cycles=MAX_CYCLES,
    target_train_samples=TARGET_TRAIN_SAMPLES,
)
MODEL_SCHEMA = current_feature_schema()
MODEL_FEATURES = MODEL_SCHEMA.name
MODEL_INPUT_COUNT = MODEL_SCHEMA.input_count
MODEL_MAX_ACTIVE_FEATURES = MODEL_SCHEMA.max_active_features
MODEL_ARCHITECTURE = f"{MODEL_INPUT_COUNT},84,50,1"
CANDIDATE_TARGET_DIR = CARGO_TARGET_DIR / "candidate-native"
GENERATOR_CONFIG = GeneratorConfig(
    selfplay_bin=SELFPLAY_BIN,
    repo=REPO,
    openings=OPENINGS,
    shard_dir=SHARD_DIR,
    log_path=LOG_PATH,
    chunk_samples=GENERATION_CHUNK_SAMPLES,
    workers=GENERATION_WORKERS,
    selfplay_ms=SELFPLAY_MS,
    max_abs_score=MAX_ABS_SCORE,
    backlog_samples=GENERATION_BACKLOG_SAMPLES,
    progress_interval_seconds=PROGRESS_INTERVAL_SECONDS,
)
EMIT_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()

SCRATCH_DIRS = (
    BIN_DIR,
    SHARD_DIR,
    WORK_DIR,
    SOURCE_BIN_DIR,
    REFERENCE_SOURCE_DIR,
    CARGO_TARGET_DIR,
    TRAINER_ARTIFACT_ROOT,
    DATASET_CACHE_DIR,
)
FINAL_OUTPUT_DIRS = (CANONICAL_TRAINING_DIR,)
RUN_DIRS = SCRATCH_DIRS + RUN_OUTPUT_DIRS + FINAL_OUTPUT_DIRS


def usage() -> str:
    return (
        "usage: train [--clean]\n"
        "   or: python3 nnue/train.py [--clean]\n\n"
        "  --clean   delete the configured /tmp run directory before starting\n\n"
        "nnue/train.py always uses the latest GitHub release as the fixed reference."
    )


def parse_launcher_args(argv: list[str]) -> bool:
    clean = False
    for arg in argv[1:]:
        if arg == "--clean":
            clean = True
        elif arg in {"-h", "--help"}:
            print(usage(), flush=True)
            raise SystemExit(0)
        else:
            message = f"unknown option {arg}\n\n{usage()}"
            raise SystemExit(message)
    return clean


def launcher_process_needles(argv: list[str]) -> list[str]:
    needles: set[str] = {str(Path(__file__).resolve())}
    raw_launcher = Path(argv[0]) if argv and argv[0] else None
    if raw_launcher is not None:
        if raw_launcher.is_absolute():
            needles.add(str(raw_launcher))
        else:
            needles.add(str((Path.cwd() / raw_launcher).resolve(strict=False)))
        try:
            needles.add(str(raw_launcher.resolve(strict=False)))
        except OSError:
            pass
    cargo_launcher = Path.home() / ".cargo/bin/train"
    needles.add(str(cargo_launcher))
    try:
        needles.add(str(cargo_launcher.resolve(strict=False)))
    except OSError:
        pass
    return sorted(needle for needle in needles if needle)


def current_process_family_pids() -> set[int]:
    current_pid = os.getpid()
    ignored = {current_pid}
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return ignored
    if output.returncode != 0:
        return ignored

    parents: dict[int, int] = {}
    for line in output.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        try:
            parents[int(parts[0])] = int(parts[1])
        except ValueError:
            continue

    pid = current_pid
    while True:
        parent = parents.get(pid)
        if parent is None or parent <= 0 or parent in ignored:
            break
        ignored.add(parent)
        pid = parent
    return ignored


def other_train_processes(argv: list[str]) -> list[tuple[int, str]]:
    ignored_pids = current_process_family_pids()
    needles = launcher_process_needles(argv)
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if output.returncode != 0:
        return []

    processes: list[tuple[int, str]] = []
    for line in output.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in ignored_pids:
            continue
        command = parts[1]
        if command_is_launcher_shell(command):
            continue
        if any(needle in command for needle in needles):
            processes.append((pid, command))
    return processes


def command_is_launcher_shell(command: str) -> bool:
    stripped = command.strip()
    return (
        stripped.startswith("/bin/zsh -c ")
        or stripped.startswith("/bin/zsh -lc ")
        or stripped.startswith("zsh -c ")
        or stripped.startswith("zsh -lc ")
        or stripped.startswith("/bin/bash -c ")
        or stripped.startswith("/bin/bash -lc ")
        or stripped.startswith("bash -c ")
        or stripped.startswith("bash -lc ")
        or stripped.startswith("login ")
        or stripped.startswith("screen ")
        or stripped.startswith("SCREEN ")
        or " screen -dmS " in stripped
        or " pgrep -laf " in stripped
    )


def last_result_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 256 * 1024))
        tail = handle.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if re.match(r"^\d\d:\d\d (cycle=\d+|tournament=|training_data_exported\b)", line):
            return line
    return None


def last_progress_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    progress_markers = (
        "dataset_load=",
        "epoch ",
        "selfplay match",
        "selfplay generate",
        "cargo build",
        "training=",
        "screening=",
        "build_candidate=",
        "match=",
        "tournament",
        "training_data_export",
    )
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 256 * 1024))
        tail = handle.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line and any(marker in line for marker in progress_markers):
            return line
    return None


def last_foreground_progress_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    progress_markers = (
        "$ embedded nnue trainer",
        "dataset_load=",
        "epoch ",
        "training=",
        "screening=",
        "build_candidate=",
        "match=",
        "tournament",
        "training_data_export",
    )
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 64 * 1024 * 1024))
        tail = handle.read().decode("utf-8", errors="replace")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line and any(marker in line for marker in progress_markers):
            return line
    return None


def progress_cycle(line: str | None) -> int | None:
    if line is None:
        return None
    for pattern in (r"\bcycle=(\d+)\b", r"\bcycle0*(\d+)\b", r"\bcycle(\d+)_fen\b"):
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    return None


def progress_train_samples(line: str | None) -> int | None:
    if line is None:
        return None
    for pattern in (r"\btrain=(\d+)\b", r"\btrain_samples=(\d+)\b"):
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    return None


def should_report_foreground(foreground: str | None, latest: str | None, active_phase: str | None) -> bool:
    if foreground is None or foreground == latest:
        return False
    if active_phase is not None and not active_phase.startswith("active_phase: background_generate "):
        foreground_cycle = progress_cycle(foreground)
        latest_cycle = progress_cycle(latest)
        return (
            foreground_cycle is not None
            and latest_cycle is not None
            and foreground_cycle > latest_cycle
        )
    foreground_cycle = progress_cycle(foreground)
    latest_cycle = progress_cycle(latest)
    if foreground_cycle is not None and latest_cycle is not None:
        return foreground_cycle > latest_cycle
    foreground_samples = progress_train_samples(foreground)
    latest_samples = progress_train_samples(latest)
    if foreground_samples is not None and latest_samples is not None:
        return foreground_samples > latest_samples
    if active_phase is None:
        return True
    return active_phase.startswith("active_phase: background_generate ")


def run_root_process_markers() -> list[str]:
    roots = {str(RUN_ROOT), str(RUN_ROOT.resolve(strict=False))}
    expanded_roots: set[str] = set()
    for root in roots:
        expanded_roots.add(root)
        if root.startswith("/private/tmp/"):
            expanded_roots.add(root.removeprefix("/private"))
    markers: list[str] = []
    for root in sorted(expanded_roots):
        markers.extend(
            [
                f"{root}/bin/",
                f"{root}/source-bin/",
                f"{root}/reference-sources/",
                f"{root}/cargo-target/",
                f"{root}/trainer-artifacts/",
            ]
        )
    return markers


def run_root_processes() -> list[tuple[int, str]]:
    ignored_pids = current_process_family_pids()
    markers = run_root_process_markers()
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if output.returncode != 0:
        return []

    processes: list[tuple[int, str]] = []
    for line in output.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in ignored_pids:
            continue
        command = parts[1]
        if any(marker in command for marker in markers):
            processes.append((pid, command))
    return processes


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_run_root_processes() -> None:
    processes = run_root_processes()
    if not processes:
        return
    pid_text = ",".join(str(pid) for pid, _command in processes)
    print(f"clean: stopping stale run processes pid={pid_text}", flush=True)
    for pid, _command in processes:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not any(process_is_alive(pid) for pid, _command in processes):
            return
        time.sleep(0.1)
    for pid, _command in processes:
        if not process_is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def report_existing_run(processes: list[tuple[int, str]], clean: bool) -> None:
    pid_text = ",".join(str(pid) for pid, _command in processes)
    prefix = "train is already running"
    if clean:
        prefix += "; not cleaning a live run directory"
    print(f"{prefix} pid={pid_text}", flush=True)
    active_phase = current_run_phase_line()
    latest = last_result_line(SCREEN_LOG_PATH)
    foreground = last_foreground_progress_line(LOG_PATH)
    if should_report_foreground(foreground, latest, active_phase):
        print(f"foreground: {foreground}", flush=True)
    if active_phase is not None:
        print(active_phase, flush=True)
    progress = None if active_phase is not None else last_progress_line(LOG_PATH)
    if progress is not None:
        print(f"progress: {progress}", flush=True)
    if latest is not None:
        print(f"latest_result: {latest}", flush=True)
    print(f"log: {SCREEN_LOG_PATH}", flush=True)


def current_run_phase_line() -> str | None:
    processes = run_root_processes()
    for phase in ("selfplay match", "cargo build", "selfplay generate"):
        for _pid, command in processes:
            if phase in command:
                summary = summarize_phase_command(command)
                return f"active_phase: {summary}"
    return None


def summarize_phase_command(command: str) -> str:
    if "selfplay match" in command:
        return (
            "match "
            f"local={_basename_arg(command, '--local-bin')} "
            f"baseline={_basename_arg(command, '--github-bin')} "
            f"pairs={_arg_after(command, '--pairs') or '?'} "
            f"time_ms={_arg_after(command, '--time') or '?'}"
        )
    if "selfplay generate" in command:
        return (
            "background_generate "
            f"target_samples={_arg_after(command, '--target-samples') or '?'} "
            f"parallel_games={_arg_after(command, '--parallel-games') or '?'} "
            f"time_ms={_arg_after(command, '--time') or '?'}"
        )
    if "cargo build" in command:
        manifest = _arg_after(command, "--manifest-path")
        return f"cargo_build manifest={Path(manifest).name if manifest else '?'}"
    return command


def _arg_after(command: str, flag: str) -> str | None:
    parts = command.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _basename_arg(command: str, flag: str) -> str:
    value = _arg_after(command, flag)
    return Path(value).name if value else "?"


def clean_run_root() -> None:
    target = RUN_ROOT.resolve(strict=False)
    forbidden = {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path(os.environ.get("TMPDIR", "/tmp")).resolve(),
        Path.home().resolve(),
        REPO.resolve(),
        NNUE_DIR.resolve(),
        WORKSPACE.resolve(),
    }
    if target in forbidden:
        raise SystemExit(f"refusing to clean unsafe run directory: {target}")
    for root in (REPO.resolve(), WORKSPACE.resolve()):
        try:
            target.relative_to(root)
        except ValueError:
            continue
        raise SystemExit(f"refusing to clean repository path: {target}")
    stop_run_root_processes()
    shutil.rmtree(target, ignore_errors=True)
    SCREEN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREEN_LOG_PATH.write_text("", encoding="utf-8")


def github_repo_slug() -> str:
    output = subprocess.run(
        ["git", "-C", REPO, "remote", "get-url", "origin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if output.returncode != 0:
        raise SystemExit(f"failed to read origin remote: {output.stderr.strip()}")
    remote = output.stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?$", remote)
    if not match:
        raise SystemExit(f"origin is not a GitHub repository URL: {remote}")
    return f"{match.group(1)}/{match.group(2)}"


def latest_github_release_ref() -> str:
    slug = github_repo_slug()
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/releases/latest",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "steinbeisser-train",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            release = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as error:
        append_log(f"# latest release lookup failed, falling back to tags: {error}")
        return latest_github_tag_ref()
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise SystemExit(f"latest GitHub release for {slug} has no tag_name")
    return tag


def latest_github_tag_ref() -> str:
    output = subprocess.run(
        ["git", "-C", REPO, "ls-remote", "--tags", "--refs", "origin", "v*"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if output.returncode != 0:
        raise SystemExit(f"failed to list GitHub tags: {output.stderr.strip()}")
    tags = [
        line.rsplit("/", 1)[-1]
        for line in output.stdout.splitlines()
        if line.strip() and "/" in line
    ]
    if not tags:
        raise SystemExit("no v* release tags found on origin")
    return max(tags, key=version_key)


def version_key(tag: str) -> tuple[object, ...]:
    parts: list[object] = []
    for part in re.split(r"([0-9]+)", tag.lstrip("v")):
        if not part:
            continue
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def next_release_dir_name(reference_ref: str) -> str:
    compact_tag = re.fullmatch(r"v(\d+)$", reference_ref)
    if compact_tag:
        return f"v{int(compact_tag.group(1)) + 1}"
    dotted_tag = re.fullmatch(r"v(\d+)\.(\d+)$", reference_ref)
    if dotted_tag:
        major, minor = dotted_tag.groups()
        return f"v{major}{int(minor) + 1}"
    raise SystemExit(f"cannot derive training directory from release tag {reference_ref!r}")


def configure_training_dirs() -> None:
    global CANONICAL_TRAINING_DIR, OUTPUT_DIR, POSITIVE_NET_DIR
    global TOURNAMENT_RESULT_DIR, RUN_OUTPUT_DIRS, FINAL_OUTPUT_DIRS, RUN_DIRS

    CANONICAL_TRAINING_DIR = REPO / "data/training" / next_release_dir_name(REFERENCE_REF)
    OUTPUT_DIR = RUN_ROOT / "outputs"
    POSITIVE_NET_DIR = OUTPUT_DIR / "networks"
    TOURNAMENT_RESULT_DIR = OUTPUT_DIR / "tournament-results"
    RUN_OUTPUT_DIRS = (OUTPUT_DIR, POSITIVE_NET_DIR, TOURNAMENT_RESULT_DIR)
    FINAL_OUTPUT_DIRS = (CANONICAL_TRAINING_DIR,)
    RUN_DIRS = SCRATCH_DIRS + RUN_OUTPUT_DIRS + FINAL_OUTPUT_DIRS


def resolve_reference_ref() -> str:
    return latest_github_release_ref()


def configure_release_ref() -> None:
    global REFERENCE_REF, REFERENCE_BIN
    REFERENCE_REF = resolve_reference_ref()
    REFERENCE_BIN = BIN_DIR / f"steinbeisser_{slug(REFERENCE_REF)}{EXE_SUFFIX}"
    configure_training_dirs()


def append_log(text: str) -> None:
    with LOG_LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")


def record_failure(message: object) -> None:
    text = str(message).strip()
    if not text:
        return
    append_log(f"# train failed: {text}")
    try:
        state = read_json(STATE_PATH, {})
        if isinstance(state, dict):
            state["last_error"] = text
            state["last_error_time"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            write_state(state)
    except Exception as error:
        append_log(f"# failed to write last_error state: {error}")

    print_and_log(f"train=failed error={text} log={LOG_PATH}")


def append_screen_log(text: str) -> None:
    SCREEN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCREEN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")


def print_and_log(text: str) -> None:
    append_screen_log(text)
    print(text, flush=True)


def emit(line: str) -> None:
    text = f"{time.strftime('%H:%M')} {line}"
    with EMIT_LOCK:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print_and_log(text)


def emit_raw(line: str) -> None:
    with EMIT_LOCK:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print_and_log(line)


def emit_status(line: str) -> None:
    if STATUS_OUTPUT:
        append_log(f"# status {time.strftime('%H:%M')} {line}")


def emit_progress(line: str) -> None:
    text = f"{time.strftime('%H:%M')} {line}"
    append_log(f"# progress {text}")


def command_text(cmd: list[str | Path]) -> str:
    return " ".join(str(part) for part in cmd)


def format_timing_seconds(seconds: float | None) -> str:
    return "NA" if seconds is None else f"{seconds:.1f}"


def run(cmd: list[str | Path], *, cwd: Path = WORKSPACE, env: dict[str, str] | None = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {command_text(cmd)}\n")
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(f"command failed with status {proc.returncode}; see {LOG_PATH}")


def run_capture_status(
    cmd: list[str | Path], *, cwd: Path = WORKSPACE, env: dict[str, str] | None = None
) -> tuple[int, str]:
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    append_log(f"$ {command_text(cmd)}\n{proc.stdout}")
    return proc.returncode, proc.stdout


def single_core_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STEINBEISSER_TRAIN_THREADS": str(NNUE_TRAIN_THREADS),
            "STEINBEISSER_LOADER_WORKERS": str(NNUE_LOADER_WORKERS),
            "STEINBEISSER_TRAIN_ARTIFACT_ROOT": str(TRAINER_ARTIFACT_ROOT),
            "STEINBEISSER_NNUE_CLI": str(NNUE_BIN),
            "STEINBEISSER_NNUE_FEATURE_SET": MODEL_FEATURES,
            "STEINBEISSER_NNUE_ARCHITECTURE": MODEL_ARCHITECTURE,
            "STEINBEISSER_NNUE_EPOCHS": str(TRAIN_EPOCH_COUNT),
            "CARGO_TARGET_DIR": str(CARGO_TARGET_DIR),
            "OMP_NUM_THREADS": str(NNUE_TRAIN_THREADS),
            "OPENBLAS_NUM_THREADS": str(NNUE_TRAIN_THREADS),
            "MKL_NUM_THREADS": str(NNUE_TRAIN_THREADS),
            "VECLIB_MAXIMUM_THREADS": str(NNUE_TRAIN_THREADS),
            "NUMEXPR_NUM_THREADS": str(NNUE_TRAIN_THREADS),
            "XLA_FLAGS": (
                "--xla_cpu_multi_thread_eigen=false "
                f"intra_op_parallelism_threads={NNUE_TRAIN_THREADS}"
            ),
        }
    )
    return env


def run_trainer(env: dict[str, str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    old_env = os.environ.copy()
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log:
        log.write("$ embedded nnue trainer\n")
        try:
            os.environ.clear()
            os.environ.update(env)
            with redirect_stdout(log), redirect_stderr(log):
                trainer_main([])
        except SystemExit as error:
            code = error.code
            if code not in (None, 0):
                raise SystemExit(f"embedded trainer failed with status {code}; see {LOG_PATH}") from error
        finally:
            os.environ.clear()
            os.environ.update(old_env)


def cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(CARGO_TARGET_DIR)
    return env


def setup() -> None:
    emit_status(
        "setup=start "
        f"reference={REFERENCE_REF} "
        f"selfplay_ms={SELFPLAY_MS} "
        f"screen_ms={MATCH_MS} "
        f"screen_games={MATCH_GAMES} "
        f"parallel_games={GENERATION_WORKERS} "
        f"tournament_parallel_matches={TOURNAMENT_PARALLEL_MATCHES} "
        f"tournament_target_games_per_engine={TOURNAMENT_GAMES_PER_ENGINE} "
        f"loader_workers={NNUE_LOADER_WORKERS} "
        f"train_threads={NNUE_TRAIN_THREADS} "
        f"min_train_increment={MIN_TRAIN_INCREMENT} "
        f"generation_backlog_samples={GENERATION_BACKLOG_SAMPLES} "
        f"target_train_positions={TARGET_TRAIN_SAMPLES} "
        f"max_abs_score={MAX_ABS_SCORE} "
        f"dataset_cache={DATASET_CACHE_DIR} "
        f"dataset_cache_format={dataset_cache_format()} "
        f"dataset_cache_mmap={dataset_cache_mmap_mode() is not None} "
        f"output_dir={OUTPUT_DIR}"
    )
    if MATCH_GAMES <= 0 or MATCH_GAMES % 2 != 0:
        raise SystemExit("STEINBEISSER_TRAIN_MATCH_GAMES must be a positive even number")
    if TOURNAMENT_PARALLEL_MATCHES <= 0:
        raise SystemExit("STEINBEISSER_TRAIN_TOURNAMENT_PARALLEL_MATCHES must be positive")
    TRAIN_CONFIG.validate()
    if GENERATION_BACKLOG_SAMPLES < 0:
        raise SystemExit("STEINBEISSER_TRAIN_GENERATION_BACKLOG_SAMPLES must be non-negative")
    if SCREEN_CHECKPOINTS <= 0:
        raise SystemExit("STEINBEISSER_TRAIN_SCREEN_CHECKPOINTS must be positive")
    if TOURNAMENT_GAMES_PER_ENGINE <= 0:
        raise SystemExit(
            "STEINBEISSER_TRAIN_TOURNAMENT_GAMES_PER_ENGINE must be positive"
        )
    if TOURNAMENT_GAMES_PER_PAIRING_OVERRIDE < 0:
        raise SystemExit(
            "STEINBEISSER_TRAIN_TOURNAMENT_GAMES_PER_PAIRING must be non-negative"
        )
    if MAX_ABS_SCORE < 0:
        raise SystemExit("STEINBEISSER_TRAIN_MAX_ABS_SCORE must be non-negative")
    os.environ["STEINBEISSER_TRAIN_ARTIFACT_ROOT"] = str(TRAINER_ARTIFACT_ROOT)
    os.environ["CARGO_TARGET_DIR"] = str(CARGO_TARGET_DIR)
    for path in RUN_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    for required in [OPENINGS, NNUE_MANIFEST]:
        if not required.exists():
            raise SystemExit(f"missing required path: {required}")
    emit_status("setup=openings")
    prepare_match_openings(OPENING_CONFIG)
    emit_status("setup=build_rust_tools")
    run(
        [
            "cargo",
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            NNUE_MANIFEST,
            "--bin",
            "nnue",
            "--bin",
            "nnue-selfplay",
        ],
        env=cargo_env(),
    )
    emit_status(f"setup=build_reference reference={REFERENCE_REF}")
    run(
        [
            SELFPLAY_BIN,
            "build-ref",
            "--repo",
            REPO,
            "--github-ref",
            REFERENCE_REF,
            "--github-bin",
            REFERENCE_BIN,
        ],
        cwd=REPO,
    )
    emit_status(f"setup=done reference_bin={REFERENCE_BIN}")


def write_cycle_corpus_rust(cycle: int, max_samples: int | None) -> tuple[int, Path] | None:
    if not NNUE_BIN.is_file():
        return None
    command: list[str | Path] = [
        NNUE_BIN,
        "corpus-build",
        "--shards",
        SHARD_DIR,
        "--work-dir",
        WORK_DIR,
        "--cycle",
        str(cycle),
        "--validation-samples",
        str(VALIDATION_SAMPLES),
        "--max-abs-score",
        str(MAX_ABS_SCORE),
        "--feature-set",
        MODEL_FEATURES,
        "--input-count",
        str(MODEL_INPUT_COUNT),
        "--max-active-features",
        str(MODEL_MAX_ACTIVE_FEATURES),
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    status, output = run_capture_status(command, cwd=REPO)
    if status != 0:
        raise SystemExit(f"rust corpus-build failed with status {status}; see {LOG_PATH}\n{output}")
    try:
        report = json.loads(output)
        if not isinstance(report, dict):
            raise ValueError("corpus-build returned a non-object JSON payload")
        return int(report.get("samples", 0)), Path(str(report["corpus_dir"]))
    except Exception as error:
        raise SystemExit(f"rust corpus-build output parse failed: {error}") from error


def write_cycle_corpus(
    cycle: int,
    max_samples: int | None = None,
) -> tuple[int, Path]:
    rust_result = write_cycle_corpus_rust(cycle, max_samples)
    if rust_result is None:
        raise SystemExit("Rust corpus-build is required after setup but was not available")
    return rust_result


def generate_until_trainable(
    cycle: int,
    previous_train_samples: int,
    generator: ContinuousGenerator,
) -> tuple[int, Path]:
    minimum_total_samples = VALIDATION_SAMPLES + required_train_samples(previous_train_samples)
    generator.set_target(minimum_total_samples)
    last_wait_emit = 0.0
    while True:
        generator.check()
        samples, corpus_dir = write_cycle_corpus(cycle, minimum_total_samples)
        generator.note_unique_samples(samples)
        if samples >= minimum_total_samples:
            emit_status(
                "corpus=ready "
                f"cycle={cycle} "
                f"unique_samples={samples} "
                f"train_samples={samples - VALIDATION_SAMPLES} "
                f"val_samples={VALIDATION_SAMPLES} "
                f"dir={corpus_dir}"
            )
            emit_progress(
                "corpus=ready "
                f"cycle={cycle} "
                f"unique_samples={samples}/{minimum_total_samples} "
                f"train_samples={samples - VALIDATION_SAMPLES} "
                f"val_samples={VALIDATION_SAMPLES}"
            )
            return samples, corpus_dir
        now = time.monotonic()
        if last_wait_emit == 0.0 or now - last_wait_emit >= PROGRESS_INTERVAL_SECONDS:
            emit_status(
                "generation=waiting "
                f"cycle={cycle} "
                f"shards={sum(1 for _ in SHARD_DIR.glob('*.sbin'))} "
                f"unique_samples={samples} "
                f"required_unique_samples={minimum_total_samples}"
            )
            emit_progress(
                "generation=waiting "
                f"cycle={cycle} "
                f"shards={sum(1 for _ in SHARD_DIR.glob('*.sbin'))} "
                f"unique_samples={samples}/{minimum_total_samples}"
            )
            last_wait_emit = now
        deadline = time.monotonic() + SNAPSHOT_POLL_SECONDS
        while time.monotonic() < deadline:
            generator.check()
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def train_increment_for(previous_train_samples: int) -> int:
    return TRAIN_CONFIG.train_increment(previous_train_samples)


def required_train_samples(previous_train_samples: int) -> int:
    return TRAIN_CONFIG.required_train_samples(previous_train_samples)


@dataclass(frozen=True)
class CycleTrainingArtifact:
    experiment_dir: Path
    corpus_dir: Path
    train_samples: int
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class CycleResult:
    cycle: int
    corpus_dir: Path
    experiment_dir: Path
    train_samples: int
    rows: list[dict[str, object]]
    selected_models: list[Path]
    selected_bins: list[Path]
    release_matches: list[MatchResult]
    positive_records: list[dict[str, object]]
    reused_training: bool
    corpus_seconds: float | None
    training_seconds: float | None
    build_seconds: float | None
    release_match_seconds: float | None
    total_seconds: float

    @property
    def model(self) -> Path:
        return self.selected_models[0]

    @property
    def source_bin(self) -> Path:
        return self.selected_bins[0]


def ranked_quantized_rows(experiment_dir: Path) -> list[dict[str, object]]:
    metrics_path = experiment_dir / "metrics.json"
    metrics = read_json(metrics_path)
    if not isinstance(metrics, dict):
        raise SystemExit(f"{metrics_path} must contain a JSON object")
    rows = [
        row
        for row in metrics.get("history", [])
        if isinstance(row, dict) and row.get("quantized_val_loss") is not None
    ]
    if not rows:
        raise SystemExit(f"no quantized validation history in {metrics_path}")
    return sorted(rows, key=lambda row: (float(row["quantized_val_loss"]), int(row["epoch"])))


def screen_checkpoint_model_path(
    experiment_dir: Path,
    row: dict[str, object],
    rank: int,
) -> Path:
    raw_path = str(row.get("screen_checkpoint_file") or "")
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            return path
    epoch = int(row["epoch"])
    return experiment_dir / "screen_checkpoints" / f"rank_{rank:02}_epoch_{epoch:04}_model.nnq"


def completed_cycle_artifact(
    cycle: int,
    minimum_train_samples: int,
) -> CycleTrainingArtifact | None:
    cycle_root = TRAINER_ARTIFACT_ROOT / "fen-cycles"
    candidates = sorted(
        cycle_root.glob(f"cycle{cycle:04}_*"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for experiment_dir in candidates:
        metrics_path = experiment_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = read_json(metrics_path)
            if not isinstance(metrics, dict):
                continue
            train_samples = int(metrics.get("train_dataset_size") or 0)
            manifest_path = Path(str(metrics.get("manifest_path") or ""))
            rows = ranked_quantized_rows(experiment_dir)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if train_samples < minimum_train_samples:
            continue
        corpus_dir = manifest_path.parent
        if not all((corpus_dir / name).is_file() for name in ("train.sbin", "val.sbin", "manifest.json")):
            continue
        selected_rows = rows[:SCREEN_CHECKPOINTS]
        checkpoints = [
            screen_checkpoint_model_path(experiment_dir, row, rank)
            for rank, row in enumerate(selected_rows, start=1)
        ]
        if len(selected_rows) < SCREEN_CHECKPOINTS or not all(path.is_file() for path in checkpoints):
            continue
        return CycleTrainingArtifact(
            experiment_dir=experiment_dir,
            corpus_dir=corpus_dir,
            train_samples=train_samples,
            rows=selected_rows,
        )
    return None


def train_cycle(cycle: int, corpus_dir: Path, train_samples: int) -> tuple[Path, list[dict[str, object]]]:
    experiment_dir = TRAINER_ARTIFACT_ROOT / "fen-cycles" / f"cycle{cycle:04}_{time.time_ns()}"
    emit_status(
        "training=start "
        f"cycle={cycle} "
        f"train_samples={train_samples} "
        f"epochs={TRAIN_EPOCH_COUNT} "
        f"experiment={experiment_dir}"
    )
    env = single_core_env()
    env.update(
        {
            "STEINBEISSER_NNUE_TRAIN_PATH": str(corpus_dir / "train.sbin"),
            "STEINBEISSER_NNUE_VAL_PATH": str(corpus_dir / "val.sbin"),
            "STEINBEISSER_NNUE_MANIFEST_PATH": str(corpus_dir / "manifest.json"),
            "STEINBEISSER_NNUE_OUTPUT_DIR": str(experiment_dir),
            "STEINBEISSER_NNUE_DATASET_CACHE_DIR": str(DATASET_CACHE_DIR),
        }
    )
    run_trainer(env)
    ranked = ranked_quantized_rows(experiment_dir)
    emit_status(
        "training=done "
        f"cycle={cycle} "
        f"screen_epochs={row_epochs(ranked[:SCREEN_CHECKPOINTS])} "
        f"screen_qval_losses={row_qvals(ranked[:SCREEN_CHECKPOINTS])}"
    )
    return experiment_dir, ranked[:SCREEN_CHECKPOINTS]


def screen_model_path(experiment_dir: Path, row: dict[str, object], rank: int) -> Path:
    model = screen_checkpoint_model_path(experiment_dir, row, rank)
    if not model.is_file():
        raise SystemExit(f"missing selected checkpoint model: {model}")
    return model


def cycle_candidate_id(cycle: int, rank: int = 1) -> str:
    return f"cycle{cycle:04}_q{rank:02}"


def run_json_command(cmd: list[str | Path], *, label: str) -> object:
    status, output = run_capture_status(cmd, cwd=REPO)
    if status != 0:
        raise SystemExit(f"{label} failed with status {status}; see {LOG_PATH}\n{output}")
    try:
        return json.loads(output or "null")
    except json.JSONDecodeError as error:
        raise SystemExit(f"{label} returned invalid JSON: {error}; see {LOG_PATH}") from error


def build_reference_net_engine(cycle: int, model: Path, rank: int = 1) -> Path:
    candidate_id = cycle_candidate_id(cycle, rank)
    target = SOURCE_BIN_DIR / f"{candidate_id}{EXE_SUFFIX}"
    emit_status(
        "build_candidate=start "
        f"candidate={candidate_id} "
        f"model={model.name}"
    )
    report = run_json_command(
        [
            NNUE_BIN,
            "materialize-candidate",
            "--repo",
            REPO,
            "--reference-ref",
            REFERENCE_REF,
            "--model",
            model,
            "--source-dir",
            REFERENCE_SOURCE_DIR / candidate_id,
            "--target",
            target,
            "--target-dir",
            CANDIDATE_TARGET_DIR / candidate_id,
            "--candidate-id",
            candidate_id,
        ],
        label="rust materialize-candidate",
    )
    if not isinstance(report, dict):
        raise SystemExit("rust materialize-candidate returned a non-object JSON payload")
    built = Path(str(report.get("binary") or target))
    if not built.is_file():
        raise SystemExit(f"candidate build did not produce {built}")
    emit_status(
        "build_candidate=done "
        f"candidate={candidate_id} "
        f"bin={built}"
    )
    return built


def run_selfplay_match(
    candidate: Path,
    baseline: Path,
    openings: Path = MATCH_OPENINGS,
    games: int = MATCH_GAMES,
    time_ms: int = MATCH_MS,
    label: str | None = None,
    allow_local_failure: bool = True,
    allow_baseline_failure: bool = False,
) -> MatchResult:
    if label is not None:
        emit_status(
            "match=start "
            f"{label} "
            f"games={games} "
            f"time_ms={time_ms} "
            f"openings={openings}"
        )
    cmd = [
        NNUE_BIN,
        "screen-match",
        "--selfplay-bin",
        SELFPLAY_BIN,
        "--repo",
        REPO,
        "--candidate",
        candidate,
        "--baseline",
        baseline,
        "--openings",
        openings,
        "--games",
        str(games),
        "--time-ms",
        str(time_ms),
        "--seed",
        str(time.time_ns() & ((1 << 63) - 1)),
        "--github-ref",
        "baseline",
    ]
    if allow_local_failure:
        cmd.append("--allow-local-failure")
    if allow_baseline_failure:
        cmd.append("--allow-baseline-failure")
    payload = run_json_command(cmd, label="rust screen-match")
    if not isinstance(payload, dict):
        raise SystemExit("rust screen-match returned a non-object JSON payload")
    result = MatchResult(
        wins=int(payload["wins"]),
        draws=int(payload["draws"]),
        losses=int(payload["losses"]),
        elo=float(payload["elo"]),
        elo_lower=float(payload["elo_lower"]),
        elo_upper=float(payload["elo_upper"]),
    )
    if label is not None:
        match_status = "match=forfeit " if payload.get("forfeit") else "match=done "
        emit_status(
            match_status
            + f"{label} "
            + f"wdl={result.wins}-{result.draws}-{result.losses} "
            + f"elo={result.elo:+.2f}[{result.elo_lower:+.2f},{result.elo_upper:+.2f}]"
        )
    return result


def load_state() -> RunState:
    return RunState.load()


def write_state(state: RunState | dict[str, object]) -> None:
    if isinstance(state, RunState):
        state.save()
    else:
        for key in ("best_qval_loss", "best_qval_cycle", "stale_qval_cycles"):
            state.pop(key, None)
        write_json(STATE_PATH, state)

def result_line(
    cycle: int,
    train_samples: int,
    rows: list[dict[str, object]],
    release_matches: list[MatchResult],
) -> str:
    best = rows[0]
    qval = float(best["quantized_val_loss"])
    epoch = int(best["epoch"])
    return (
        f"cycle={cycle} train={train_samples} val={VALIDATION_SAMPLES} "
        f"best_epoch={epoch} best_qval_loss={qval:.6f} "
        f"{match_list_fields(release_matches, f'vs_{slug(REFERENCE_REF)}')}"
    )


def positive_net_path(cycle: int, rank: int, epoch: int) -> Path:
    return POSITIVE_NET_DIR / f"{cycle_candidate_id(cycle, rank)}_epoch{epoch:04}.nnq"


def public_positive_net_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "id": record.get("id"),
        "cycle": record.get("cycle"),
        "rank": record.get("rank"),
        "epoch": record.get("epoch"),
        "qval_loss": record.get("qval_loss"),
        "model": record.get("model"),
        "train_prefix_samples": record.get("train_samples"),
        "reference_ref": record.get("reference_ref"),
        "elo_vs_release": record.get("elo_vs_release"),
        "elo_95_ci": record.get("elo_95_ci"),
        "wdl_vs_release": record.get("wdl_vs_release"),
    }


def write_positive_net_manifest(records: list[dict[str, object]]) -> None:
    exported_records = [public_positive_net_record(record) for record in records]
    write_json(
        POSITIVE_NET_DIR / "positive_nets.json",
        {
            "reference_ref": REFERENCE_REF,
            "count": len(exported_records),
            "nets": exported_records,
        },
    )


def screen_candidate_records(
    cycle: int,
    train_samples: int,
    corpus_dir: Path,
    rows: list[dict[str, object]],
    selected_models: list[Path],
    selected_bins: list[Path],
    release_matches: list[MatchResult],
) -> list[dict[str, object]]:
    if not (len(rows) == len(selected_models) == len(selected_bins) == len(release_matches)):
        raise SystemExit("screened candidate lists have inconsistent lengths")
    records: list[dict[str, object]] = []
    for rank, (row, model, source_bin, match) in enumerate(
        zip(rows, selected_models, selected_bins, release_matches),
        start=1,
    ):
        if match.elo <= 0.0:
            continue
        epoch = int(row["epoch"])
        kept_model = positive_net_path(cycle, rank, epoch)
        kept_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model, kept_model)
        records.append(
            {
                "id": cycle_candidate_id(cycle, rank),
                "cycle": cycle,
                "rank": rank,
                "epoch": epoch,
                "qval_loss": float(row["quantized_val_loss"]),
                "model": str(kept_model),
                "source_bin": str(source_bin),
                "corpus_dir": str(corpus_dir),
                "train_samples": train_samples,
                "reference_ref": REFERENCE_REF,
                "elo_vs_release": match.elo,
                "elo_95_ci": [match.elo_lower, match.elo_upper],
                "wdl_vs_release": [match.wins, match.draws, match.losses],
            }
        )
    return records


def select_tournament_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    seen_bins: set[str] = set()
    positives: list[dict[str, object]] = []
    for candidate in candidates:
        if str(candidate.get("reference_ref") or "") != REFERENCE_REF:
            continue
        if float(candidate.get("elo_vs_release", float("-inf"))) <= 0.0:
            continue
        source_bin = str(candidate.get("source_bin") or "")
        if not source_bin or source_bin in seen_bins:
            continue
        if not Path(source_bin).is_file():
            append_log(f"# skipping tournament candidate with missing binary: {source_bin}")
            continue
        positives.append(candidate)
        seen_bins.add(source_bin)
    return sorted(
        positives,
        key=lambda candidate: (
            -float(candidate.get("elo_vs_release", float("-inf"))),
            float(candidate.get("qval_loss", float("inf"))),
            int(candidate.get("cycle", 0)),
            int(candidate.get("rank", 0)),
        ),
    )


def candidate_player_name(candidate: dict[str, object]) -> str:
    cycle = int(candidate.get("cycle", 0))
    rank = int(candidate.get("rank", 0))
    epoch = int(candidate.get("epoch", 0))
    return f"cycle{cycle:04}-q{rank:02}-e{epoch:04}"


def tournament_player_from_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(candidate["id"]),
        "name": candidate_player_name(candidate),
        "source_bin": str(candidate["source_bin"]),
        "model": str(candidate.get("model") or ""),
        "corpus_dir": str(candidate.get("corpus_dir") or ""),
        "cycle": int(candidate.get("cycle", 0)),
        "rank": int(candidate.get("rank", 0)),
        "epoch": int(candidate.get("epoch", 0)),
        "train_samples": int(candidate.get("train_samples", 0)),
        "reference_ref": str(candidate.get("reference_ref") or REFERENCE_REF),
        "screen_elo": float(candidate.get("elo_vs_release", 0.0)),
        "qval_loss": float(candidate.get("qval_loss", 0.0)),
        "is_reference": False,
    }


def reference_tournament_player() -> dict[str, object]:
    return {
        "id": f"reference:{REFERENCE_REF}",
        "name": REFERENCE_REF,
        "source_bin": str(REFERENCE_BIN),
        "screen_elo": 0.0,
        "qval_loss": None,
        "is_reference": True,
    }


def even_game_count(games: int) -> int:
    if games <= 0:
        return 2
    return games if games % 2 == 0 else games + 1


def tournament_games_per_pairing(player_count: int) -> int:
    opponents = max(1, player_count - 1)
    required = even_game_count(math.ceil(TOURNAMENT_GAMES_PER_ENGINE / opponents))
    configured = even_game_count(TOURNAMENT_GAMES_PER_PAIRING_OVERRIDE)
    return max(required, configured)


def tournament_summary_has_required_games(summary: dict[str, object] | None) -> bool:
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        return False
    standings = summary.get("standings")
    if not isinstance(standings, list) or not standings:
        return False
    games = [
        int(row.get("games", 0))
        for row in standings
        if isinstance(row, dict)
    ]
    pair_results = summary.get("pair_results")
    games_per_pairing = int(summary.get("games_per_pairing", 0))
    openings_per_pairing = int(summary.get("openings_per_pairing", 0))
    unique_openings_per_pairing = int(
        summary.get("unique_openings_per_pairing", summary.get("unique_openings", 0))
    )
    return (
        bool(games)
        and min(games) >= TOURNAMENT_GAMES_PER_ENGINE
        and games_per_pairing > 0
        and games_per_pairing % 2 == 0
        and openings_per_pairing >= games_per_pairing // 2
        and unique_openings_per_pairing >= games_per_pairing // 2
    )


def standing_row(
    player: dict[str, object],
    scorecard: Scorecard,
    reference_elo: float,
) -> dict[str, object]:
    return {
        "player": str(player["name"]),
        "id": str(player["id"]),
        "elo_vs_latest": scorecard.elo - reference_elo,
        "games": scorecard.games,
        "score_pct": scorecard.score_pct,
        "wins": scorecard.wins,
        "draws": scorecard.draws,
        "losses": scorecard.losses,
        "screen_elo": float(player["screen_elo"]),
        "qval_loss": player["qval_loss"],
        "is_reference": bool(player["is_reference"]),
        "model": str(player.get("model") or ""),
        "corpus_dir": str(player.get("corpus_dir") or ""),
        "cycle": int(player.get("cycle", 0)),
        "rank": int(player.get("rank", 0)),
        "epoch": int(player.get("epoch", 0)),
        "train_samples": int(player.get("train_samples", 0)),
        "reference_ref": str(player.get("reference_ref") or REFERENCE_REF),
    }


def run_final_tournament(
    state: RunState,
    match_runner=run_selfplay_match,
) -> dict[str, object]:
    selected = select_tournament_candidates(
        [candidate for candidate in state.screened_candidates if isinstance(candidate, dict)]
    )
    if not selected:
        emit("tournament=skipped reason=no_positive_screen_hits")
        return {"status": "skipped", "reason": "no_positive_screen_hits"}
    if not REFERENCE_BIN.is_file():
        raise SystemExit(f"latest release binary missing for tournament: {REFERENCE_BIN}")

    players = [tournament_player_from_candidate(candidate) for candidate in selected]
    players.append(reference_tournament_player())
    games_per_pairing = tournament_games_per_pairing(len(players))
    paired_openings_per_pairing = games_per_pairing // 2
    total_pairs = len(players) * (len(players) - 1) // 2
    openings, tournament_openings = prepare_tournament_openings(
        OPENING_CONFIG,
        paired_openings_per_pairing,
    )
    opening_count = len(tournament_openings)

    emit_status(
        "tournament=start "
        f"players={len(players)} positive_nets={len(selected)} "
        f"games_per_pairing={games_per_pairing} "
        f"target_games_per_engine={TOURNAMENT_GAMES_PER_ENGINE} "
        f"unique_openings_per_pairing={opening_count} "
        f"openings_per_pairing={paired_openings_per_pairing} "
        f"time_ms={MATCH_MS} "
        f"parallel_matches={max(1, min(TOURNAMENT_PARALLEL_MATCHES, total_pairs))}"
    )
    stats = {str(player["id"]): Scorecard() for player in players}
    pair_results: list[dict[str, object]] = []
    pair_jobs: list[tuple[int, dict[str, object], dict[str, object]]] = []
    for left_index, left in enumerate(players):
        for right in players[left_index + 1 :]:
            pair_number = len(pair_jobs) + 1
            pair_jobs.append((pair_number, left, right))

    def run_pair(
        job: tuple[int, dict[str, object], dict[str, object]],
    ) -> tuple[int, dict[str, object], dict[str, object], MatchResult]:
        pair_number, left, right = job
        emit_status(
            "tournament_pair=start "
            f"pair={pair_number}/{total_pairs} "
            f"left={left['name']} "
            f"right={right['name']} "
            f"openings={paired_openings_per_pairing}"
        )
        match = match_runner(
            Path(str(left["source_bin"])),
            Path(str(right["source_bin"])),
            openings,
            games_per_pairing,
            MATCH_MS,
            allow_local_failure=not bool(left["is_reference"]),
            allow_baseline_failure=not bool(right["is_reference"]),
        )
        return pair_number, left, right, match

    parallel_matches = max(1, min(TOURNAMENT_PARALLEL_MATCHES, total_pairs))
    with ThreadPoolExecutor(max_workers=parallel_matches) as executor:
        futures = [executor.submit(run_pair, job) for job in pair_jobs]
        for future in as_completed(futures):
            pair_number, left, right, match = future.result()
            emit_status(
                "tournament_pair=done "
                f"pair={pair_number}/{total_pairs} "
                f"left={left['name']} "
                f"right={right['name']} "
                f"wdl_left={match.wins}-{match.draws}-{match.losses} "
                f"elo_left={match.elo:+.2f}[{match.elo_lower:+.2f},{match.elo_upper:+.2f}]"
            )
            stats[str(left["id"])].add(match.wins, match.draws, match.losses)
            stats[str(right["id"])].add(match.losses, match.draws, match.wins)
            pair_results.append(
                {
                    "pair": pair_number,
                    "left": str(left["name"]),
                    "right": str(right["name"]),
                    "wdl_left": [match.wins, match.draws, match.losses],
                    "elo_left": match.elo,
                    "elo_95_ci": [match.elo_lower, match.elo_upper],
                    "openings": paired_openings_per_pairing,
                }
            )
    pair_results.sort(key=lambda row: int(row["pair"]))

    reference_id = f"reference:{REFERENCE_REF}"
    reference_stats = stats[reference_id]
    reference_elo = reference_stats.elo
    standings = [
        standing_row(player, stats[str(player["id"])], reference_elo)
        for player in players
    ]
    standings.sort(
        key=lambda row: (
            -float(row["elo_vs_latest"]),
            -float(row["score_pct"]),
            str(row["player"]),
        )
    )
    emit_tournament_table(standings)
    return {
        "status": "completed",
        "reference_ref": REFERENCE_REF,
        "positive_nets": len(selected),
        "games_per_pairing": games_per_pairing,
        "target_games_per_engine": TOURNAMENT_GAMES_PER_ENGINE,
        "openings": opening_count,
        "unique_openings": opening_count,
        "unique_openings_per_pairing": opening_count,
        "openings_per_pairing": paired_openings_per_pairing,
        "opening_source": str(
            OPENING_CONFIG.tournament_openings
            if OPENING_CONFIG.tournament_openings_explicit
            else OPENING_CONFIG.random_openings
        ),
        "opening_book": str(openings),
        "time_ms": MATCH_MS,
        "standings": standings,
        "pair_results": pair_results,
    }


def emit_tournament_table(standings: list[dict[str, object]]) -> None:
    emit_raw("")
    for line in tournament_table_lines(standings):
        emit_raw(line)
    emit_raw("")


def rust_summary_path(name: str, summary: dict[str, object]) -> Path:
    path = WORK_DIR / name
    write_json(path, summary)
    return path


def write_tournament_results(summary: dict[str, object]) -> dict[str, str] | None:
    if summary.get("status") != "completed":
        return None
    report = run_json_command(
        [
            NNUE_BIN,
            "export-results",
            "--summary",
            rust_summary_path("tournament-summary.json", summary),
            "--out-dir",
            TOURNAMENT_RESULT_DIR,
        ],
        label="rust export-results",
    )
    if report is None:
        return None
    if not isinstance(report, dict):
        raise SystemExit("rust export-results returned a non-object JSON payload")
    write_tournament_barplot(Path(str(report.get("json", TOURNAMENT_RESULT_DIR / "results.json"))))
    return {str(key): str(value) for key, value in report.items()}


def tournament_winner(summary: dict[str, object]) -> dict[str, object]:
    standings = summary.get("standings")
    if not isinstance(standings, list):
        raise SystemExit("completed tournament summary is missing standings")
    rows = [
        row for row in standings
        if isinstance(row, dict) and not bool(row.get("is_reference"))
    ]
    if not rows:
        raise SystemExit("completed tournament has no positive candidate rows")
    return max(
        rows,
        key=lambda row: (
            float(row.get("elo_vs_latest", float("-inf"))),
            float(row.get("score_pct", float("-inf"))),
            int(row.get("games", 0)),
        ),
    )


def copy_final_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing final artifact source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_final_training_bundle(summary: dict[str, object]) -> None:
    winner = tournament_winner(summary)
    write_screening_plot(REFERENCE_REF)
    write_tournament_barplot(TOURNAMENT_RESULT_DIR / "results.json")

    final_files = {
        "train.sbin",
        "val.sbin",
        "network.nnq",
        "elo_screen.png",
        "elo_tournament.png",
    }
    copy_final_file(Path(str(winner.get("model") or "")), CANONICAL_TRAINING_DIR / "network.nnq")
    copy_final_file(OUTPUT_DIR / "elo_screen.png", CANONICAL_TRAINING_DIR / "elo_screen.png")
    copy_final_file(
        TOURNAMENT_RESULT_DIR / "elo_tournament.png",
        CANONICAL_TRAINING_DIR / "elo_tournament.png",
    )
    for entry in CANONICAL_TRAINING_DIR.iterdir():
        if entry.name in final_files:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def export_positive_training_data(summary: dict[str, object]) -> dict[str, object] | None:
    if summary.get("status") != "completed":
        return None
    export = run_json_command(
        [
            NNUE_BIN,
            "export-positive-training-data",
            "--summary",
            rust_summary_path("training-export-summary.json", summary),
            "--out-dir",
            CANONICAL_TRAINING_DIR,
            "--work-dir",
            WORK_DIR,
            "--reference-ref",
            REFERENCE_REF,
        ],
        label="rust export-positive-training-data",
    )
    if export is None:
        return None
    if not isinstance(export, dict):
        raise SystemExit("rust export-positive-training-data returned a non-object JSON payload")
    if export.get("status") == "skipped":
        emit_status(
            "training_data_export=skipped "
            f"reason={export.get('reason')}"
        )
        return export
    write_final_training_bundle(summary)
    emit_status(
        "training_data_exported "
        f"dir={export.get('training_dir')} "
        f"winner={export.get('winner_player')} "
        f"train_prefix_samples={export.get('train_prefix_samples')} "
        f"reference={REFERENCE_REF}"
    )
    return export


def cleanup_final_artifacts() -> None:
    run_root_resolved = RUN_ROOT.resolve()

    def remove_run_tree(path: Path) -> None:
        try:
            path.resolve().relative_to(run_root_resolved)
        except ValueError:
            append_log(f"# cleanup skipped non-run-root path: {path}")
            return
        if path.exists():
            shutil.rmtree(path)

    for path in SCRATCH_DIRS + RUN_OUTPUT_DIRS:
        remove_run_tree(path)
    for path in FINAL_OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    emit_status("cleanup=done retained=logs,training_bundle")


def finish_training(state: RunState) -> None:
    if state.tournament_completed:
        summary = state.tournament_summary
        if not tournament_summary_has_required_games(summary):
            append_log(
                "# tournament rerun required: completed summary does not meet "
                f"{TOURNAMENT_GAMES_PER_ENGINE} games per engine"
            )
            state.tournament_completed = False
            state.tournament_summary = None
            state.tournament_results = None
            state.training_data_export = None
            state.save()
            summary = None
        else:
            summary = state.tournament_summary
    if state.tournament_completed:
        if summary is not None and summary.get("status") == "completed":
            standings = summary.get("standings")
            if isinstance(standings, list):
                emit_tournament_table([row for row in standings if isinstance(row, dict)])
            if not state.tournament_results:
                state.tournament_results = write_tournament_results(summary)
                state.save()
            elif isinstance(state.tournament_results, dict):
                write_tournament_barplot(
                    Path(str(state.tournament_results.get("json", TOURNAMENT_RESULT_DIR / "results.json")))
                )
        if summary is not None and not state.training_data_export:
            state.training_data_export = export_positive_training_data(summary)
            state.save()
        emit("tournament=skipped reason=already_completed")
        cleanup_final_artifacts()
        return
    summary = run_final_tournament(state)
    state.tournament_completed = True
    state.tournament_summary = summary
    state.tournament_results = write_tournament_results(summary)
    state.save()
    state.training_data_export = export_positive_training_data(summary)
    state.save()
    cleanup_final_artifacts()


def raise_open_file_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError) as error:
        append_log(f"# open_file_limit unchanged soft={soft} hard={hard} error={error}")


def train_or_reuse_cycle(
    cycle: int,
    previous_train_samples: int,
    generator: ContinuousGenerator,
) -> tuple[CycleTrainingArtifact, bool, float | None, float | None]:
    required_samples = required_train_samples(previous_train_samples)
    required_increment = train_increment_for(previous_train_samples)
    generator.set_target(VALIDATION_SAMPLES + required_samples)
    emit_status(
        "cycle=start "
        f"cycle={cycle} "
        f"previous_train_samples={previous_train_samples} "
        f"required_train_samples={required_samples} "
        f"required_increment={required_increment}"
    )
    cached = completed_cycle_artifact(cycle, required_samples)
    if cached is not None:
        generator.note_unique_samples(cached.train_samples + VALIDATION_SAMPLES)
        emit_status(
            "training=reuse "
            f"cycle={cycle} "
            f"train_samples={cached.train_samples} "
            f"experiment={cached.experiment_dir} "
            f"screen_epochs={row_epochs(cached.rows)} "
            f"screen_qval_losses={row_qvals(cached.rows)}"
        )
        return cached, True, 0.0, 0.0

    corpus_started = time.perf_counter()
    total_samples, corpus_dir = generate_until_trainable(
        cycle,
        previous_train_samples,
        generator,
    )
    corpus_seconds = time.perf_counter() - corpus_started
    train_samples = total_samples - VALIDATION_SAMPLES
    train_increment = train_samples - previous_train_samples
    if train_increment < required_increment:
        raise SystemExit(
            f"cycle {cycle} has only {train_increment} new training samples; "
            f"need at least {required_increment}"
        )
    generator.check()
    training_started = time.perf_counter()
    experiment_dir, rows = train_cycle(cycle, corpus_dir, train_samples)
    training_seconds = time.perf_counter() - training_started
    return (
        CycleTrainingArtifact(
            experiment_dir=experiment_dir,
            corpus_dir=corpus_dir,
            train_samples=train_samples,
            rows=rows,
        ),
        False,
        corpus_seconds,
        training_seconds,
    )


def screen_cycle_models(
    cycle: int,
    artifact: CycleTrainingArtifact,
    generator: ContinuousGenerator,
) -> tuple[list[Path], list[Path], list[MatchResult], list[dict[str, object]], float, float]:
    generator.check()
    selected_models = [
        screen_model_path(artifact.experiment_dir, row, rank)
        for rank, row in enumerate(artifact.rows, start=1)
    ]
    emit_status(
        "screening=build_candidates "
        f"cycle={cycle} "
        f"count={len(selected_models)}"
    )
    build_started = time.perf_counter()
    selected_bins = [
        build_reference_net_engine(cycle, model, rank)
        for rank, model in enumerate(selected_models, start=1)
    ]
    build_seconds = time.perf_counter() - build_started
    generator.check()
    release_match_started = time.perf_counter()
    release_matches = [
        run_selfplay_match(
            selected_bin,
            REFERENCE_BIN,
            label=f"cycle={cycle} q={rank} vs_{slug(REFERENCE_REF)}",
        )
        for rank, selected_bin in enumerate(selected_bins, start=1)
    ]
    release_match_seconds = time.perf_counter() - release_match_started
    generator.check()
    positive_records = screen_candidate_records(
        cycle,
        artifact.train_samples,
        artifact.corpus_dir,
        artifact.rows,
        selected_models,
        selected_bins,
        release_matches,
    )
    return (
        selected_models,
        selected_bins,
        release_matches,
        positive_records,
        build_seconds,
        release_match_seconds,
    )


def run_cycle(
    cycle: int,
    previous_train_samples: int,
    generator: ContinuousGenerator,
) -> CycleResult:
    cycle_started = time.perf_counter()
    artifact, reused_training, corpus_seconds, training_seconds = train_or_reuse_cycle(
        cycle,
        previous_train_samples,
        generator,
    )
    (
        selected_models,
        selected_bins,
        release_matches,
        positive_records,
        build_seconds,
        release_match_seconds,
    ) = screen_cycle_models(cycle, artifact, generator)
    return CycleResult(
        cycle=cycle,
        corpus_dir=artifact.corpus_dir,
        experiment_dir=artifact.experiment_dir,
        train_samples=artifact.train_samples,
        rows=artifact.rows,
        selected_models=selected_models,
        selected_bins=selected_bins,
        release_matches=release_matches,
        positive_records=positive_records,
        reused_training=reused_training,
        corpus_seconds=corpus_seconds,
        training_seconds=training_seconds,
        build_seconds=build_seconds,
        release_match_seconds=release_match_seconds,
        total_seconds=time.perf_counter() - cycle_started,
    )


def emit_cycle_result(result: CycleResult) -> None:
    append_log(
        "# cycle_timing "
        f"cycle={result.cycle} "
        f"reused_training={int(result.reused_training)} "
        f"corpus_s={format_timing_seconds(result.corpus_seconds)} "
        f"training_s={format_timing_seconds(result.training_seconds)} "
        f"build_s={format_timing_seconds(result.build_seconds)} "
        f"release_match_s={format_timing_seconds(result.release_match_seconds)} "
        f"total_s={format_timing_seconds(result.total_seconds)}"
    )
    emit(result_line(result.cycle, result.train_samples, result.rows, result.release_matches))
    write_screening_plot(REFERENCE_REF)


def main(argv: list[str]) -> int:
    clean = parse_launcher_args(argv)
    running_processes = other_train_processes(argv)
    if running_processes:
        report_existing_run(running_processes, clean)
        return 0
    raise_open_file_limit()
    if clean:
        clean_run_root()
    configure_release_ref()
    state = load_state()
    emit_status(
        "train=start "
        f"reference={REFERENCE_REF} "
        f"run_root={RUN_ROOT} "
        f"resume_cycle={state.cycle} "
        f"resume_train_samples={state.train_samples} "
        f"screened_candidates={len(state.screened_candidates)}"
    )
    setup()

    generator = ContinuousGenerator(REFERENCE_BIN, GENERATOR_CONFIG, emit_status, command_text)
    generator.start()
    training_finished = False
    try:
        while not state.complete(TRAIN_CONFIG):
            result = run_cycle(state.next_cycle, state.train_samples, generator)
            state.record_cycle(result)
            write_positive_net_manifest(state.screened_candidates)
            emit_cycle_result(result)
            state.save()
        training_finished = True
    finally:
        generator.stop()
    if training_finished:
        finish_training(load_state())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except SystemExit as error:
        code = error.code
        if code not in (None, 0):
            record_failure(code)
        raise
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        record_failure(error)
        raise
