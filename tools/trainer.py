#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, NoReturn, Sequence, TextIO

import numpy as np


ABAPACK_MAGIC = b"ABAPACK1"
ABAPACK_VERSION = 5
ABAPACK_KIND_PREPARED = 2
TEACHER_TARGET_MAGIC = b"TEACHER1"
TEACHER_TARGET_VERSION = 1
ABAPACK_SPLIT_TO_BYTE = {"train": 1, "val": 2, "test": 3}
ABAPACK_BYTE_TO_SPLIT = {value: key for key, value in ABAPACK_SPLIT_TO_BYTE.items()}

U8 = struct.Struct("<B")
U16 = struct.Struct("<H")
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
I8 = struct.Struct("<b")
I32 = struct.Struct("<i")
F32 = struct.Struct("<f")

DEFAULT_DATASET_NAME = "steinbeisser-rc5_50K_book_75ms_unique.abapack"
DEFAULT_TIME_MS = 5
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_RUNTIME_LOSS_INTERVAL = 1
ACTIVE_LEARNING_RUNTIME_LOSS_INTERVAL = 1
DEFAULT_LEARNING_RATE = 1.9e-3
DEFAULT_WEIGHT_DECAY = 7e-4
DEFAULT_DROPOUT = 0.0
DEFAULT_LAMBDA_MIX = (0.99, 0.01)
DEFAULT_BACKEND = "jax"
DEFAULT_PROFILE = "random"
DEFAULT_TIER = "full"
DEFAULT_BASE_RECIPE = "base"
DEFAULT_FIXED_VALIDATION_SAMPLES = 40_000
DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES = 20_000
DEFAULT_FAST_TRAIN_SAMPLES = 1_000_000
DEFAULT_FAST_VALIDATION_SAMPLES = 20_000
DEFAULT_CHECKPOINT_INTERVAL = 1
DEFAULT_CHECKPOINT_GAMES = 500
DEFAULT_CONFIRM_GAMES = 2_000
DEFAULT_MICRO_TRAIN_SAMPLES = 250_000
DEFAULT_MICRO_MATCH_GAMES = 120
DEFAULT_FAST_MATCH_GAMES = 120
DEFAULT_PATIENCE = 10
ACTIVE_LEARNING_PATIENCE = 10
DEFAULT_TOP_CHECKPOINT_COUNT = 3


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_THREADS = max(1, env_int("STEINBEISSER_TRAIN_THREADS", 15))
DEFAULT_LOADER_WORKERS = max(1, env_int("STEINBEISSER_LOADER_WORKERS", 14))
DEFAULT_REPEAT_OCCURRENCE_WEIGHT = "sqrt_cap2"
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_DENSE_FEATURE_MASK = ()
MODEL_INPUTS = "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_no_group_drop_count_material_v23"
ARCHITECTURE = "130,84,50,1"
TARGET_TRANSFORM = "linear_clip_v1"
LOSS = "huber"
HUBER_DELTA = 1.0
ACTIVATION = "relu"
NORM = "none"
BLOCK_TYPE = "plain"
DEFAULT_CLASS_WEIGHTING = "inverse_sqrt"
DEFAULT_MAX_ABS_SCORE = 3500
TERMINAL_SCORE_THRESHOLD = 5_000.0
LABBOOK_PATH = Path(
    os.environ.get("STEINBEISSER_TRAIN_ARTIFACT_ROOT", Path(__file__).resolve().parent)
) / "labbook.md"
SCORE_BUCKET_SIZE = 500
PLY_BUCKET_SIZE = 10
SELECTION_INDEX_VERSION = 2
SELECTION_CORPUS_VERSION = "v3"
ACTIVE_LEARNING_VERSION = "v1"
SUBSET_SWEEP_NAME = "subset_lab_v1"
MASK64 = (1 << 64) - 1
UINT64_DENOMINATOR = float((1 << 64) + 1)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    strategy: str
    neutral_scale_cp: float | None = None
    tail_floor: float = 1.0
    terminal_cap_fraction: float | None = None
    bucket_axes: tuple[str, ...] = ()
    opening_cap_multiplier: float | None = None
    opening_cap_min: int | None = None
    opening_cap_max: int | None = None


@dataclass(frozen=True)
class TierSpec:
    name: str
    train_samples: int
    fixed_validation_samples: int
    diagnostic_validation_samples: int
    epochs: int
    final_match_games: int
    checkpoint_interval_epochs: int | None
    checkpoint_match_games: int | None


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    dropout: float


@dataclass(frozen=True)
class ActiveLearningSpec:
    chunk_sizes: tuple[int, ...] | None = None
    chunk_size: int | None = None
    pool_cutoff_cp: int | None = None
    round_match_top_k: int = 1

    @property
    def manual_chunk_sizes(self) -> tuple[int, ...] | None:
        return self.chunk_sizes

    @property
    def auto_chunk_size(self) -> int | None:
        return self.chunk_size

    @property
    def is_auto(self) -> bool:
        return self.chunk_size is not None

    def planned_total_selected_samples(self, pool_train_samples: int) -> int:
        if self.chunk_sizes is not None:
            return sum(self.chunk_sizes)
        if self.chunk_size is not None:
            return pool_train_samples
        fail("active-learning spec is missing both chunk schedule and chunk size")

    @property
    def chunk_label(self) -> str:
        if self.chunk_sizes is not None:
            return "-".join(sample_count_label(value) for value in self.chunk_sizes)
        if self.chunk_size is not None:
            return f"auto{sample_count_label(self.chunk_size)}"
        fail("active-learning spec is missing both chunk schedule and chunk size")


@dataclass(frozen=True)
class ActiveLearningRoundSpec:
    index: int
    chunk_size: int
    selected_samples: int

    @property
    def round_slug(self) -> str:
        return f"round_{self.index:02d}_{sample_count_label(self.selected_samples)}"


PROFILE_SPECS = {
    "random": ProfileSpec("random", "weighted_score"),
    "terminal_cap_only": ProfileSpec("terminal_cap_only", "weighted_score", terminal_cap_fraction=0.05),
    "neutral_light": ProfileSpec(
        "neutral_light",
        "weighted_score",
        neutral_scale_cp=1200.0,
        tail_floor=0.35,
        terminal_cap_fraction=0.10,
    ),
    "neutral_medium": ProfileSpec(
        "neutral_medium",
        "weighted_score",
        neutral_scale_cp=800.0,
        tail_floor=0.20,
        terminal_cap_fraction=0.05,
    ),
    "neutral_hard": ProfileSpec(
        "neutral_hard",
        "weighted_score",
        neutral_scale_cp=500.0,
        tail_floor=0.10,
        terminal_cap_fraction=0.03,
    ),
    "score_bucket_flat_v1": ProfileSpec(
        "score_bucket_flat_v1",
        "bucket_flat",
        bucket_axes=("score_bucket",),
    ),
    "ply_bucket_flat_v1": ProfileSpec(
        "ply_bucket_flat_v1",
        "bucket_flat",
        bucket_axes=("ply_bucket",),
    ),
    "result_balanced_v1": ProfileSpec(
        "result_balanced_v1",
        "bucket_flat",
        bucket_axes=("result_bucket",),
    ),
    "opening_diverse_cap_v1": ProfileSpec(
        "opening_diverse_cap_v1",
        "opening_cap",
        opening_cap_multiplier=4.0,
        opening_cap_min=8,
        opening_cap_max=64,
    ),
    "score_ply_hybrid_v1": ProfileSpec(
        "score_ply_hybrid_v1",
        "bucket_flat",
        bucket_axes=("score_bucket", "ply_bucket"),
    ),
}

TIER_SPECS = {
    "micro": TierSpec(
        "micro",
        DEFAULT_MICRO_TRAIN_SAMPLES,
        DEFAULT_FIXED_VALIDATION_SAMPLES,
        DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES,
        8,
        DEFAULT_MICRO_MATCH_GAMES,
        None,
        None,
    ),
    "fast": TierSpec(
        "fast",
        DEFAULT_FAST_TRAIN_SAMPLES,
        DEFAULT_FAST_VALIDATION_SAMPLES,
        DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES,
        100,
        0,
        DEFAULT_CHECKPOINT_INTERVAL,
        DEFAULT_FAST_MATCH_GAMES,
    ),
    "probe": TierSpec("probe", 500_000, 40_000, 20_000, 12, 200, None, None),
    "screen": TierSpec("screen", 2_000_000, 40_000, 20_000, 30, 300, None, None),
    "full": TierSpec(
        "full",
        4_000_000,
        40_000,
        20_000,
        100,
        500,
        DEFAULT_CHECKPOINT_INTERVAL,
        DEFAULT_CHECKPOINT_GAMES,
    ),
}

RECIPE_SPECS = {
    "base": RecipeSpec("base", DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT),
    "low_lr": RecipeSpec("low_lr", 0.002, 1e-4, 256, 0.0),
    "very_low_lr": RecipeSpec("very_low_lr", 0.0015, 1e-4, 256, 0.0),
    "high_lr": RecipeSpec("high_lr", 0.004, 1e-4, 256, 0.0),
    "low_wd": RecipeSpec("low_wd", 0.003, 3e-5, 256, 0.0),
}


@dataclass(frozen=True)
class CliArgs:
    dataset_arg: str
    engine_arg: str
    reference_source: str | None
    checkpoint_book: str | None
    lambda_mix: tuple[float, float]
    time_ms: int
    seed: int
    train_samples: int | None
    validation_samples: int
    diagnostic_validation_samples: int | None
    epochs: int | None
    batch_size: int
    learning_rate: float | None
    weight_decay: float | None
    dropout: float | None
    checkpoint_interval: int | None
    checkpoint_start: int | None
    checkpoint_games: int | None
    final_match_games: int | None
    patience: int | None
    runtime_loss_interval: int | None
    backend: str | None
    activation: str | None
    feature_set: str | None
    dense_mask: tuple[int, ...] | None
    architecture: str | None
    repeat_occurrence_weight: str | None
    class_weighting: str | None
    ema_decay: float | None
    profile: str
    tier: str
    sweep: str | None
    active_learning_chunks: tuple[int, ...] | None
    active_learning_chunk_size: int | None
    active_learning_pool_cutoff_cp: int | None
    active_learning_match_top_k: int
    max_abs_score: int | None


@dataclass(frozen=True)
class DatasetInput:
    source_path: Path
    prepared_path: Path
    dataset_stem: str
    dataset_slug: str
    sibling_unique_shard: bool


@dataclass(frozen=True)
class BaseCorpus:
    corpus_dir: Path
    train_path: Path
    val_path: Path
    manifest_path: Path
    manifest: dict[str, object]
    train_samples: int
    val_samples: int


@dataclass(frozen=True)
class SelectionIndex:
    key_hashes: np.ndarray
    abs_scores: np.ndarray
    terminal_mask: np.ndarray
    score_buckets: np.ndarray
    ply_buckets: np.ndarray
    result_buckets: np.ndarray
    opening_hashes: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.key_hashes.shape[0])


@dataclass(frozen=True)
class CorpusPaths:
    corpus_dir: Path
    train_path: Path
    val_path: Path
    diagnostic_val_path: Path
    manifest_path: Path
    profile_path: Path
    dataset_cache_dir: Path
    train_samples: int
    val_samples: int
    diagnostic_val_samples: int


@dataclass(frozen=True)
class CandidateSpec:
    mode: str
    dataset_slug: str
    dataset_path: Path
    engine_template_path: Path
    profile: ProfileSpec
    tier: TierSpec
    recipe: RecipeSpec
    time_ms: int
    seed: int
    backend: str
    lambda_mix: tuple[float, float]
    checkpoint_start_epoch: int | None
    sweep_name: str | None
    active_learning: ActiveLearningSpec | None
    max_abs_score: int | None = DEFAULT_MAX_ABS_SCORE
    reference_source_path: Path | None = None
    repeat_occurrence_weight: str | None = DEFAULT_REPEAT_OCCURRENCE_WEIGHT
    class_weighting: str = DEFAULT_CLASS_WEIGHTING
    ema_decay: float | None = DEFAULT_EMA_DECAY
    feature_set_name: str = MODEL_INPUTS
    architecture_spec: str = ARCHITECTURE
    activation_name: str = "relu"
    dense_feature_mask: tuple[int, ...] = DEFAULT_DENSE_FEATURE_MASK
    runtime_loss_interval: int | None = None
    patience: int | None = None

    @property
    def lambda_score(self) -> float:
        return self.lambda_mix[0]

    @property
    def lambda_result(self) -> float:
        return self.lambda_mix[1]

    @property
    def candidate_id(self) -> str:
        parts = [
            self.dataset_slug,
            self.mode,
            self.tier.name,
            self.profile.name,
            self.recipe.name,
            f"seed{self.seed}",
        ]
        if self.active_learning is not None:
            parts.append(f"al{self.active_learning.chunk_label.replace('-', '_')}")
            if self.active_learning.pool_cutoff_cp is not None:
                parts.append(f"poolcap{self.active_learning.pool_cutoff_cp}")
        parts.append(score_filter_tag(self.max_abs_score))
        if self.repeat_occurrence_weight not in {None, "none"}:
            parts.append(f"rep{slugify(self.repeat_occurrence_weight)}")
        if self.class_weighting != DEFAULT_CLASS_WEIGHTING:
            parts.append(f"cw{slugify(self.class_weighting)}")
        if self.ema_decay is not None:
            parts.append(f"ema{ema_decay_tag(self.ema_decay)}")
        if self.feature_set_name != MODEL_INPUTS:
            parts.append(f"fs{feature_set_experiment_tag(self.feature_set_name)}")
        if self.architecture_spec != ARCHITECTURE:
            parts.append(f"arch{self.architecture_spec.replace(',', 'x')}")
        if self.activation_name != "relu":
            parts.append(f"act{self.activation_name}")
        if self.dense_feature_mask:
            parts.append("dm" + "_".join(str(index) for index in self.dense_feature_mask))
        if self.runtime_loss_interval is not None and self.runtime_loss_interval != DEFAULT_RUNTIME_LOSS_INTERVAL:
            parts.append(f"rt{self.runtime_loss_interval}")
        if self.patience is not None:
            parts.append(f"pat{self.patience}")
        return "_".join(parts)

    @property
    def epochs(self) -> int:
        return self.tier.epochs

    @property
    def train_samples(self) -> int:
        if self.active_learning is not None:
            return self.active_learning.planned_total_selected_samples(self.tier.train_samples)
        return self.tier.train_samples

    @property
    def pool_train_samples(self) -> int:
        return self.tier.train_samples

    @property
    def validation_samples(self) -> int:
        return self.tier.fixed_validation_samples

    @property
    def diagnostic_validation_samples(self) -> int:
        return self.tier.diagnostic_validation_samples


@dataclass(frozen=True)
class MatchResult:
    run_path: Path
    wins: int
    draws: int
    losses: int
    score_pct: float
    elo: float
    elo_lower: float
    elo_upper: float
    avg_depth: float
    avg_nps: float
    avg_response_ms: float
    decision: str
    games: int


@dataclass
class PreparedSample:
    black_bits: int
    white_bits: int
    side_to_move: str
    position_key: int
    mean_score: float
    mean_clipped_score: float
    mean_result: float
    mean_ply: float
    effective_game_turns_played: float
    occurrence_count: int
    sample_weight: float
    win_count: int
    draw_count: int
    loss_count: int
    result_bucket: int
    mean_completed_depth: float | None
    mean_no_progress_plies: float | None
    ejection_rate: float | None
    recorded_mean_score: float | None
    label_source: str | None
    label_budget_ms: int | None
    label_depth: int | None
    source_dataset: str | None


@dataclass
class PreparedChain:
    run_file: str
    game_index: int
    opening_name: str
    opening_position: str
    opening_hash: int
    split: str
    samples: list[PreparedSample]


@dataclass
class SummaryAccumulator:
    file_name: str
    chains: int = 0
    samples: int = 0
    raw_occurrences: int = 0
    opening_hashes: set[int] = field(default_factory=set)
    class_counts: dict[str, int] = field(
        default_factory=lambda: {"-1": 0, "0": 0, "1": 0}
    )

    def observe_chain(self, chain: PreparedChain) -> None:
        if not chain.samples:
            return
        self.chains += 1
        self.samples += len(chain.samples)
        self.raw_occurrences += sum(sample.occurrence_count for sample in chain.samples)
        self.opening_hashes.add(chain.opening_hash)
        for sample in chain.samples:
            bucket = str(sample.result_bucket)
            if bucket not in self.class_counts:
                self.class_counts[bucket] = 0
            self.class_counts[bucket] += 1

    def to_manifest_dict(self) -> dict[str, int | str]:
        return {
            "file": self.file_name,
            "chains": self.chains,
            "samples": self.samples,
            "raw_occurrences": self.raw_occurrences,
            "unique_openings": len(self.opening_hashes),
        }


class JsonBlobFilter:
    def __init__(self) -> None:
        self.suppressing = False
        self.brace_depth = 0

    def filter(self, line: str) -> str | None:
        stripped = line.lstrip()
        if not self.suppressing and stripped.startswith("{"):
            self.suppressing = True
            self.brace_depth = line.count("{") - line.count("}")
            if self.brace_depth <= 0:
                self.suppressing = False
                self.brace_depth = 0
            return None
        if self.suppressing:
            self.brace_depth += line.count("{") - line.count("}")
            if self.brace_depth <= 0:
                self.suppressing = False
                self.brace_depth = 0
            return None
        return line


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def status(message: str) -> None:
    print(message, flush=True)


def workspace_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if (root / "nnue/Cargo.toml").is_file():
        return root
    parent = root.parent
    if (parent / "nnue/Cargo.toml").is_file():
        return parent
    return root


def artifact_root(root: Path) -> Path:
    raw = os.environ.get("STEINBEISSER_TRAIN_ARTIFACT_ROOT")
    return Path(raw).expanduser().resolve() if raw else root


def parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="trainer",
        usage="trainer <dataset_name.json|prepared.abapack> <engine_name.rs> [options]",
        description="Train and screen Steinbeisser NNUE candidates against a fixed rc5 baseline.",
    )
    parser.add_argument("dataset_arg")
    parser.add_argument("engine_arg")
    parser.add_argument("--reference-source")
    parser.add_argument("--checkpoint-book")
    parser.add_argument(
        "--lambda-mix",
        nargs=2,
        type=float,
        metavar=("SCORE", "RESULT"),
        default=DEFAULT_LAMBDA_MIX,
    )
    parser.add_argument("--time", dest="time_ms", type=int, default=DEFAULT_TIME_MS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=DEFAULT_FIXED_VALIDATION_SAMPLES,
    )
    parser.add_argument("--diagnostic-validation-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--checkpoint-start", type=int)
    parser.add_argument("--checkpoint-games", type=int)
    parser.add_argument("--final-match-games", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--runtime-loss-interval", type=int)
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "mlx", "numpy", "python"],
    )
    parser.add_argument("--activation", choices=["relu", "silu", "screlu"])
    parser.add_argument("--feature-set")
    parser.add_argument("--dense-mask", type=parse_dense_mask)
    parser.add_argument("--architecture")
    parser.add_argument("--repeat-occurrence-weight", choices=["none", "sqrt_cap2"])
    parser.add_argument("--class-weighting", choices=["off", "inverse_sqrt"])
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument(
        "--max-abs-score",
        type=parse_max_abs_score,
        default=DEFAULT_MAX_ABS_SCORE,
        help="maximum abs(mean_clipped_score) kept for training corpora; use 0 to disable",
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS), default=DEFAULT_PROFILE)
    parser.add_argument("--tier", choices=sorted(TIER_SPECS), default=DEFAULT_TIER)
    parser.add_argument("--sweep", choices=["systematic_v1", SUBSET_SWEEP_NAME])
    parser.add_argument(
        "--active-learning-chunks",
        help="comma-separated training chunk sizes, for example 500k,500k,1m",
    )
    parser.add_argument(
        "--active-learning-chunk-size",
        "--active-learning-chunksize",
        dest="active_learning_chunk_size",
        type=parse_sample_count,
        help="single active-learning chunk size; trainer auto-repeats it until the eligible pool is exhausted",
    )
    parser.add_argument(
        "--active-learning-pool-cutoff",
        dest="active_learning_pool_cutoff_cp",
        type=int,
        help="optional abs(mean_clipped_score) cap for the active-learning pool",
    )
    parser.add_argument(
        "--active-learning-match-top-k",
        type=int,
        default=1,
        help="for active-learning rounds, test the best N quantized checkpoints against rc5",
    )
    namespace = parser.parse_args(list(argv))
    return CliArgs(
        dataset_arg=namespace.dataset_arg,
        engine_arg=namespace.engine_arg,
        reference_source=namespace.reference_source,
        checkpoint_book=namespace.checkpoint_book,
        lambda_mix=normalize_lambda_mix(*namespace.lambda_mix),
        time_ms=namespace.time_ms,
        seed=namespace.seed,
        train_samples=namespace.train_samples,
        validation_samples=namespace.validation_samples,
        diagnostic_validation_samples=namespace.diagnostic_validation_samples,
        epochs=namespace.epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        weight_decay=namespace.weight_decay,
        dropout=namespace.dropout,
        checkpoint_interval=namespace.checkpoint_interval,
        checkpoint_start=namespace.checkpoint_start,
        checkpoint_games=namespace.checkpoint_games,
        final_match_games=namespace.final_match_games,
        patience=namespace.patience,
        runtime_loss_interval=namespace.runtime_loss_interval,
        backend=namespace.backend,
        activation=namespace.activation,
        feature_set=namespace.feature_set,
        dense_mask=namespace.dense_mask,
        architecture=namespace.architecture,
        repeat_occurrence_weight=namespace.repeat_occurrence_weight,
        class_weighting=namespace.class_weighting,
        ema_decay=namespace.ema_decay,
        profile=namespace.profile,
        tier=namespace.tier,
        sweep=namespace.sweep,
        active_learning_chunks=parse_active_learning_chunks(namespace.active_learning_chunks),
        active_learning_chunk_size=namespace.active_learning_chunk_size,
        active_learning_pool_cutoff_cp=namespace.active_learning_pool_cutoff_cp,
        active_learning_match_top_k=namespace.active_learning_match_top_k,
        max_abs_score=namespace.max_abs_score,
    )


def resolve_existing_file(candidates: Sequence[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    options = ", ".join(str(candidate) for candidate in candidates)
    fail(f"{label} was not found; looked at: {options}")


def slugify(text: str) -> str:
    slug: list[str] = []
    prev_underscore = False
    for char in text:
        if char.isascii() and char.isalnum():
            slug.append(char.lower())
            prev_underscore = False
            continue
        if not prev_underscore:
            slug.append("_")
            prev_underscore = True
    return "".join(slug).strip("_")


def normalize_lambda_mix(score_contribution: float, result_contribution: float) -> tuple[float, float]:
    if score_contribution < 0.0 or result_contribution < 0.0:
        fail("lambda mix weights must be non-negative")
    total = score_contribution + result_contribution
    if total <= 0.0:
        fail("lambda mix weights must sum to a positive value")
    return (score_contribution / total, result_contribution / total)


FEATURE_SET_EXPERIMENT_TAGS = {
    MODEL_INPUTS: "v11",
    "stm_piece_cell_plus_fast_stats_turn_limit_late_ejection_v12": "v12",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_v13": "v13",
    "stm_piece_cell_plus_fast_stats_turn_limit_late_ejection_no_progress_v14": "v14",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_no_group_v15": "v15nogroup",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_proxy_v16": "v16proxy",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_drop_groups_v17": "v17dropgrp",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_no_group_drop_count_material_v23": "v23dropctmat",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_v24": "v24triang",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_lite_v25": "v25trianglite",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_v26": "v26triangcontact",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_single_v27": "v27triangsingle",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contest_single_v28": "v28triangcontest",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_edge_contest_single_v29": "v29triangedgect",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_multi_axis_single_v30": "v30triangmultiax",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_complete3_v31": "v31triangc3",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_pressure2_v32": "v32triangp2",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_support4_v33": "v33triangs4",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_ownplus_v34": "v34triangop",
    "stm_piece_cell_plus_fast_stats_turn_limit_no_progress_triangulation_contact_enemyheavy_v35": "v35triangeh",
}


def feature_set_experiment_tag(feature_set_name: str) -> str:
    tagged = FEATURE_SET_EXPERIMENT_TAGS.get(feature_set_name)
    if tagged is not None:
        return tagged
    slug = slugify(feature_set_name)
    if len(slug) <= 24:
        return slug
    digest = hashlib.blake2s(feature_set_name.encode("utf-8"), digest_size=4).hexdigest()
    return f"{slug[:15]}_{digest}"


def dense_feature_mask_label(indices: Sequence[int]) -> str:
    return ",".join(str(index) for index in indices) if indices else "none"


def default_dense_feature_mask_for_feature_set(feature_set_name: str) -> tuple[int, ...]:
    if feature_set_name == MODEL_INPUTS:
        return DEFAULT_DENSE_FEATURE_MASK
    return ()


def parse_dense_mask(value: str) -> tuple[int, ...]:
    text = value.strip().lower()
    if text in {"", "none"}:
        return ()
    indices: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as exc:
            fail(f"invalid dense mask index {item!r}: {exc}")
        if index < 0:
            fail(f"dense mask indices must be non-negative, got {index}")
        indices.append(index)
    return tuple(dict.fromkeys(indices))


def read_abapack_kind(path: Path) -> str | None:
    if path.suffix != ".abapack":
        return None
    with path.open("rb") as handle:
        header = handle.read(10)
    if len(header) < 10 or header[:8] != ABAPACK_MAGIC:
        fail(f"{path} has invalid abapack magic")
    kind = header[9]
    if kind == 1:
        return "raw"
    if kind == ABAPACK_KIND_PREPARED:
        return "prepared"
    fail(f"{path} has unsupported abapack kind byte {kind}")


def canonical_dataset_slug(dataset: DatasetInput) -> str:
    stem = dataset.prepared_path.stem
    if stem.endswith("_unique"):
        stem = stem[: -len("_unique")]
    return slugify(stem)


def resolve_dataset(root: Path, dataset_arg: str) -> DatasetInput:
    dataset_path = resolve_existing_file(
        [
            Path(dataset_arg),
            root / dataset_arg,
            root / "data/games" / dataset_arg,
            root / "data/training" / dataset_arg,
            root / "data/nnue" / dataset_arg,
            root / "data/steinbeisser_training" / dataset_arg,
        ],
        "dataset",
    )
    dataset_stem = dataset_path.stem
    dataset_slug = slugify(dataset_stem)
    dataset_kind = read_abapack_kind(dataset_path)
    if dataset_kind == "prepared":
        return DatasetInput(
            source_path=dataset_path,
            prepared_path=dataset_path,
            dataset_stem=dataset_stem,
            dataset_slug=dataset_slug,
            sibling_unique_shard=False,
        )
    if dataset_kind == "raw":
        fail(
            f"raw abapack {dataset_path} is not supported here; use a prepared shard or the source games json"
        )
    if dataset_path.suffix != ".json":
        fail(
            f"unsupported dataset input {dataset_path}; expected a games json or prepared abapack"
        )
    unique_path = dataset_path.with_name(f"{dataset_stem}_unique.abapack")
    if not unique_path.is_file():
        fail(f"missing sibling unique prepared shard {unique_path}; run slicer first")
    if read_abapack_kind(unique_path.resolve()) != "prepared":
        fail(f"sibling unique shard {unique_path} exists, but it is not a prepared abapack")
    return DatasetInput(
        source_path=dataset_path,
        prepared_path=unique_path.resolve(),
        dataset_stem=dataset_stem,
        dataset_slug=dataset_slug,
        sibling_unique_shard=True,
    )


def resolve_engine_path(root: Path, engine_arg: str) -> Path:
    return resolve_existing_file(
        [
            Path(engine_arg),
            root / engine_arg,
            root / "CodinGame" / engine_arg,
        ],
        "engine source",
    )


WALL_BOOK_HASH_SEED = 0xABA1_0BAD_CAFE_5EED
WALL_ROW_LENGTHS = (5, 6, 7, 8, 9, 8, 7, 6, 5)
WALL_SYMBOLS = frozenset({"B", "W", "."})


def resolve_checkpoint_book(root: Path, checkpoint_book_arg: str | None = None) -> Path:
    if checkpoint_book_arg is not None:
        return resolve_existing_file(
            [
                Path(checkpoint_book_arg),
                root / checkpoint_book_arg,
                root / "data/positions" / checkpoint_book_arg,
            ],
            "checkpoint screening book",
        )
    return resolve_existing_file(
        [
            root / "data/positions/opening_book_200K.txt",
            root / "data/positions/opening_book_200k.txt",
            root / "data/positions/opening_book_100K.txt",
            root / "data/positions/opening_book_100k.txt",
        ],
        "checkpoint screening book",
    )


def wall_variations_source_candidates(root: Path) -> list[Path]:
    return [
        root / "data/positions/wall_variations.txt",
        Path("/Users/jonas/Downloads/wall_variations.txt"),
    ]


def resolve_wall_variations_source(root: Path) -> Path:
    return resolve_existing_file(wall_variations_source_candidates(root), "wall variations source")


def wall_row_offset(row: int) -> int:
    return sum(WALL_ROW_LENGTHS[:row])


def wall_coord_name(row: int, column: int) -> str:
    return f"{chr(ord('A') + row)}{column}"


def wall_index_for_coord(row: int, column: int) -> int:
    if row < 0 or row >= len(WALL_ROW_LENGTHS):
        fail(f"wall coordinate row out of range: {row}")
    max_column = WALL_ROW_LENGTHS[row]
    if column <= 0 or column > max_column:
        fail(f"wall coordinate column out of range for row {row}: {column}")
    return wall_row_offset(row) + (column - 1)


def wall_next_splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E37_79B9_7F4A_7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
    value ^= value >> 31
    return state, value & MASK64


def wall_hash_keys() -> tuple[int, list[int], list[int]]:
    state = WALL_BOOK_HASH_SEED
    state, side_to_move_key = wall_next_splitmix64(state)
    black_keys: list[int] = []
    white_keys: list[int] = []
    for _ in range(sum(WALL_ROW_LENGTHS)):
        state, key = wall_next_splitmix64(state)
        black_keys.append(key)
    for _ in range(sum(WALL_ROW_LENGTHS)):
        state, key = wall_next_splitmix64(state)
        white_keys.append(key)
    return side_to_move_key, black_keys, white_keys


def compute_wall_position_key(turn: str, black_bits: int, white_bits: int) -> int:
    side_to_move_key, black_keys, white_keys = wall_hash_keys()
    key = 0
    if turn == "white":
        key ^= side_to_move_key
    bits = black_bits
    while bits:
        index = (bits & -bits).bit_length() - 1
        key ^= black_keys[index]
        bits &= bits - 1
    bits = white_bits
    while bits:
        index = (bits & -bits).bit_length() - 1
        key ^= white_keys[index]
        bits &= bits - 1
    return key & MASK64


def parse_wall_variations(source_path: Path) -> list[dict[str, object]]:
    positions: list[dict[str, object]] = []
    lines = source_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        if not raw or raw.startswith("#"):
            index += 1
            continue
        if not raw.startswith("==="):
            fail(f"unexpected wall variations line {index + 1}: {lines[index]}")
        name = raw.strip("= ").strip()
        index += 1
        turn: str | None = None
        board_rows: list[list[str]] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                break
            if stripped.startswith("==="):
                break
            if ":" in stripped:
                key, value = [part.strip() for part in stripped.split(":", 1)]
                if key == "turn":
                    turn = value.lower()
                index += 1
                continue
            tokens = stripped.split()
            if not tokens or any(token not in WALL_SYMBOLS for token in tokens):
                fail(f"invalid wall board row in {source_path}: {lines[index]}")
            board_rows.append(tokens)
            index += 1
        if turn not in {"black", "white"}:
            fail(f"wall variation {name} is missing a valid turn")
        if len(board_rows) != len(WALL_ROW_LENGTHS):
            fail(
                f"wall variation {name} has {len(board_rows)} board rows, expected {len(WALL_ROW_LENGTHS)}"
            )
        black_coords: list[str] = []
        white_coords: list[str] = []
        black_bits = 0
        white_bits = 0
        for row, (tokens, expected_len) in enumerate(zip(board_rows, WALL_ROW_LENGTHS)):
            if len(tokens) != expected_len:
                fail(
                    f"wall variation {name} row {row} has {len(tokens)} cells, expected {expected_len}"
                )
            for column, token in enumerate(tokens, start=1):
                coord = wall_coord_name(row, column)
                cell_bit = 1 << wall_index_for_coord(row, column)
                if token == "B":
                    black_coords.append(coord)
                    black_bits |= cell_bit
                elif token == "W":
                    white_coords.append(coord)
                    white_bits |= cell_bit
        if black_bits & white_bits:
            fail(f"wall variation {name} contains overlapping marbles")
        positions.append(
            {
                "name": name,
                "turn": turn,
                "black_coords": black_coords,
                "white_coords": white_coords,
                "black_bits": black_bits,
                "white_bits": white_bits,
                "zobrist": compute_wall_position_key(turn, black_bits, white_bits),
            }
        )
    if not positions:
        fail(f"wall variations source {source_path} did not contain any positions")
    return positions


def render_wall_variations_book(source_path: Path) -> str:
    positions = parse_wall_variations(source_path)
    lines = [
        "# Wall variation opening book generated by trainer-v5.",
        f"# Source: {source_path}",
        f"# Positions: {len(positions)}.",
        "",
    ]
    for entry in positions:
        lines.extend(
            [
                f"name: {entry['name']}",
                f"turn: {entry['turn']}",
                f"zobrist: {int(entry['zobrist']):016x}",
                f"black_bits: {int(entry['black_bits']):016x}",
                f"white_bits: {int(entry['white_bits']):016x}",
                "black: " + (",".join(entry["black_coords"]) if entry["black_coords"] else "-"),
                "white: " + (",".join(entry["white_coords"]) if entry["white_coords"] else "-"),
                "",
            ]
        )
    return "\n".join(lines)


def ensure_wall_variations_book(root: Path) -> Path:
    source_path = resolve_wall_variations_source(root)
    output_path = root / "data/positions" / "wall_variations_book.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_wall_variations_book(source_path)
    if not output_path.is_file() or output_path.read_text(encoding="utf-8") != rendered:
        output_path.write_text(rendered, encoding="utf-8")
    return output_path.resolve()


def ensure_repo_binary(root: Path, manifest_path: Path, bin_name: str) -> Path:
    target_root = (
        Path(os.environ["CARGO_TARGET_DIR"])
        if os.environ.get("CARGO_TARGET_DIR")
        else manifest_path.parent / "target"
    )
    binary_path = target_root / "release" / bin_name
    if binary_path.is_file():
        return binary_path.resolve()
    status(f"trainer: cargo build {manifest_path.parent.name}:{bin_name}")
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            str(manifest_path),
            "--bin",
            bin_name,
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        parts: list[str] = []
        if result.stdout.strip():
            parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.strip()}")
        suffix = f"\n{chr(10).join(parts)}" if parts else ""
        fail(f"cargo build failed for {bin_name} with status {result.returncode}{suffix}")
    if not binary_path.is_file():
        fail(f"expected binary at {binary_path}")
    return binary_path.resolve()


def resolve_backend(args: CliArgs) -> str:
    return args.backend if args.backend is not None else DEFAULT_BACKEND


def sample_count_label(value: int) -> str:
    if value == 0:
        return "0"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def score_filter_tag(max_abs_score: int | None) -> str:
    return "absall" if max_abs_score is None else f"abs{max_abs_score}"


def parse_sample_count(text: str) -> int:
    value = text.strip().lower().replace("_", "")
    if not value:
        fail("sample count must not be empty")
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("m"):
        multiplier = 1_000_000
        value = value[:-1]
    if not value or not value.isdigit():
        fail(f"invalid sample count {text!r}; use integers or k/m suffixes")
    parsed = int(value) * multiplier
    if parsed <= 0:
        fail(f"sample count must be positive, got {text!r}")
    return parsed


def parse_max_abs_score(text: str) -> int | None:
    value = text.strip().replace("_", "")
    if not value or not value.isdigit():
        raise argparse.ArgumentTypeError("max abs score must be a non-negative integer")
    parsed = int(value)
    return None if parsed == 0 else parsed


def parse_active_learning_chunks(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        fail("active-learning chunk schedule must contain at least one chunk")
    chunks = tuple(parse_sample_count(part) for part in parts)
    if any(value <= 0 for value in chunks):
        fail("active-learning chunks must all be positive")
    return chunks


def active_learning_plan_text(active_learning: ActiveLearningSpec, pool_train_samples: int) -> str:
    if active_learning.manual_chunk_sizes is not None:
        chunks = ",".join(sample_count_label(value) for value in active_learning.manual_chunk_sizes)
        final_train = sample_count_label(sum(active_learning.manual_chunk_sizes))
        return f"chunks={chunks} pool_train={pool_train_samples} final_train={final_train}"
    if active_learning.auto_chunk_size is not None:
        return (
            f"chunk_size={sample_count_label(active_learning.auto_chunk_size)} "
            f"pool_train={pool_train_samples} final_train=auto"
        )
    fail("active-learning spec is missing both chunk schedule and chunk size")


def runtime_loss_interval_for_spec(spec: CandidateSpec) -> int:
    if spec.runtime_loss_interval is not None:
        return spec.runtime_loss_interval
    return ACTIVE_LEARNING_RUNTIME_LOSS_INTERVAL if spec.active_learning is not None else DEFAULT_RUNTIME_LOSS_INTERVAL


def patience_for_spec(spec: CandidateSpec) -> int:
    if spec.patience is not None:
        return spec.patience
    if spec.active_learning is not None:
        return ACTIVE_LEARNING_PATIENCE
    return DEFAULT_PATIENCE


def weight_tag(value: float) -> str:
    return f"{max(0, round(value * 1000)):04d}"


def learning_rate_tag(value: float) -> str:
    return f"{max(0, round(value * 1_000_000)):06d}"


def weight_decay_tag(value: float) -> str:
    return f"{max(0, round(value * 1_000_000)):06d}"


def ema_decay_tag(value: float) -> str:
    return f"{max(0, round(value * 10_000)):04d}"


def dropout_tag(value: float) -> str:
    return f"{max(0, round(value * 1000)):03d}"


def hard_link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_order_value(name: str) -> int:
    return {"micro": 0, "fast": 1, "probe": 2, "screen": 3, "full": 4}.get(name, 99)


def recipe_name_for_spec(recipe: RecipeSpec) -> str:
    for name, known in RECIPE_SPECS.items():
        if recipe == known:
            return name
    return recipe.name


def experiment_root_for_candidate(root: Path, spec: CandidateSpec) -> Path:
    def compact_component(text: str, max_len: int = 220) -> str:
        if len(text) <= max_len:
            return text
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        keep = max_len - len(digest) - 1
        return f"{text[:keep]}_{digest}"

    lr_suffix = f"_lr{learning_rate_tag(spec.recipe.learning_rate)}"
    wd_suffix = f"_wd{weight_decay_tag(spec.recipe.weight_decay)}"
    dropout_suffix = f"_do{dropout_tag(spec.recipe.dropout)}"
    lambda_mix_suffix = (
        ""
        if spec.lambda_mix == DEFAULT_LAMBDA_MIX
        else f"_lm{weight_tag(spec.lambda_score)}_{weight_tag(spec.lambda_result)}"
    )
    source_suffix = (
        ""
        if spec.engine_template_path.stem == "steinbeisser-rc5"
        else f"_src{slugify(spec.engine_template_path.stem)}"
    )
    feature_set_suffix = (
        ""
        if spec.feature_set_name == MODEL_INPUTS
        else f"_fs{feature_set_experiment_tag(spec.feature_set_name)}"
    )
    architecture_suffix = "" if spec.architecture_spec == ARCHITECTURE else f"_arch{spec.architecture_spec.replace(',', 'x')}"
    activation_suffix = "" if spec.activation_name == "relu" else f"_act{spec.activation_name}"
    dense_mask_suffix = (
        ""
        if not spec.dense_feature_mask
        else f"_dm{'_'.join(str(index) for index in spec.dense_feature_mask)}"
    )
    repeat_occurrence_suffix = (
        ""
        if spec.repeat_occurrence_weight in {None, "none"}
        else f"_rep{slugify(spec.repeat_occurrence_weight)}"
    )
    class_weighting_suffix = (
        ""
        if spec.class_weighting == DEFAULT_CLASS_WEIGHTING
        else f"_cw{slugify(spec.class_weighting)}"
    )
    ema_decay_suffix = "" if spec.ema_decay is None else f"_ema{ema_decay_tag(spec.ema_decay)}"
    score_filter_suffix = f"_{score_filter_tag(spec.max_abs_score)}"
    runtime_loss_suffix = (
        ""
        if spec.runtime_loss_interval is None or spec.runtime_loss_interval == DEFAULT_RUNTIME_LOSS_INTERVAL
        else f"_rt{spec.runtime_loss_interval}"
    )
    patience_suffix = "" if spec.patience is None else f"_pat{spec.patience}"
    checkpoint_start_suffix = (
        ""
        if spec.checkpoint_start_epoch is None
        or spec.tier.checkpoint_interval_epochs is None
        or spec.checkpoint_start_epoch == spec.tier.checkpoint_interval_epochs
        else f"_ckptstart{spec.checkpoint_start_epoch}"
    )
    backend_suffix = "" if spec.backend == DEFAULT_BACKEND else f"_{spec.backend}"
    active_learning_suffix = ""
    pool_suffix = ""
    if spec.active_learning is not None:
        active_learning_suffix = f"_al{spec.active_learning.chunk_label}"
        if spec.active_learning.pool_cutoff_cp is not None:
            active_learning_suffix += f"_poolcap{spec.active_learning.pool_cutoff_cp}"
        pool_suffix = f"_pool{sample_count_label(spec.pool_train_samples)}"
    prefix = spec.sweep_name if spec.sweep_name is not None else "single"
    name = (
        f"{spec.dataset_slug}_{prefix}_{spec.tier.name}_{spec.profile.name}_{spec.recipe.name}"
        f"_seed{spec.seed}"
        f"{active_learning_suffix}"
        f"{pool_suffix}"
        f"{source_suffix}"
        f"_train{sample_count_label(spec.train_samples)}"
        f"_bs{spec.recipe.batch_size}"
        f"{lr_suffix}{wd_suffix}{dropout_suffix}"
        f"{feature_set_suffix}{architecture_suffix}{activation_suffix}{dense_mask_suffix}{repeat_occurrence_suffix}{class_weighting_suffix}{ema_decay_suffix}{score_filter_suffix}{runtime_loss_suffix}{patience_suffix}"
        f"{lambda_mix_suffix}"
        f"{checkpoint_start_suffix}"
        f"_ep{spec.epochs}"
        f"_val{sample_count_label(spec.validation_samples)}"
        f"_diag{sample_count_label(spec.diagnostic_validation_samples)}"
        f"{backend_suffix}"
    )
    name = compact_component(name)
    root = artifact_root(root)
    if spec.sweep_name is not None:
        return root / "data/nnue/experiments" / f"{spec.dataset_slug}_{spec.sweep_name}_seed{spec.seed}" / "candidates" / name
    return root / "data/nnue/experiments" / name


def sweep_root(root: Path, dataset_slug: str, sweep_name: str, seed: int) -> Path:
    root = artifact_root(root)
    return root / "data/nnue/experiments" / f"{dataset_slug}_{sweep_name}_seed{seed}"


def candidate_source_path(experiment_root: Path, engine_stem: str, candidate_id: str) -> Path:
    return experiment_root / f"{engine_stem}-{candidate_id}.rs"


def checkpoint_candidate_source_path(
    experiment_root: Path,
    engine_stem: str,
    candidate_id: str,
) -> Path:
    return experiment_root / "checkpoint_sources" / f"{engine_stem}-{candidate_id}-checkpoint.rs"


def ranked_checkpoint_candidate_source_path(
    experiment_root: Path,
    engine_stem: str,
    candidate_id: str,
    epoch: int,
) -> Path:
    return experiment_root / "checkpoint_sources" / f"{engine_stem}-{candidate_id}-epoch{epoch:04d}.rs"


def run_json_command(root: Path, command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        parts: list[str] = []
        if result.stdout.strip():
            parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.strip()}")
        suffix = f"\n{chr(10).join(parts)}" if parts else ""
        fail(f"command failed with status {result.returncode}{suffix}")
    return result.stdout


def encode_ascii85(payload: bytes) -> str:
    encoded: list[str] = []
    full_length = (len(payload) // 4) * 4
    for offset in range(0, full_length, 4):
        value = int.from_bytes(payload[offset : offset + 4], "big")
        if value == 0:
            encoded.append("z")
            continue
        digits = [0] * 5
        current = value
        for index in range(4, -1, -1):
            digits[index] = current % 85
            current //= 85
        encoded.extend(chr(digit + 33) for digit in digits)
    remainder = payload[full_length:]
    if remainder:
        padded = remainder + (b"\x00" * (4 - len(remainder)))
        current = int.from_bytes(padded, "big")
        digits = [0] * 5
        for index in range(4, -1, -1):
            digits[index] = current % 85
            current //= 85
        encoded.extend(chr(digit + 33) for digit in digits[: len(remainder) + 1])
    return "".join(encoded)


def render_nnue_model_const(payload: str) -> str:
    for hash_count in range(2, 33):
        hashes = "#" * hash_count
        if f'"{hashes}' not in payload:
            return f'const NNUE_MODEL_A85: &str = r{hashes}"{payload}"{hashes};'
    fail("unable to choose a safe raw-string delimiter for NNUE payload")


def bake_nnq_into_source(template_source: Path, nnq_path: Path, output_source: Path) -> None:
    template = template_source.read_text(encoding="utf-8")
    payload = encode_ascii85(nnq_path.read_bytes())
    replacement = render_nnue_model_const(payload)
    needle = 'const NNUE_MODEL_A85: &str = r'
    start = template.find(needle)
    if start < 0:
        fail(f"{template_source} does not contain a supported NNUE_MODEL_A85 constant")
    open_quote = template.find('"', start)
    if open_quote < 0:
        fail("failed to find opening quote for NNUE_MODEL_A85")
    hash_count = 0
    index = open_quote - 1
    while index >= start and template[index] == "#":
        hash_count += 1
        index -= 1
    close_marker = '"' + ("#" * hash_count) + ";"
    payload_start = open_quote + 1
    close_index = template.find(close_marker, payload_start)
    if close_index < 0:
        fail("failed to find closing marker for NNUE_MODEL_A85")
    rewritten = template[:start] + replacement + template[close_index + len(close_marker) :]
    output_source.write_text(rewritten, encoding="utf-8")


def read_runtime_nnq_architecture(nnq_path: Path) -> tuple[int, int, int, int]:
    data = nnq_path.read_bytes()
    if len(data) < 4 or data[:4] != b"NNQ1":
        fail(f"{nnq_path} is not a supported NNQ runtime model")
    offset = 4
    try:
        (
            version,
            _feature_set_id,
            _activation_id,
            _scalar_backend_id,
            _target_transform_id,
            sparse_input_count,
            dense_input_count,
            hidden_layer_count,
        ) = struct.unpack_from("<HBBBBIII", data, offset)
        offset += struct.calcsize("<HBBBBIII")
        if hidden_layer_count < 2:
            fail(f"{nnq_path} has unsupported hidden layer count {hidden_layer_count}")
        hidden_sizes = list(struct.unpack_from(f"<{hidden_layer_count}I", data, offset))
        offset += 4 * hidden_layer_count
        (_sparse_scale, dense_scale_count) = struct.unpack_from("<fI", data, offset)
        offset += struct.calcsize("<fI")
        offset += 4 * dense_scale_count
        (dense_offset_count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        offset += 4 * dense_offset_count
        offset += 4  # output bias
        if version in {4, 5, 7}:
            offset += 4 * hidden_layer_count
        offset += 4 * hidden_sizes[0]  # quantized accumulator biases
        offset += 2 * (sparse_input_count * hidden_sizes[0])  # sparse weights
        offset += 4 * (dense_input_count * hidden_sizes[0])  # dense weights
        (h0_padded_input_size, _hidden_weight_scale) = struct.unpack_from("<If", data, offset)
        offset += struct.calcsize("<If")
        offset += 4 * hidden_sizes[1]  # hidden biases
        offset += int(h0_padded_input_size) * hidden_sizes[1]  # hidden weights
        (h1_padded_input_size, _output_weight_scale) = struct.unpack_from("<If", data, offset)
    except struct.error as exc:
        fail(f"failed to parse NNQ runtime header from {nnq_path}: {exc}")
    return hidden_sizes[0], hidden_sizes[1], int(h0_padded_input_size), int(h1_padded_input_size)


def patch_runtime_arch_constants_in_source(
    source_path: Path,
    nnq_path: Path,
) -> None:
    nh, nnue_h1, nnue_h0_pad, nnue_h1_pad = read_runtime_nnq_architecture(nnq_path)
    text = source_path.read_text(encoding="utf-8")
    replacements = (
        (r"(?m)^const Nh: usize = \d+;$", f"const Nh: usize = {nh};"),
        (r"(?m)^const NNUE_H1: usize = \d+;$", f"const NNUE_H1: usize = {nnue_h1};"),
        (r"(?m)^const NNUE_H0_PAD: usize = \d+;$", f"const NNUE_H0_PAD: usize = {nnue_h0_pad};"),
        (r"(?m)^const NNUE_H1_PAD: usize = \d+;$", f"const NNUE_H1_PAD: usize = {nnue_h1_pad};"),
    )
    rewritten = text
    for pattern, replacement in replacements:
        rewritten, count = re.subn(pattern, replacement, rewritten, count=1)
        if count != 1:
            fail(f"failed to patch runtime NNUE architecture constant {pattern} into {source_path}")
    source_path.write_text(rewritten, encoding="utf-8")


def patch_dense_feature_mask_in_source(source_path: Path, dense_indices: Sequence[int]) -> None:
    if not dense_indices:
        return
    text = source_path.read_text(encoding="utf-8")
    needle = (
        "    let mut i = 0;\n"
        "    while i < _E {\n"
        "        dense[i] = clamp_feature((raw[i] - df[i]) / ds[i].max(1.0));\n"
        "        i += 1;\n"
        "    }\n"
        "    dense\n"
    )
    mask_lines = "".join(f"    dense[{index}] = 0.0;\n" for index in dense_indices)
    replacement = (
        "    let mut i = 0;\n"
        "    while i < _E {\n"
        "        dense[i] = clamp_feature((raw[i] - df[i]) / ds[i].max(1.0));\n"
        "        i += 1;\n"
        "    }\n"
        f"{mask_lines}"
        "    dense\n"
    )
    if needle not in text:
        fail(f"failed to patch dense feature mask into {source_path}")
    source_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def bake_runtime_source(
    template_source: Path,
    nnq_path: Path,
    output_source: Path,
    dense_feature_mask: Sequence[int],
) -> None:
    bake_nnq_into_source(template_source, nnq_path, output_source)
    patch_runtime_arch_constants_in_source(output_source, nnq_path)
    patch_dense_feature_mask_in_source(output_source, dense_feature_mask)


def stream_training_output(stream: TextIO) -> None:
    blob_filter = JsonBlobFilter()
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            filtered = blob_filter.filter(line)
            if filtered is not None:
                sys.stdout.write(filtered)
                sys.stdout.flush()
    finally:
        stream.close()


def parse_metric(text: str) -> float:
    value = text.strip().lower()
    if value in {"inf", "+inf"}:
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(text.strip())


def format_metric(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.2f}"


def parse_summary_value(summary_text: str, key: str) -> str | None:
    import re

    match = re.search(rf"^{re.escape(key)}:\s+(\S+)", summary_text, re.MULTILINE)
    return match.group(1) if match is not None else None


def parse_display_number(raw: str | None) -> float:
    if raw is None:
        return 0.0
    value = raw.strip()
    if value in {"", "--", "n/a"}:
        return 0.0
    multiplier = 1.0
    if value.endswith("%"):
        value = value[:-1]
    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    return float(value.replace(",", "")) * multiplier


def parse_match_summary(summary_text: str, run_path: Path, game_count: int) -> MatchResult:
    import re

    wdl_match = re.search(r"current_vs_reference:\s+(\d+)W\s+(\d+)D\s+(\d+)L", summary_text)
    elo_match = re.search(
        r"elo\(current-reference\):\s+(\S+)\s+\[(\S+),\s+(\S+)\]\s+\(([^)]+)\)",
        summary_text,
    )
    if wdl_match is None or elo_match is None:
        fail(f"failed to parse match summary:\n{summary_text}")
    wins = int(wdl_match.group(1))
    draws = int(wdl_match.group(2))
    losses = int(wdl_match.group(3))
    score_pct = ((wins + 0.5 * draws) / game_count * 100.0) if game_count else 0.0
    return MatchResult(
        run_path=run_path,
        wins=wins,
        draws=draws,
        losses=losses,
        score_pct=score_pct,
        elo=parse_metric(elo_match.group(1)),
        elo_lower=parse_metric(elo_match.group(2)),
        elo_upper=parse_metric(elo_match.group(3)),
        avg_depth=parse_display_number(parse_summary_value(summary_text, "current_timed_depth")),
        avg_nps=parse_display_number(parse_summary_value(summary_text, "current_timed_nps")),
        avg_response_ms=parse_display_number(
            parse_summary_value(summary_text, "current_timed_time_ms_mean")
        ),
        decision=elo_match.group(4),
        games=game_count,
    )


def positions_for_game_count(game_count: int) -> int:
    if game_count <= 0 or game_count % 2 != 0:
        fail(f"match game count must be a positive even number, got {game_count}")
    return game_count // 2


def run_match(
    root: Path,
    match_bin: Path,
    candidate_source: Path,
    reference_source: Path,
    book_path: Path,
    time_ms: int,
    game_count: int,
) -> MatchResult:
    output = run_json_command(
        root,
        [
            str(match_bin),
            "--candidate-source",
            str(candidate_source),
            "--reference-source",
            str(reference_source),
            "--book",
            str(book_path),
            "--positions",
            str(positions_for_game_count(game_count)),
            "--time-ms",
            str(time_ms),
            "--games-per-opening",
            "1",
            "--json",
            "--keep-artifacts",
        ],
    )
    payload = json.loads(output)
    run_path = Path(str(payload["run_path"])).resolve()
    return parse_match_summary(str(payload["summary"]), run_path, game_count)


def run_runtime_loss(
    root: Path,
    nnue_bin: Path,
    model_path: Path,
    dataset_path: Path,
    lambda_score: float,
) -> dict[str, object]:
    output = run_json_command(
        root,
        [
            str(nnue_bin),
            "runtime-loss",
            "--model",
            str(model_path),
            "--dataset",
            str(dataset_path),
            "--lambda",
            str(lambda_score),
        ],
    )
    return json.loads(output)


def run_model_match(
    root: Path,
    match_bin: Path,
    engine_template_path: Path,
    reference_source_path: Path,
    output_source_path: Path,
    model_nnq: Path,
    dense_feature_mask: Sequence[int],
    book_path: Path,
    time_ms: int,
    game_count: int,
) -> MatchResult:
    status(f"trainer: baking {model_nnq} into {output_source_path}")
    bake_runtime_source(engine_template_path, model_nnq, output_source_path, dense_feature_mask)
    status(
        "trainer: validating "
        f"{output_source_path} against {reference_source_path} on {book_path} "
        f"({game_count} games)"
    )
    return run_match(
        root=root,
        match_bin=match_bin,
        candidate_source=output_source_path,
        reference_source=reference_source_path,
        book_path=book_path,
        time_ms=time_ms,
        game_count=game_count,
    )


def write_u8(handle: BinaryIO, value: int) -> None:
    handle.write(U8.pack(value))


def write_u16(handle: BinaryIO, value: int) -> None:
    handle.write(U16.pack(value))


def write_u32(handle: BinaryIO, value: int) -> None:
    handle.write(U32.pack(value))


def write_u64(handle: BinaryIO, value: int) -> None:
    handle.write(U64.pack(value))


def write_i8(handle: BinaryIO, value: int) -> None:
    handle.write(I8.pack(value))


def write_f32(handle: BinaryIO, value: float) -> None:
    handle.write(F32.pack(float(value)))


def write_bool(handle: BinaryIO, value: bool) -> None:
    write_u8(handle, 1 if value else 0)


def write_string(handle: BinaryIO, value: str) -> None:
    data = value.encode("utf-8")
    write_u32(handle, len(data))
    handle.write(data)


def write_optional_u8(handle: BinaryIO, value: int | None) -> None:
    write_bool(handle, value is not None)
    if value is not None:
        write_u8(handle, value)


def write_optional_u32(handle: BinaryIO, value: int | None) -> None:
    write_bool(handle, value is not None)
    if value is not None:
        write_u32(handle, value)


def write_optional_f32(handle: BinaryIO, value: float | None) -> None:
    write_bool(handle, value is not None)
    if value is not None:
        write_f32(handle, value)


def write_optional_string(handle: BinaryIO, value: str | None) -> None:
    write_bool(handle, value is not None)
    if value is not None:
        write_string(handle, value)


def read_exact(handle: BinaryIO, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise EOFError("unexpected end of file")
    return payload


def read_u8(handle: BinaryIO) -> int:
    return U8.unpack(read_exact(handle, 1))[0]


def read_u16(handle: BinaryIO) -> int:
    return U16.unpack(read_exact(handle, 2))[0]


def read_u32(handle: BinaryIO) -> int:
    return U32.unpack(read_exact(handle, 4))[0]


def read_u32_or_eof(handle: BinaryIO) -> int | None:
    payload = handle.read(4)
    if payload == b"":
        return None
    if len(payload) != 4:
        raise EOFError("truncated u32 at EOF")
    return U32.unpack(payload)[0]


def read_u64(handle: BinaryIO) -> int:
    return U64.unpack(read_exact(handle, 8))[0]


def read_i8(handle: BinaryIO) -> int:
    return I8.unpack(read_exact(handle, 1))[0]


def read_f32(handle: BinaryIO) -> float:
    return F32.unpack(read_exact(handle, 4))[0]


def read_bool(handle: BinaryIO) -> bool:
    return read_u8(handle) != 0


def read_string(handle: BinaryIO) -> str:
    size = read_u32(handle)
    return read_exact(handle, size).decode("utf-8")


def read_optional_u8(handle: BinaryIO) -> int | None:
    return read_u8(handle) if read_bool(handle) else None


def read_optional_u32(handle: BinaryIO) -> int | None:
    return read_u32(handle) if read_bool(handle) else None


def read_optional_f32(handle: BinaryIO) -> float | None:
    return read_f32(handle) if read_bool(handle) else None


def read_optional_string(handle: BinaryIO) -> str | None:
    return read_string(handle) if read_bool(handle) else None


def read_prepared_header(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(10)
    if len(header) != 10 or header[:8] != ABAPACK_MAGIC:
        fail(f"{path} has invalid abapack magic")
    version = header[8]
    kind = header[9]
    if version <= 0 or version > ABAPACK_VERSION:
        fail(f"{path} has unsupported abapack version {version}")
    if kind != ABAPACK_KIND_PREPARED:
        fail(f"{path} is not a prepared abapack")
    return version


class PreparedWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("wb")
        self.handle.write(ABAPACK_MAGIC)
        self.handle.write(bytes([ABAPACK_VERSION, ABAPACK_KIND_PREPARED]))

    def write_chain(self, chain: PreparedChain) -> None:
        write_u32(self.handle, len(chain.samples))
        write_string(self.handle, chain.run_file)
        write_u32(self.handle, chain.game_index)
        write_string(self.handle, chain.opening_name)
        write_string(self.handle, chain.opening_position)
        write_u64(self.handle, chain.opening_hash)
        split_byte = ABAPACK_SPLIT_TO_BYTE.get(chain.split)
        if split_byte is None:
            fail(f"unsupported prepared split {chain.split}")
        write_u8(self.handle, split_byte)
        for sample in chain.samples:
            write_u64(self.handle, sample.black_bits)
            write_u64(self.handle, sample.white_bits)
            write_u8(self.handle, ord(sample.side_to_move))
            write_u64(self.handle, sample.position_key)
            write_f32(self.handle, sample.mean_score)
            write_f32(self.handle, sample.mean_clipped_score)
            write_f32(self.handle, sample.mean_result)
            write_f32(self.handle, sample.mean_ply)
            write_f32(self.handle, sample.effective_game_turns_played)
            write_u32(self.handle, sample.occurrence_count)
            write_f32(self.handle, sample.sample_weight)
            write_u32(self.handle, sample.win_count)
            write_u32(self.handle, sample.draw_count)
            write_u32(self.handle, sample.loss_count)
            write_i8(self.handle, sample.result_bucket)
            write_optional_f32(self.handle, sample.mean_completed_depth)
            write_optional_f32(self.handle, sample.mean_no_progress_plies)
            write_optional_f32(self.handle, sample.ejection_rate)
            write_optional_f32(self.handle, sample.recorded_mean_score)
            write_optional_string(self.handle, sample.label_source)
            write_optional_u32(self.handle, sample.label_budget_ms)
            write_optional_u8(self.handle, sample.label_depth)
            write_optional_string(self.handle, sample.source_dataset)

    def close(self) -> None:
        self.handle.close()


class TeacherTargetReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("rb")
        magic = self.handle.read(8)
        if magic != TEACHER_TARGET_MAGIC:
            self.handle.close()
            fail(f"{path} has invalid teacher target magic")
        version = read_u16(self.handle)
        if version != TEACHER_TARGET_VERSION:
            self.handle.close()
            fail(f"{path} has unsupported teacher target version {version}")
        self.remaining = read_u64(self.handle)

    def next_score_for(self, position_key: int) -> float:
        if self.remaining <= 0:
            fail(f"{self.path} ended early before prepared sample position_key {position_key}")
        cached_position_key = read_u64(self.handle)
        cached_score = read_f32(self.handle)
        self.remaining -= 1
        if cached_position_key != int(position_key):
            fail(
                f"{self.path} position_key mismatch: expected {position_key}, "
                f"found {cached_position_key}"
            )
        return float(cached_score)

    def close(self) -> None:
        self.handle.close()

    def finish(self) -> None:
        try:
            if self.remaining != 0:
                fail(f"{self.path} still contains {self.remaining} unused teacher targets")
        finally:
            self.close()

    def __enter__(self) -> "TeacherTargetReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.close()


def create_empty_prepared_abapack(path: Path) -> None:
    writer = PreparedWriter(path)
    writer.close()


def iter_prepared_chains(path: Path) -> Iterable[PreparedChain]:
    version = read_prepared_header(path)
    with path.open("rb") as handle:
        handle.seek(10)
        while True:
            sample_count = read_u32_or_eof(handle)
            if sample_count is None:
                return
            run_file = read_string(handle)
            game_index = read_u32(handle)
            opening_name = read_string(handle)
            opening_position = read_string(handle)
            opening_hash = read_u64(handle)
            split = ABAPACK_BYTE_TO_SPLIT.get(read_u8(handle))
            if split is None:
                fail(f"{path} contains an unsupported prepared split byte")
            samples: list[PreparedSample] = []
            for _ in range(sample_count):
                black_bits = read_u64(handle)
                white_bits = read_u64(handle)
                side_to_move = chr(read_u8(handle))
                position_key = read_u64(handle)
                mean_score = read_f32(handle)
                mean_clipped_score = read_f32(handle)
                mean_result = read_f32(handle)
                mean_ply = read_f32(handle)
                effective_game_turns_played = read_f32(handle) if version >= 5 else mean_ply
                occurrence_count = read_u32(handle)
                sample_weight = read_f32(handle)
                win_count = read_u32(handle)
                draw_count = read_u32(handle)
                loss_count = read_u32(handle)
                result_bucket = read_i8(handle)
                mean_completed_depth = read_optional_f32(handle) if version >= 4 else None
                mean_no_progress_plies = read_optional_f32(handle) if version >= 4 else None
                ejection_rate = read_optional_f32(handle) if version >= 4 else None
                recorded_mean_score = read_optional_f32(handle) if version >= 2 else None
                label_source = read_optional_string(handle) if version >= 2 else None
                label_budget_ms = read_optional_u32(handle) if version >= 2 else None
                label_depth = read_optional_u8(handle) if version >= 2 else None
                source_dataset = read_optional_string(handle) if version >= 2 else None
                samples.append(
                    PreparedSample(
                        black_bits=black_bits,
                        white_bits=white_bits,
                        side_to_move=side_to_move,
                        position_key=position_key,
                        mean_score=mean_score,
                        mean_clipped_score=mean_clipped_score,
                        mean_result=mean_result,
                        mean_ply=mean_ply,
                        effective_game_turns_played=effective_game_turns_played,
                        occurrence_count=occurrence_count,
                        sample_weight=sample_weight,
                        win_count=win_count,
                        draw_count=draw_count,
                        loss_count=loss_count,
                        result_bucket=result_bucket,
                        mean_completed_depth=mean_completed_depth,
                        mean_no_progress_plies=mean_no_progress_plies,
                        ejection_rate=ejection_rate,
                        recorded_mean_score=recorded_mean_score,
                        label_source=label_source,
                        label_budget_ms=label_budget_ms,
                        label_depth=label_depth,
                        source_dataset=source_dataset,
                    )
                )
            yield PreparedChain(
                run_file=run_file,
                game_index=game_index,
                opening_name=opening_name,
                opening_position=opening_position,
                opening_hash=opening_hash,
                split=split,
                samples=samples,
            )


def count_prepared_samples(path: Path) -> int:
    total = 0
    for chain in iter_prepared_chains(path):
        total += len(chain.samples)
    return total


def filtered_prepared_corpus_path(root: Path, dataset: DatasetInput, max_abs_score: int | None) -> Path:
    if max_abs_score is None:
        return dataset.prepared_path
    root = artifact_root(root)
    output_dir = (
        root
        / "data/training"
        / f"{canonical_dataset_slug(dataset)}_{score_filter_tag(max_abs_score)}_prepared_{SELECTION_CORPUS_VERSION}"
    )
    output_path = output_dir / "prepared.abapack"
    summary_path = output_dir / "summary.json"
    if output_path.is_file() and summary_path.is_file():
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".abapack.tmp")
    writer = PreparedWriter(tmp_path)
    input_samples = 0
    kept_samples = 0
    input_chains = 0
    kept_chains = 0
    try:
        for chain in iter_prepared_chains(dataset.prepared_path):
            input_chains += 1
            input_samples += len(chain.samples)
            kept = [
                sample
                for sample in chain.samples
                if abs(float(sample.mean_clipped_score)) <= float(max_abs_score)
            ]
            if not kept:
                continue
            kept_chains += 1
            kept_samples += len(kept)
            writer.write_chain(replace(chain, samples=kept))
    finally:
        writer.close()
    if kept_samples == 0:
        tmp_path.unlink(missing_ok=True)
        fail(f"{dataset.prepared_path} has no samples within abs score {max_abs_score}")
    tmp_path.replace(output_path)
    write_json(
        summary_path,
        {
            "source": str(dataset.prepared_path),
            "max_abs_score": max_abs_score,
            "input_chains": input_chains,
            "kept_chains": kept_chains,
            "dropped_chains": input_chains - kept_chains,
            "input_samples": input_samples,
            "kept_samples": kept_samples,
            "dropped_samples": input_samples - kept_samples,
        },
    )
    return output_path


def base_random_corpus_dir(
    root: Path,
    dataset_slug: str,
    seed: int,
    val_samples: int,
    max_abs_score: int | None,
) -> Path:
    root = artifact_root(root)
    return root / "data/training" / (
        f"{dataset_slug}_{score_filter_tag(max_abs_score)}"
        f"_base_random_val{sample_count_label(val_samples)}_seed{seed}"
    )


def load_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_base_random_corpus(
    root: Path,
    nnue_bin: Path,
    dataset: DatasetInput,
    seed: int,
    val_samples: int,
    max_abs_score: int | None,
) -> BaseCorpus:
    dataset_slug = canonical_dataset_slug(dataset)
    source_prepared_path = filtered_prepared_corpus_path(root, dataset, max_abs_score)
    corpus_dir = base_random_corpus_dir(root, dataset_slug, seed, val_samples, max_abs_score)
    train_path = corpus_dir / "train.abapack"
    val_path = corpus_dir / "val.abapack"
    manifest_path = corpus_dir / "manifest.json"
    if train_path.is_file() and val_path.is_file() and manifest_path.is_file():
        manifest = load_json_file(manifest_path)
        return BaseCorpus(
            corpus_dir=corpus_dir,
            train_path=train_path,
            val_path=val_path,
            manifest_path=manifest_path,
            manifest=manifest,
            train_samples=int(manifest["train"]["samples"]),
            val_samples=int(manifest["val"]["samples"]),
        )
    corpus_dir.parent.mkdir(parents=True, exist_ok=True)
    total_samples = count_prepared_samples(source_prepared_path)
    if total_samples <= val_samples:
        fail(
            f"{dataset.prepared_path} has only {total_samples} samples; need more than {val_samples}"
        )
    train_samples = total_samples - val_samples
    status(f"trainer: materializing base random corpus {corpus_dir}")
    run_json_command(
        root,
        [
            str(nnue_bin),
            "materialize-single-corpus",
            "--input",
            str(source_prepared_path),
            "--output-dir",
            str(corpus_dir),
            "--train-samples",
            str(train_samples),
            "--val-samples",
            str(val_samples),
            "--feature-set",
            MODEL_INPUTS,
            "--seed",
            str(seed),
        ],
    )
    manifest = load_json_file(manifest_path)
    return BaseCorpus(
        corpus_dir=corpus_dir,
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        manifest=manifest,
        train_samples=int(manifest["train"]["samples"]),
        val_samples=int(manifest["val"]["samples"]),
    )


def selection_index_path(base_corpus: BaseCorpus) -> Path:
    return base_corpus.corpus_dir / f"selection_index_v{SELECTION_INDEX_VERSION}.npz"


def score_bucket_for_abs_score(abs_score: float) -> int:
    return max(0, int(abs_score // SCORE_BUCKET_SIZE))


def ply_bucket_for_turns(turns: float) -> int:
    return max(0, int(max(0.0, turns) // PLY_BUCKET_SIZE))


def splitmix64_scalar(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def splitmix64_numpy(values: np.ndarray) -> np.ndarray:
    z = (values + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(MASK64)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(MASK64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(MASK64)
    return (z ^ (z >> np.uint64(31))) & np.uint64(MASK64)


def sample_identity_hash(chain: PreparedChain, sample: PreparedSample, sample_index: int) -> int:
    value = chain.opening_hash & MASK64
    value ^= splitmix64_scalar(chain.game_index + 0x9E37)
    value ^= splitmix64_scalar(sample.position_key)
    value ^= splitmix64_scalar(sample_index + 0xBF58)
    return splitmix64_scalar(value)


def ensure_selection_index(base_corpus: BaseCorpus) -> SelectionIndex:
    path = selection_index_path(base_corpus)
    if path.is_file():
        with np.load(path) as payload:
            return SelectionIndex(
                key_hashes=payload["key_hashes"],
                abs_scores=payload["abs_scores"],
                terminal_mask=payload["terminal_mask"].astype(bool),
                score_buckets=payload["score_buckets"].astype(np.int32),
                ply_buckets=payload["ply_buckets"].astype(np.int32),
                result_buckets=payload["result_buckets"].astype(np.int8),
                opening_hashes=payload["opening_hashes"].astype(np.uint64),
            )
    sample_count = int(base_corpus.manifest["train"]["samples"])
    key_hashes = np.empty(sample_count, dtype=np.uint64)
    abs_scores = np.empty(sample_count, dtype=np.float32)
    terminal_mask = np.empty(sample_count, dtype=np.bool_)
    score_buckets = np.empty(sample_count, dtype=np.int32)
    ply_buckets = np.empty(sample_count, dtype=np.int32)
    result_buckets = np.empty(sample_count, dtype=np.int8)
    opening_hashes = np.empty(sample_count, dtype=np.uint64)
    ordinal = 0
    for chain in iter_prepared_chains(base_corpus.train_path):
        for sample_index, sample in enumerate(chain.samples):
            abs_score = abs(sample.mean_clipped_score)
            key_hashes[ordinal] = np.uint64(sample_identity_hash(chain, sample, sample_index))
            abs_scores[ordinal] = np.float32(abs_score)
            terminal_mask[ordinal] = bool(abs_score >= TERMINAL_SCORE_THRESHOLD)
            score_buckets[ordinal] = score_bucket_for_abs_score(abs_score)
            ply_buckets[ordinal] = ply_bucket_for_turns(sample.effective_game_turns_played)
            result_buckets[ordinal] = np.int8(sample.result_bucket)
            opening_hashes[ordinal] = np.uint64(chain.opening_hash)
            ordinal += 1
    if ordinal != sample_count:
        fail(
            f"selection index sample count mismatch for {base_corpus.train_path}: expected {sample_count}, saw {ordinal}"
        )
    np.savez(
        path,
        key_hashes=key_hashes,
        abs_scores=abs_scores,
        terminal_mask=terminal_mask,
        score_buckets=score_buckets,
        ply_buckets=ply_buckets,
        result_buckets=result_buckets,
        opening_hashes=opening_hashes,
    )
    return SelectionIndex(
        key_hashes=key_hashes,
        abs_scores=abs_scores,
        terminal_mask=terminal_mask,
        score_buckets=score_buckets,
        ply_buckets=ply_buckets,
        result_buckets=result_buckets,
        opening_hashes=opening_hashes,
    )


def stable_tag_seed(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest(), "little")


def priority_array(
    key_hashes: np.ndarray,
    seed: int,
    tag: str,
    weights: np.ndarray,
) -> np.ndarray:
    seed_mix = splitmix64_scalar(seed ^ stable_tag_seed(tag))
    mixed = splitmix64_numpy(key_hashes ^ np.uint64(seed_mix))
    uniforms = (mixed.astype(np.float64) + 1.0) / UINT64_DENOMINATOR
    priorities = -np.log(uniforms) / weights
    return priorities


def profile_weight_array(abs_scores: np.ndarray, profile: ProfileSpec) -> np.ndarray:
    if profile.neutral_scale_cp is None:
        return np.ones(abs_scores.shape[0], dtype=np.float64)
    abs_scores_f64 = abs_scores.astype(np.float64, copy=False)
    return profile.tail_floor + (1.0 - profile.tail_floor) * np.exp(-abs_scores_f64 / profile.neutral_scale_cp)


def equal_quota_targets(
    capacities: np.ndarray,
    count: int,
    seed: int,
    tag: str,
    bucket_values: np.ndarray,
) -> np.ndarray:
    capacities_i64 = capacities.astype(np.int64, copy=True)
    total_available = int(capacities_i64.sum())
    if count > total_available:
        fail(f"requested {count} samples but only {total_available} are available")
    targets = np.zeros(capacities_i64.shape[0], dtype=np.int64)
    remaining = capacities_i64.copy()
    remaining_count = int(count)
    bucket_order = np.argsort(
        np.array(
            [
                splitmix64_scalar((int(value) & MASK64) ^ stable_tag_seed(f"{tag}:bucket:{index}") ^ seed)
                for index, value in enumerate(bucket_values.tolist())
            ],
            dtype=np.uint64,
        ),
        kind="stable",
    )
    while remaining_count > 0:
        active = np.flatnonzero(remaining > 0)
        if active.size == 0:
            fail("bucket allocation ran out of active buckets")
        share = remaining_count // active.size
        if share > 0:
            add = np.minimum(remaining[active], share)
            targets[active] += add
            remaining[active] -= add
            remaining_count -= int(add.sum())
            continue
        active_mask = np.zeros(remaining.shape[0], dtype=bool)
        active_mask[active] = True
        active_order = bucket_order[active_mask[bucket_order]]
        take = min(remaining_count, int(active_order.shape[0]))
        chosen = active_order[:take]
        targets[chosen] += 1
        remaining[chosen] -= 1
        remaining_count -= take
    return targets.astype(np.int32, copy=False)


def select_bucket_targets(
    eligible_indices: np.ndarray,
    bucket_values: np.ndarray,
    priorities: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    unique_buckets, inverse = np.unique(bucket_values, return_inverse=True)
    if unique_buckets.shape[0] != targets.shape[0]:
        fail("bucket target shape mismatch")
    parts: list[np.ndarray] = []
    for bucket_index, target in enumerate(targets.tolist()):
        if target <= 0:
            continue
        local_positions = np.flatnonzero(inverse == bucket_index)
        local_priorities = priorities[local_positions]
        chosen_local = np.argpartition(local_priorities, target - 1)[:target]
        parts.append(eligible_indices[local_positions[chosen_local]])
    if not parts:
        return np.empty(0, dtype=np.uint32)
    merged = np.concatenate(parts).astype(np.uint32, copy=False)
    merged.sort()
    return merged


def bucket_values_for_profile(index: SelectionIndex, profile: ProfileSpec) -> np.ndarray:
    if profile.bucket_axes == ("score_bucket",):
        return index.score_buckets.astype(np.int64, copy=False)
    if profile.bucket_axes == ("ply_bucket",):
        return index.ply_buckets.astype(np.int64, copy=False)
    if profile.bucket_axes == ("result_bucket",):
        return index.result_buckets.astype(np.int64, copy=False)
    if profile.bucket_axes == ("score_bucket", "ply_bucket"):
        return (
            index.score_buckets.astype(np.int64, copy=False) * 100_000
            + index.ply_buckets.astype(np.int64, copy=False)
        )
    fail(f"unsupported bucket axes for profile {profile.name}: {profile.bucket_axes}")


def select_bucket_balanced_indices(
    index: SelectionIndex,
    eligible_mask: np.ndarray,
    count: int,
    seed: int,
    tag: str,
    profile: ProfileSpec,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(eligible_mask).astype(np.uint32, copy=False)
    if count > int(eligible_indices.shape[0]):
        fail(f"requested {count} samples but only {eligible_indices.shape[0]} are available")
    bucket_values = bucket_values_for_profile(index, profile)[eligible_indices]
    priorities = priority_array(
        index.key_hashes[eligible_indices],
        seed,
        tag,
        np.ones(eligible_indices.shape[0], dtype=np.float64),
    )
    unique_buckets, counts = np.unique(bucket_values, return_counts=True)
    targets = equal_quota_targets(counts, count, seed, tag, unique_buckets)
    return select_bucket_targets(eligible_indices, bucket_values, priorities, targets)


def select_opening_capped_indices(
    index: SelectionIndex,
    eligible_mask: np.ndarray,
    count: int,
    seed: int,
    tag: str,
    profile: ProfileSpec,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(eligible_mask).astype(np.uint32, copy=False)
    if count > int(eligible_indices.shape[0]):
        fail(f"requested {count} samples but only {eligible_indices.shape[0]} are available")
    opening_values = index.opening_hashes[eligible_indices]
    unique_openings, counts = np.unique(opening_values, return_counts=True)
    if unique_openings.shape[0] == 0:
        fail("opening-diverse selection found no eligible openings")
    per_opening_cap = max(
        int(profile.opening_cap_min or 1),
        int(math.ceil((count / unique_openings.shape[0]) * float(profile.opening_cap_multiplier or 1.0))),
    )
    if profile.opening_cap_max is not None:
        per_opening_cap = min(per_opening_cap, int(profile.opening_cap_max))
    capped_counts = np.minimum(counts.astype(np.int64, copy=False), per_opening_cap).astype(np.int32, copy=False)
    capped_total = int(capped_counts.sum())
    priorities = priority_array(
        index.key_hashes[eligible_indices],
        seed,
        tag,
        np.ones(eligible_indices.shape[0], dtype=np.float64),
    )
    take_capped = min(count, capped_total)
    targets = equal_quota_targets(capped_counts, take_capped, seed, tag, unique_openings.astype(np.int64, copy=False))
    selected = select_bucket_targets(eligible_indices, opening_values.astype(np.int64, copy=False), priorities, targets)
    shortfall = count - int(selected.shape[0])
    if shortfall <= 0:
        return selected
    remaining_mask = eligible_mask.copy()
    remaining_mask[selected] = False
    filler = select_smallest(priorities=np.where(eligible_mask, priorities_full(index, eligible_indices, priorities), np.inf), count=shortfall, mask=remaining_mask)
    merged = np.concatenate([selected, filler.astype(np.uint32, copy=False)]).astype(np.uint32, copy=False)
    merged.sort()
    return merged


def priorities_full(
    index: SelectionIndex,
    eligible_indices: np.ndarray,
    eligible_priorities: np.ndarray,
) -> np.ndarray:
    full = np.full(index.sample_count, np.inf, dtype=np.float64)
    full[eligible_indices] = eligible_priorities
    return full


def select_smallest(
    priorities: np.ndarray,
    count: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.uint32)
    if mask is not None:
        available = int(mask.sum())
        if count > available:
            fail(f"requested {count} samples but only {available} are available")
        work = priorities.copy()
        work[~mask] = np.inf
    else:
        if count > priorities.shape[0]:
            fail(f"requested {count} samples but only {priorities.shape[0]} are available")
        work = priorities
    selected = np.argpartition(work, count - 1)[:count]
    selected = selected[np.argsort(work[selected], kind="stable")]
    selected = selected.astype(np.uint32, copy=False)
    selected.sort()
    return selected


def select_profiled_indices(
    index: SelectionIndex,
    eligible_mask: np.ndarray,
    count: int,
    seed: int,
    tag: str,
    profile: ProfileSpec,
) -> np.ndarray:
    if profile.strategy == "bucket_flat":
        return select_bucket_balanced_indices(index, eligible_mask, count, seed, tag, profile)
    if profile.strategy == "opening_cap":
        return select_opening_capped_indices(index, eligible_mask, count, seed, tag, profile)
    weights = profile_weight_array(index.abs_scores, profile)
    priorities = priority_array(index.key_hashes, seed, tag, weights)
    if profile.terminal_cap_fraction is None:
        return select_smallest(priorities, count, eligible_mask)
    terminal_limit = int(math.floor(count * profile.terminal_cap_fraction))
    nonterminal_target = count - terminal_limit
    nonterminal_mask = eligible_mask & ~index.terminal_mask
    terminal_mask = eligible_mask & index.terminal_mask
    nonterminal_available = int(nonterminal_mask.sum())
    terminal_available = int(terminal_mask.sum())
    keep_nonterminal = min(nonterminal_target, nonterminal_available)
    keep_terminal = min(terminal_limit, terminal_available)
    selected_nonterminal = select_smallest(
        priorities,
        keep_nonterminal,
        nonterminal_mask,
    )
    selected_terminal = select_smallest(priorities, keep_terminal, terminal_mask)
    selected_parts = [selected_nonterminal, selected_terminal]
    selected_count = int(selected_nonterminal.shape[0] + selected_terminal.shape[0])
    shortfall = count - selected_count
    if shortfall > 0:
        remaining_mask = eligible_mask.copy()
        remaining_mask[selected_nonterminal] = False
        remaining_mask[selected_terminal] = False
        selected_parts.append(select_smallest(priorities, shortfall, remaining_mask))
    merged = np.concatenate(selected_parts).astype(np.uint32, copy=False)
    merged.sort()
    return merged


def profile_diag_dir(
    root: Path,
    dataset_slug: str,
    profile: ProfileSpec,
    seed: int,
    max_abs_score: int | None,
) -> Path:
    root = artifact_root(root)
    return root / "data/training" / (
        f"{dataset_slug}_diag_{profile.name}_{sample_count_label(DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES)}"
        f"_seed{seed}_{score_filter_tag(max_abs_score)}_{SELECTION_CORPUS_VERSION}"
    )


def profile_tier_corpus_dir(
    root: Path,
    dataset_slug: str,
    profile: ProfileSpec,
    tier: TierSpec,
    seed: int,
    max_abs_score: int | None,
) -> Path:
    root = artifact_root(root)
    return root / "data/training" / (
        f"{dataset_slug}_{profile.name}_{tier.name}"
        f"_train{sample_count_label(tier.train_samples)}"
        f"_val{sample_count_label(tier.fixed_validation_samples)}"
        f"_diag{sample_count_label(tier.diagnostic_validation_samples)}"
        f"_seed{seed}_{score_filter_tag(max_abs_score)}_{SELECTION_CORPUS_VERSION}"
    )


def max_abs_score_mask(index: SelectionIndex, max_abs_score: int | None) -> np.ndarray:
    if max_abs_score is None:
        return np.ones(index.sample_count, dtype=bool)
    return index.abs_scores <= float(max_abs_score)


def score_bucket_label(bucket: int) -> str:
    lower = bucket * SCORE_BUCKET_SIZE
    upper = lower + SCORE_BUCKET_SIZE - 1
    return f"{lower}-{upper}"


def ply_bucket_label(bucket: int) -> str:
    lower = bucket * PLY_BUCKET_SIZE
    upper = lower + PLY_BUCKET_SIZE - 1
    return f"{lower}-{upper}"


def histogram_dict(values: np.ndarray, formatter: Callable[[int], str]) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        formatter(int(bucket)): int(count)
        for bucket, count in zip(unique.tolist(), counts.tolist())
    }


def result_histogram_dict(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(bucket)): int(count) for bucket, count in zip(unique.tolist(), counts.tolist())}


def opening_stats_for_selection(index: SelectionIndex, selection: np.ndarray) -> dict[str, int | float]:
    if selection.shape[0] == 0:
        return {
            "unique_openings": 0,
            "median_samples_per_opening": 0.0,
            "p95_samples_per_opening": 0.0,
            "max_samples_per_opening": 0,
        }
    _, counts = np.unique(index.opening_hashes[selection], return_counts=True)
    counts_f64 = counts.astype(np.float64, copy=False)
    return {
        "unique_openings": int(counts.shape[0]),
        "median_samples_per_opening": float(np.median(counts_f64)),
        "p95_samples_per_opening": float(np.percentile(counts_f64, 95)),
        "max_samples_per_opening": int(counts.max()),
    }


def selection_diagnostics(
    index: SelectionIndex,
    selection: np.ndarray,
    profile: ProfileSpec,
    tier: TierSpec | None,
    seed: int,
    tag: str,
) -> dict[str, object]:
    terminal_percentage = (
        float(index.terminal_mask[selection].mean() * 100.0) if selection.shape[0] else 0.0
    )
    diagnostics: dict[str, object] = {
        "profile": asdict(profile),
        "tier": asdict(tier) if tier is not None else None,
        "seed": seed,
        "tag": tag,
        "sample_count": int(selection.shape[0]),
        "terminal_percentage": terminal_percentage,
        "abs_score_histogram": histogram_dict(index.score_buckets[selection], score_bucket_label),
        "ply_histogram": histogram_dict(index.ply_buckets[selection], ply_bucket_label),
        "result_histogram": result_histogram_dict(index.result_buckets[selection]),
        "disjoint_train_diagnostic": None,
    }
    diagnostics.update(opening_stats_for_selection(index, selection))
    return diagnostics


def base_val_summary(base_corpus: BaseCorpus) -> dict[str, object]:
    return dict(base_corpus.manifest["val"])


def empty_test_summary() -> dict[str, int | str]:
    return {
        "file": "test.abapack",
        "chains": 0,
        "samples": 0,
        "raw_occurrences": 0,
        "unique_openings": 0,
    }


def summary_from_manifest_entry(filename: str, payload: dict[str, object]) -> SummaryAccumulator:
    summary = SummaryAccumulator(filename)
    summary.chains = int(payload["chains"])
    summary.samples = int(payload["samples"])
    summary.raw_occurrences = int(payload["raw_occurrences"])
    class_counts = payload.get("class_counts")
    if isinstance(class_counts, dict):
        summary.class_counts = {str(key): int(value) for key, value in class_counts.items()}
    summary.opening_hashes = set()
    return summary


def build_selection_stats(
    selection_name: str,
    base_corpus: BaseCorpus,
    train_summary: SummaryAccumulator,
    val_summary: SummaryAccumulator | None = None,
) -> dict[str, dict[str, int]]:
    base_train = base_corpus.manifest["train"]
    base_val = base_corpus.manifest["val"]
    base_total_samples = int(base_train["samples"]) + int(base_val["samples"])
    base_total_raw = int(base_train["raw_occurrences"]) + int(base_val["raw_occurrences"])
    kept_val_samples = val_summary.samples if val_summary is not None else int(base_val["samples"])
    kept_val_raw = val_summary.raw_occurrences if val_summary is not None else int(base_val["raw_occurrences"])
    kept_samples = train_summary.samples + kept_val_samples
    kept_raw = train_summary.raw_occurrences + kept_val_raw
    return {
        selection_name: {
            "kept_samples": kept_samples,
            "dropped_samples": base_total_samples - kept_samples,
            "kept_raw_occurrences": kept_raw,
            "dropped_raw_occurrences": base_total_raw - kept_raw,
        }
    }


def build_manifest(
    template_manifest: dict[str, object],
    output_dir: Path,
    dataset_path: Path,
    profile: ProfileSpec,
    tier: TierSpec,
    seed: int,
    max_abs_score: int | None,
    train_summary: SummaryAccumulator,
    val_summary: SummaryAccumulator,
    diagnostic_summary: SummaryAccumulator,
) -> dict[str, object]:
    manifest = copy.deepcopy(template_manifest)
    manifest["source_file"] = (
        f"trainer_profile_corpus:{dataset_path}:profile={profile.name}:tier={tier.name}:seed={seed}:"
        f"max_abs_score={max_abs_score if max_abs_score is not None else 'none'}"
    )
    manifest["output_dir"] = str(output_dir)
    manifest["split_rule"] = f"trainer_profile_corpus_{SELECTION_CORPUS_VERSION}"
    manifest["selection_profile"] = profile.name
    manifest["max_abs_score"] = max_abs_score
    manifest["selection_stats"] = build_selection_stats(profile.name, BaseCorpus(
        corpus_dir=Path(str(template_manifest["output_dir"])),
        train_path=Path("train.abapack"),
        val_path=Path("val.abapack"),
        manifest_path=Path("manifest.json"),
        manifest=template_manifest,
        train_samples=int(template_manifest["train"]["samples"]),
        val_samples=int(template_manifest["val"]["samples"]),
    ), train_summary, val_summary)
    manifest["train_class_counts"] = train_summary.class_counts
    manifest["train"] = train_summary.to_manifest_dict()
    manifest["val"] = val_summary.to_manifest_dict()
    manifest["test"] = empty_test_summary()
    manifest["teacher_selection_basis"] = None
    manifest["profile_diagnostic_summary"] = diagnostic_summary.to_manifest_dict()
    return manifest


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_selected_corpus(
    source_path: Path,
    destination_path: Path,
    selection: np.ndarray,
    split: str,
) -> SummaryAccumulator:
    writer = PreparedWriter(destination_path)
    summary = SummaryAccumulator(destination_path.name)
    pointer = 0
    ordinal = 0
    target_count = int(selection.shape[0])
    for chain in iter_prepared_chains(source_path):
        kept_samples: list[PreparedSample] = []
        for sample in chain.samples:
            if pointer < target_count and ordinal == int(selection[pointer]):
                kept_samples.append(sample)
                pointer += 1
            ordinal += 1
        if kept_samples:
            kept_chain = PreparedChain(
                run_file=chain.run_file,
                game_index=chain.game_index,
                opening_name=chain.opening_name,
                opening_position=chain.opening_position,
                opening_hash=chain.opening_hash,
                split=split,
                samples=kept_samples,
            )
            writer.write_chain(kept_chain)
            summary.observe_chain(kept_chain)
    writer.close()
    if pointer != target_count:
        fail(f"selection replay for {destination_path} wrote {pointer} samples, expected {target_count}")
    return summary


def selection_hashes_for_prepared(path: Path) -> np.ndarray:
    hashes: list[int] = []
    for chain in iter_prepared_chains(path):
        for sample_index, sample in enumerate(chain.samples):
            hashes.append(sample_identity_hash(chain, sample, sample_index))
    return np.array(hashes, dtype=np.uint64)


def select_random_subset_from_prepared(
    source_path: Path,
    count: int,
    seed: int,
    tag: str,
) -> np.ndarray:
    key_hashes = selection_hashes_for_prepared(source_path)
    if count > int(key_hashes.shape[0]):
        fail(f"requested {count} samples from {source_path}, but only {key_hashes.shape[0]} are available")
    priorities = priority_array(
        key_hashes,
        seed,
        tag,
        np.ones(key_hashes.shape[0], dtype=np.float64),
    )
    return select_smallest(priorities, count)


def ensure_diagnostic_corpus(
    root: Path,
    dataset_slug: str,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    profile: ProfileSpec,
    seed: int,
    max_abs_score: int | None,
) -> tuple[Path, SummaryAccumulator]:
    output_dir = profile_diag_dir(root, dataset_slug, profile, seed, max_abs_score)
    diag_path = output_dir / "diagnostic_val.abapack"
    summary_path = output_dir / "summary.json"
    profile_path = output_dir / "profile.json"
    if diag_path.is_file() and summary_path.is_file() and profile_path.is_file():
        payload = load_json_file(summary_path)
        summary = SummaryAccumulator(diag_path.name)
        summary.chains = int(payload["chains"])
        summary.samples = int(payload["samples"])
        summary.raw_occurrences = int(payload["raw_occurrences"])
        summary.opening_hashes = set()
        summary.class_counts = {key: int(value) for key, value in payload["class_counts"].items()}
        return diag_path, summary
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_mask = max_abs_score_mask(index, max_abs_score)
    selection = select_profiled_indices(
        index=index,
        eligible_mask=eligible_mask,
        count=DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES,
        seed=seed,
        tag=f"diag:{profile.name}",
        profile=profile,
    )
    summary = write_selected_corpus(base_corpus.train_path, diag_path, selection, "val")
    diagnostics = selection_diagnostics(
        index=index,
        selection=selection,
        profile=profile,
        tier=None,
        seed=seed,
        tag=f"diag:{profile.name}",
    )
    write_json(
        summary_path,
        {
            "chains": summary.chains,
            "samples": summary.samples,
            "raw_occurrences": summary.raw_occurrences,
            "class_counts": summary.class_counts,
        },
    )
    write_json(
        profile_path,
        {
            "dataset": str(base_corpus.train_path),
            "profile": asdict(profile),
            "seed": seed,
            "max_abs_score": max_abs_score,
            "diagnostic_val_path": str(diag_path),
            "terminal_threshold": TERMINAL_SCORE_THRESHOLD,
            "summary": summary.to_manifest_dict(),
            "selection_diagnostics": diagnostics,
        },
    )
    return diag_path, summary


def ensure_training_corpus(
    root: Path,
    dataset: DatasetInput,
    dataset_slug: str,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    profile: ProfileSpec,
    tier: TierSpec,
    seed: int,
    max_abs_score: int | None,
) -> CorpusPaths:
    output_dir = profile_tier_corpus_dir(root, dataset_slug, profile, tier, seed, max_abs_score)
    train_path = output_dir / "train.abapack"
    val_path = output_dir / "val.abapack"
    diagnostic_val_path = output_dir / "diagnostic_val.abapack"
    test_path = output_dir / "test.abapack"
    manifest_path = output_dir / "manifest.json"
    profile_path = output_dir / "profile.json"
    dataset_cache_dir = output_dir / "dataset_cache"
    if tier.diagnostic_validation_samples > 0:
        diag_source_path, diagnostic_summary = ensure_diagnostic_corpus(
            root=root,
            dataset_slug=dataset_slug,
            base_corpus=base_corpus,
            index=index,
            profile=profile,
            seed=seed,
            max_abs_score=max_abs_score,
        )
    else:
        diag_source_path = base_corpus.val_path
        diagnostic_summary = summary_from_manifest_entry(diag_source_path.name, dict(base_corpus.manifest["val"]))
    if (
        train_path.is_file()
        and val_path.is_file()
        and diagnostic_val_path.is_file()
        and test_path.is_file()
        and manifest_path.is_file()
        and profile_path.is_file()
    ):
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        return CorpusPaths(
            corpus_dir=output_dir,
            train_path=train_path,
            val_path=val_path,
            diagnostic_val_path=diagnostic_val_path,
            manifest_path=manifest_path,
            profile_path=profile_path,
            dataset_cache_dir=dataset_cache_dir,
            train_samples=tier.train_samples,
            val_samples=tier.fixed_validation_samples,
            diagnostic_val_samples=(
                tier.diagnostic_validation_samples
                if tier.diagnostic_validation_samples > 0
                else base_corpus.val_samples
            ),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_mask = max_abs_score_mask(index, max_abs_score)
    if tier.diagnostic_validation_samples > 0:
        diag_selection = select_profiled_indices(
            index=index,
            eligible_mask=eligible_mask,
            count=tier.diagnostic_validation_samples,
            seed=seed,
            tag=f"diag:{profile.name}",
            profile=profile,
        )
    else:
        diag_selection = np.empty(0, dtype=np.uint32)
    eligible_mask[diag_selection] = False
    train_selection = select_profiled_indices(
        index=index,
        eligible_mask=eligible_mask,
        count=tier.train_samples,
        seed=seed,
        tag=f"train:{profile.name}:{tier.name}",
        profile=profile,
    )
    train_summary = write_selected_corpus(base_corpus.train_path, train_path, train_selection, "train")
    train_diagnostics = selection_diagnostics(
        index=index,
        selection=train_selection,
        profile=profile,
        tier=tier,
        seed=seed,
        tag=f"train:{profile.name}:{tier.name}",
    )
    train_diagnostics["disjoint_train_diagnostic"] = bool(
        len(set(train_selection.tolist()).intersection(set(diag_selection.tolist()))) == 0
    )
    if tier.diagnostic_validation_samples > 0:
        diagnostic_diagnostics = selection_diagnostics(
            index=index,
            selection=diag_selection,
            profile=profile,
            tier=tier,
            seed=seed,
            tag=f"diag:{profile.name}",
        )
        diagnostic_diagnostics["disjoint_train_diagnostic"] = bool(train_diagnostics["disjoint_train_diagnostic"])
    else:
        diagnostic_diagnostics = {
            "profile": asdict(profile),
            "tier": asdict(tier),
            "seed": seed,
            "tag": f"diag:{profile.name}",
            "sample_count": base_corpus.val_samples,
            "source": "shared_val",
            "disjoint_train_diagnostic": True,
        }
    if tier.fixed_validation_samples == base_corpus.val_samples:
        hard_link_or_copy(base_corpus.val_path, val_path)
        val_summary = summary_from_manifest_entry(val_path.name, dict(base_corpus.manifest["val"]))
    else:
        val_selection = select_random_subset_from_prepared(
            base_corpus.val_path,
            tier.fixed_validation_samples,
            seed,
            f"val:{profile.name}:{tier.name}",
        )
        val_summary = write_selected_corpus(base_corpus.val_path, val_path, val_selection, "val")
    hard_link_or_copy(diag_source_path, diagnostic_val_path)
    create_empty_prepared_abapack(test_path)
    manifest = build_manifest(
        template_manifest=base_corpus.manifest,
        output_dir=output_dir,
        dataset_path=dataset.prepared_path,
        profile=profile,
        tier=tier,
        seed=seed,
        max_abs_score=max_abs_score,
        train_summary=train_summary,
        val_summary=val_summary,
        diagnostic_summary=diagnostic_summary,
    )
    write_json(manifest_path, manifest)
    write_json(
        profile_path,
        {
            "dataset": str(dataset.prepared_path),
            "profile": asdict(profile),
            "tier": asdict(tier),
            "seed": seed,
            "max_abs_score": max_abs_score,
            "terminal_threshold": TERMINAL_SCORE_THRESHOLD,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "diagnostic_val_path": str(diagnostic_val_path),
            "manifest_path": str(manifest_path),
            "train_summary": train_summary.to_manifest_dict(),
            "val_summary": val_summary.to_manifest_dict(),
            "diagnostic_summary": diagnostic_summary.to_manifest_dict(),
            "train_selection_diagnostics": train_diagnostics,
            "diagnostic_selection_diagnostics": diagnostic_diagnostics,
        },
    )
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    return CorpusPaths(
        corpus_dir=output_dir,
        train_path=train_path,
        val_path=val_path,
        diagnostic_val_path=diagnostic_val_path,
        manifest_path=manifest_path,
        profile_path=profile_path,
        dataset_cache_dir=dataset_cache_dir,
        train_samples=tier.train_samples,
        val_samples=tier.fixed_validation_samples,
        diagnostic_val_samples=(
            tier.diagnostic_validation_samples
            if tier.diagnostic_validation_samples > 0
            else base_corpus.val_samples
        ),
    )


def resolve_active_learning_chunk_sizes(
    active_learning: ActiveLearningSpec,
    eligible_pool_samples: int,
) -> tuple[int, ...]:
    if eligible_pool_samples <= 0:
        fail("active-learning pool has no eligible samples")
    if active_learning.manual_chunk_sizes is not None:
        total = sum(active_learning.manual_chunk_sizes)
        if total > eligible_pool_samples:
            fail(
                f"active-learning schedule selects {total} samples, "
                f"but only {eligible_pool_samples} are eligible in the pool"
            )
        return active_learning.manual_chunk_sizes
    if active_learning.auto_chunk_size is not None:
        chunk_size = active_learning.auto_chunk_size
        chunks: list[int] = []
        remaining = eligible_pool_samples
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunks.append(take)
            remaining -= take
        return tuple(chunks)
    fail("active-learning spec is missing both chunk schedule and chunk size")


def active_learning_round_specs(chunk_sizes: Sequence[int]) -> list[ActiveLearningRoundSpec]:
    specs: list[ActiveLearningRoundSpec] = []
    selected = 0
    for index, chunk_size in enumerate(chunk_sizes, start=1):
        selected += chunk_size
        specs.append(
            ActiveLearningRoundSpec(
                index=index,
                chunk_size=chunk_size,
                selected_samples=selected,
            )
        )
    return specs


def active_learning_root_dir(experiment_root: Path) -> Path:
    return experiment_root / "active_learning"


def active_learning_pool_metadata_path(experiment_root: Path) -> Path:
    return active_learning_root_dir(experiment_root) / f"pool_metadata_{ACTIVE_LEARNING_VERSION}.npz"


def active_learning_round_root(experiment_root: Path, round_spec: ActiveLearningRoundSpec) -> Path:
    return active_learning_root_dir(experiment_root) / round_spec.round_slug


def ensure_active_learning_pool_abs_scores(
    experiment_root: Path,
    pool_path: Path,
    expected_samples: int,
) -> np.ndarray:
    path = active_learning_pool_metadata_path(experiment_root)
    if path.is_file():
        with np.load(path) as payload:
            abs_scores = payload["abs_scores"].astype(np.float32, copy=False)
        if int(abs_scores.shape[0]) != expected_samples:
            fail(
                f"active-learning pool metadata mismatch for {pool_path}: "
                f"expected {expected_samples}, saw {abs_scores.shape[0]}"
            )
        return abs_scores
    path.parent.mkdir(parents=True, exist_ok=True)
    abs_scores = np.empty(expected_samples, dtype=np.float32)
    ordinal = 0
    for chain in iter_prepared_chains(pool_path):
        for sample in chain.samples:
            if ordinal >= expected_samples:
                fail(f"active-learning pool {pool_path} exceeded expected sample count {expected_samples}")
            abs_scores[ordinal] = np.float32(abs(sample.mean_clipped_score))
            ordinal += 1
    if ordinal != expected_samples:
        fail(
            f"active-learning pool sample count mismatch for {pool_path}: "
            f"expected {expected_samples}, saw {ordinal}"
        )
    np.savez(path, abs_scores=abs_scores)
    return abs_scores


def active_learning_available_indices(
    eligible_mask: np.ndarray,
    selected_mask: np.ndarray,
) -> np.ndarray:
    if eligible_mask.shape != selected_mask.shape:
        fail("active-learning mask shape mismatch")
    return np.flatnonzero(eligible_mask & ~selected_mask).astype(np.uint32, copy=False)


def select_active_learning_random_ids(
    eligible_mask: np.ndarray,
    selected_mask: np.ndarray,
    count: int,
    seed: int,
    tag: str,
) -> np.ndarray:
    available_indices = active_learning_available_indices(eligible_mask, selected_mask)
    if count > int(available_indices.shape[0]):
        fail(
            f"requested {count} active-learning samples but only "
            f"{available_indices.shape[0]} eligible unselected samples are available"
        )
    priorities = priority_array(
        available_indices.astype(np.uint64, copy=False),
        seed,
        tag,
        np.ones(available_indices.shape[0], dtype=np.float64),
    )
    chosen_local = np.argpartition(priorities, count - 1)[:count]
    chosen = available_indices[chosen_local].astype(np.uint32, copy=False)
    chosen.sort()
    return chosen


def run_text_command(root: Path, command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        cwd=root,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        parts: list[str] = []
        if result.stdout.strip():
            parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.strip()}")
        suffix = f"\n{chr(10).join(parts)}" if parts else ""
        fail(f"command failed with status {result.returncode}:{suffix}")
    return result.stdout


def ensure_teacher_targets(
    root: Path,
    nnue_bin: Path,
    model_path: Path,
    pool_path: Path,
    teacher_path: Path,
) -> Path:
    if teacher_path.is_file():
        return teacher_path
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    status(f"trainer: caching teacher targets for active-learning pool {pool_path}")
    run_text_command(
        root,
        [
            str(nnue_bin),
            "cache-teacher-targets",
            "--model",
            str(model_path),
            "--input",
            str(pool_path),
            "--output",
            str(teacher_path),
        ],
    )
    return teacher_path


def select_top_disagreement_ids_from_teacher_targets(
    pool_path: Path,
    teacher_path: Path,
    eligible_mask: np.ndarray,
    selected_mask: np.ndarray,
    count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    available = int((eligible_mask & ~selected_mask).sum())
    if count > available:
        fail(
            f"requested {count} active-learning disagreement samples but only "
            f"{available} eligible unselected samples are available"
        )
    import heapq

    heap: list[tuple[float, int]] = []
    considered = 0
    min_disagreement: float | None = None
    max_disagreement: float | None = None
    sum_disagreement = 0.0
    global_index = 0
    with TeacherTargetReader(teacher_path) as reader:
        for chain in iter_prepared_chains(pool_path):
            for sample in chain.samples:
                model_score = reader.next_score_for(sample.position_key)
                if global_index >= selected_mask.shape[0]:
                    fail(
                        f"active-learning disagreement scoring exceeded mask length for {pool_path}: "
                        f"{global_index} >= {selected_mask.shape[0]}"
                    )
                if not eligible_mask[global_index] or selected_mask[global_index]:
                    global_index += 1
                    continue
                disagreement = abs(float(model_score) - float(sample.mean_score))
                considered += 1
                sum_disagreement += disagreement
                min_disagreement = disagreement if min_disagreement is None else min(min_disagreement, disagreement)
                max_disagreement = disagreement if max_disagreement is None else max(max_disagreement, disagreement)
                entry = (disagreement, -global_index)
                if len(heap) < count:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
                global_index += 1
    if global_index != selected_mask.shape[0]:
        fail(
            f"scored {global_index} pool samples, expected {selected_mask.shape[0]} from active-learning mask"
        )
    added_ids = np.array(sorted(-entry[1] for entry in heap), dtype=np.uint32)
    disagreements = sorted(entry[0] for entry in heap)
    meta = {
        "pool_path": str(pool_path),
        "teacher_targets": str(teacher_path),
        "considered_samples": considered,
        "selected_samples": int(added_ids.shape[0]),
        "mean_disagreement_remaining": (sum_disagreement / considered) if considered else 0.0,
        "min_disagreement_remaining": min_disagreement,
        "max_disagreement_remaining": max_disagreement,
        "selected_min_disagreement": disagreements[0] if disagreements else None,
        "selected_max_disagreement": disagreements[-1] if disagreements else None,
    }
    return added_ids, meta


def select_active_learning_disagreement_ids(
    root: Path,
    nnue_bin: Path,
    pool_path: Path,
    model_path: Path,
    teacher_root: Path,
    eligible_mask: np.ndarray,
    selected_mask: np.ndarray,
    count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    teacher_path = ensure_teacher_targets(
        root=root,
        nnue_bin=nnue_bin,
        model_path=model_path,
        pool_path=pool_path,
        teacher_path=teacher_root / "train.teacher.bin",
    )
    return select_top_disagreement_ids_from_teacher_targets(
        pool_path=pool_path,
        teacher_path=teacher_path,
        eligible_mask=eligible_mask,
        selected_mask=selected_mask,
        count=count,
    )


def active_learning_diagnostic_summary(
    manifest: dict[str, object],
    diagnostic_path: Path,
) -> SummaryAccumulator:
    payload = manifest.get("profile_diagnostic_summary")
    if isinstance(payload, dict):
        return summary_from_manifest_entry(diagnostic_path.name, payload)
    return summary_from_manifest_entry(diagnostic_path.name, dict(manifest["val"]))


def build_active_learning_manifest(
    template_manifest: dict[str, object],
    output_dir: Path,
    source_train_path: Path,
    selection_name: str,
    train_summary: SummaryAccumulator,
    diagnostic_summary: SummaryAccumulator,
) -> dict[str, object]:
    manifest = copy.deepcopy(template_manifest)
    manifest["source_file"] = (
        f"trainer_active_learning_subset:{source_train_path}:selection={selection_name}"
    )
    manifest["output_dir"] = str(output_dir)
    manifest["split_rule"] = f"trainer_active_learning_subset_{ACTIVE_LEARNING_VERSION}"
    manifest["selection_profile"] = selection_name
    manifest["selection_stats"] = build_selection_stats(
        selection_name,
        BaseCorpus(
            corpus_dir=Path(str(template_manifest["output_dir"])),
            train_path=Path("train.abapack"),
            val_path=Path("val.abapack"),
            manifest_path=Path("manifest.json"),
            manifest=template_manifest,
            train_samples=int(template_manifest["train"]["samples"]),
            val_samples=int(template_manifest["val"]["samples"]),
        ),
        train_summary,
    )
    manifest["train_class_counts"] = train_summary.class_counts
    manifest["train"] = train_summary.to_manifest_dict()
    manifest["val"] = dict(template_manifest["val"])
    manifest["test"] = empty_test_summary()
    manifest["teacher_selection_basis"] = None
    manifest["profile_diagnostic_summary"] = diagnostic_summary.to_manifest_dict()
    return manifest


def ensure_active_learning_round_corpus(
    round_root: Path,
    pool_corpus: CorpusPaths,
    round_spec: ActiveLearningRoundSpec,
    selected_indices: np.ndarray,
    profile: ProfileSpec,
    tier: TierSpec,
    seed: int,
    pool_cutoff_cp: int | None,
) -> CorpusPaths:
    corpus_dir = round_root / "corpus"
    train_path = corpus_dir / "train.abapack"
    val_path = corpus_dir / "val.abapack"
    diagnostic_val_path = corpus_dir / "diagnostic_val.abapack"
    test_path = corpus_dir / "test.abapack"
    manifest_path = corpus_dir / "manifest.json"
    profile_path = corpus_dir / "profile.json"
    dataset_cache_dir = corpus_dir / "dataset_cache"
    if (
        train_path.is_file()
        and val_path.is_file()
        and diagnostic_val_path.is_file()
        and test_path.is_file()
        and manifest_path.is_file()
        and profile_path.is_file()
    ):
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        return CorpusPaths(
            corpus_dir=corpus_dir,
            train_path=train_path,
            val_path=val_path,
            diagnostic_val_path=diagnostic_val_path,
            manifest_path=manifest_path,
            profile_path=profile_path,
            dataset_cache_dir=dataset_cache_dir,
            train_samples=int(selected_indices.shape[0]),
            val_samples=pool_corpus.val_samples,
            diagnostic_val_samples=pool_corpus.diagnostic_val_samples,
        )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    template_manifest = load_json_file(pool_corpus.manifest_path)
    train_summary = write_selected_corpus(pool_corpus.train_path, train_path, selected_indices, "train")
    hard_link_or_copy(pool_corpus.val_path, val_path)
    hard_link_or_copy(pool_corpus.diagnostic_val_path, diagnostic_val_path)
    create_empty_prepared_abapack(test_path)
    diagnostic_summary = active_learning_diagnostic_summary(template_manifest, diagnostic_val_path)
    manifest = build_active_learning_manifest(
        template_manifest=template_manifest,
        output_dir=corpus_dir,
        source_train_path=pool_corpus.train_path,
        selection_name=f"active_learning_round_{ACTIVE_LEARNING_VERSION}",
        train_summary=train_summary,
        diagnostic_summary=diagnostic_summary,
    )
    write_json(manifest_path, manifest)
    write_json(
        profile_path,
        {
            "selection_profile": "active_learning_round",
            "selection_version": ACTIVE_LEARNING_VERSION,
            "profile": asdict(profile),
            "tier": asdict(tier),
            "seed": seed,
            "round": round_spec.index,
            "chunk_size": round_spec.chunk_size,
            "selected_samples": round_spec.selected_samples,
            "pool_train_path": str(pool_corpus.train_path),
            "pool_cutoff_cp": pool_cutoff_cp,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "diagnostic_val_path": str(diagnostic_val_path),
            "manifest_path": str(manifest_path),
            "train_summary": train_summary.to_manifest_dict(),
            "diagnostic_summary": diagnostic_summary.to_manifest_dict(),
        },
    )
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    return CorpusPaths(
        corpus_dir=corpus_dir,
        train_path=train_path,
        val_path=val_path,
        diagnostic_val_path=diagnostic_val_path,
        manifest_path=manifest_path,
        profile_path=profile_path,
        dataset_cache_dir=dataset_cache_dir,
        train_samples=int(selected_indices.shape[0]),
        val_samples=pool_corpus.val_samples,
        diagnostic_val_samples=pool_corpus.diagnostic_val_samples,
    )


def checkpoint_epochs_for_candidate(spec: CandidateSpec) -> list[int]:
    interval = spec.tier.checkpoint_interval_epochs
    if interval is None or interval <= 0:
        return []
    start = spec.checkpoint_start_epoch if spec.checkpoint_start_epoch is not None else interval
    if start <= 0:
        fail(f"checkpoint start must be positive, got {start}")
    return list(range(start, spec.epochs + 1, interval))


def print_startup(
    spec: CandidateSpec,
    corpus: CorpusPaths,
    experiment_root: Path,
    book_path: Path,
    checkpoint_epochs: Sequence[int],
) -> None:
    checkpoint_label = ",".join(str(epoch) for epoch in checkpoint_epochs) if checkpoint_epochs else "none"
    checkpoint_interval_display = spec.tier.checkpoint_interval_epochs or 0
    checkpoint_match_games_display = spec.tier.checkpoint_match_games or 0
    checkpoint_top_k_display = DEFAULT_TOP_CHECKPOINT_COUNT if spec.active_learning is None else 0
    if checkpoint_interval_display == 1 and checkpoint_epochs:
        checkpoint_label = "every epoch"
    if spec.active_learning is not None and spec.active_learning.round_match_top_k > 1:
        checkpoint_interval_display = 1
        checkpoint_label = "every epoch (active-learning top-k override)"
        checkpoint_match_games_display = spec.tier.final_match_games
        checkpoint_top_k_display = spec.active_learning.round_match_top_k
    lines = [
        "trainer: startup hyperparameters",
        f"  entrypoint: {Path(__file__).resolve()}",
        f"  dataset: {spec.dataset_path}",
        f"  mode: {spec.mode}",
        f"  sweep: {spec.sweep_name or 'none'}",
        f"  candidate_id: {spec.candidate_id}",
        f"  profile: {spec.profile.name}",
        f"  tier: {spec.tier.name}",
        f"  recipe: {spec.recipe.name}",
        f"  corpus_dir: {corpus.corpus_dir}",
        f"  train_dataset: {corpus.train_path}",
        f"  val_dataset: {corpus.val_path}",
        f"  diagnostic_val_dataset: {corpus.diagnostic_val_path}",
        f"  manifest: {corpus.manifest_path}",
        f"  profile_file: {corpus.profile_path}",
        f"  dataset_cache_dir: {corpus.dataset_cache_dir}",
        f"  engine_template: {spec.engine_template_path}",
        f"  reference_source: {spec.reference_source_path if spec.reference_source_path is not None else spec.engine_template_path}",
        f"  experiment_dir: {experiment_root}",
        f"  epochs: {spec.epochs}",
        "  epoch_size: full",
        (
            "  corpus_sizes: "
            f"train={corpus.train_samples} "
            f"validation={corpus.val_samples} "
            f"diagnostic_validation={corpus.diagnostic_val_samples}"
        ),
        (
            "  active_learning: "
            + (
                (
                    f"{active_learning_plan_text(spec.active_learning, corpus.train_samples)} "
                    f"pool_cutoff_cp={spec.active_learning.pool_cutoff_cp if spec.active_learning.pool_cutoff_cp is not None else 'none'} "
                    f"round_match_top_k={spec.active_learning.round_match_top_k}"
                )
                if spec.active_learning is not None
                else "disabled"
            )
        ),
        f"  batch_size: {spec.recipe.batch_size}",
        f"  learning_rate: {spec.recipe.learning_rate}",
        f"  weight_decay: {spec.recipe.weight_decay}",
        f"  dropout: {spec.recipe.dropout}",
        (
            "  patience: "
            + (
                f"{patience_for_spec(spec)} per active-learning round"
                if spec.active_learning is not None
                else str(patience_for_spec(spec))
            )
        ),
        f"  backend: {spec.backend}",
        (
            "  lambda_mix: "
            f"score={spec.lambda_score:.3f} "
            f"result={spec.lambda_result:.3f}"
        ),
        f"  feature_set: {spec.feature_set_name}",
        f"  architecture: {spec.architecture_spec}",
        f"  dense_feature_mask: {dense_feature_mask_label(spec.dense_feature_mask)}",
        f"  repeat_occurrence_weight: {spec.repeat_occurrence_weight if spec.repeat_occurrence_weight is not None else 'none'}",
        f"  class_weighting: {spec.class_weighting}",
        f"  ema_decay: {spec.ema_decay if spec.ema_decay is not None else 'none'}",
        f"  target_transform: {TARGET_TRANSFORM}",
        f"  loss: {LOSS}",
        f"  huber_delta: {HUBER_DELTA}",
        f"  activation: {spec.activation_name}",
        f"  norm: {NORM}",
        f"  block_type: {BLOCK_TYPE}",
        f"  threads: {DEFAULT_THREADS}",
        f"  loader_workers: {DEFAULT_LOADER_WORKERS}",
        f"  runtime_loss_interval: {runtime_loss_interval_for_spec(spec)}",
        f"  checkpoint_interval_epochs: {checkpoint_interval_display}",
        f"  checkpoint_start_epoch: {spec.checkpoint_start_epoch or 0}",
        f"  checkpoint_epochs: {checkpoint_label}",
        f"  checkpoint_top_k: {checkpoint_top_k_display}",
        f"  checkpoint_match_games: {checkpoint_match_games_display}",
        f"  checkpoint_book: {book_path}",
        f"  final_match_games: {spec.tier.final_match_games}",
        f"  confirm_match_games: {DEFAULT_CONFIRM_GAMES}",
        f"  match_time_ms: {spec.time_ms}",
        f"  python: {sys.executable}",
    ]
    for line in lines:
        status(line)


def build_train_command(
    nnue_bin: Path,
    corpus: CorpusPaths,
    experiment_root: Path,
    spec: CandidateSpec,
    *,
    enable_checkpoints: bool = True,
    patience: int | None = None,
    checkpoint_save_interval: int | None = None,
) -> list[str]:
    selected_patience = patience if patience is not None else patience_for_spec(spec)
    command = [
        str(nnue_bin),
        "train",
        "--train",
        str(corpus.train_path),
        "--val",
        str(corpus.val_path),
        "--manifest",
        str(corpus.manifest_path),
        "--output-dir",
        str(experiment_root),
        "--single-stage",
        "--epochs",
        str(spec.epochs),
        "--epoch-size",
        "full",
        "--batch-size",
        str(spec.recipe.batch_size),
        "--learning-rate",
        str(spec.recipe.learning_rate),
        "--weight-decay",
        str(spec.recipe.weight_decay),
        "--patience",
        str(selected_patience),
        "--dropout",
        str(spec.recipe.dropout),
        "--backend",
        spec.backend,
        "--feature-set",
        spec.feature_set_name,
        "--architecture",
        spec.architecture_spec,
        "--activation",
        spec.activation_name,
        "--threads",
        str(DEFAULT_THREADS),
        "--loader-workers",
        str(DEFAULT_LOADER_WORKERS),
        "--dataset-cache-dir",
        str(corpus.dataset_cache_dir),
        "--runtime-loss-interval",
        str(runtime_loss_interval_for_spec(spec)),
        "--lambda",
        str(spec.lambda_score),
        "--target-transform",
        TARGET_TRANSFORM,
        "--loss",
        LOSS,
        "--huber-delta",
        str(HUBER_DELTA),
        "--norm",
        NORM,
        "--block-type",
        BLOCK_TYPE,
        "--seed",
        str(spec.seed),
    ]
    if spec.dense_feature_mask:
        command.extend(["--dense-mask", ",".join(str(index) for index in spec.dense_feature_mask)])
    if spec.repeat_occurrence_weight not in {None, "none"}:
        command.extend(["--repeat-occurrence-weight", spec.repeat_occurrence_weight])
    if spec.class_weighting != DEFAULT_CLASS_WEIGHTING:
        command.extend(["--class-weighting", spec.class_weighting])
    if spec.ema_decay is not None:
        command.extend(["--ema-decay", str(spec.ema_decay)])
    effective_checkpoint_interval = checkpoint_save_interval
    if effective_checkpoint_interval is None:
        effective_checkpoint_interval = spec.tier.checkpoint_interval_epochs
    if enable_checkpoints and effective_checkpoint_interval is not None and effective_checkpoint_interval > 0:
        command.extend(
            [
                "--save-best-checkpoints",
                "--checkpoint-interval",
                str(effective_checkpoint_interval),
            ]
        )
    return command


def ensure_trained_model(
    root: Path,
    nnue_bin: Path,
    corpus: CorpusPaths,
    experiment_root: Path,
    spec: CandidateSpec,
    match_bin: Path,
    book_path: Path,
    candidate_id: str,
    checkpoint_epochs: Sequence[int],
    *,
    enable_checkpoints: bool,
    patience: int | None = None,
    checkpoint_save_interval: int | None = None,
) -> None:
    model_json = experiment_root / "model.json"
    model_nnq = experiment_root / "model.nnq"
    metrics_json = experiment_root / "metrics.json"
    loss_curve = experiment_root / "loss_curves.svg"
    if model_json.is_file() and model_nnq.is_file() and metrics_json.is_file() and loss_curve.is_file():
        status(f"trainer: reusing trained model {model_json}")
        return
    status(f"trainer: training experiment {experiment_root}")
    process = subprocess.Popen(
        build_train_command(
            nnue_bin,
            corpus,
            experiment_root,
            spec,
            enable_checkpoints=enable_checkpoints,
            patience=patience,
            checkpoint_save_interval=checkpoint_save_interval,
        ),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        fail("failed to capture training output")
    stream_thread = threading.Thread(
        target=stream_training_output,
        args=(process.stdout,),
        daemon=True,
    )
    stream_thread.start()
    return_code = process.wait()
    stream_thread.join(timeout=1.0)
    if return_code != 0:
        fail(f"training command failed with status {return_code}")


def checkpoint_model_path(experiment_root: Path, epoch: int) -> Path:
    return experiment_root / "epoch_checkpoints" / f"epoch_{epoch:04}_model.nnq"


def print_checkpoint_result(epoch: int, model_path: Path, result: MatchResult) -> None:
    status(
        "checkpoint "
        f"epoch={epoch} "
        f"model_nnq={model_path} "
        f"run={result.run_path} "
        f"wdl={result.wins}-{result.draws}-{result.losses} "
        f"score_pct={result.score_pct:.2f} "
        f"elo(current-reference)={format_metric(result.elo)} "
        f"[{format_metric(result.elo_lower)}, {format_metric(result.elo_upper)}] "
        f"avg_depth={result.avg_depth:.2f} "
        f"avg_nps={result.avg_nps:.0f} "
        f"avg_response_ms={result.avg_response_ms:.2f} "
        f"({result.decision})"
    )


def print_active_learning_round_result(round_index: int, model_path: Path, result: MatchResult) -> None:
    status(
        "active-learning-round "
        f"round={round_index} "
        f"model_nnq={model_path} "
        f"run={result.run_path} "
        f"wdl={result.wins}-{result.draws}-{result.losses} "
        f"score_pct={result.score_pct:.2f} "
        f"elo(current-reference)={format_metric(result.elo)} "
        f"[{format_metric(result.elo_lower)}, {format_metric(result.elo_upper)}] "
        f"avg_depth={result.avg_depth:.2f} "
        f"avg_nps={result.avg_nps:.0f} "
        f"avg_response_ms={result.avg_response_ms:.2f} "
        f"({result.decision})"
    )


def print_active_learning_round_checkpoint_result(
    round_index: int,
    rank: int,
    epoch: int,
    model_path: Path,
    quantized_val_loss: float,
    result: MatchResult,
) -> None:
    status(
        "active-learning-round-checkpoint "
        f"round={round_index} "
        f"rank={rank} "
        f"epoch={epoch} "
        f"quantized_val_loss={quantized_val_loss:.6f} "
        f"model_nnq={model_path} "
        f"run={result.run_path} "
        f"wdl={result.wins}-{result.draws}-{result.losses} "
        f"score_pct={result.score_pct:.2f} "
        f"elo(current-reference)={format_metric(result.elo)} "
        f"[{format_metric(result.elo_lower)}, {format_metric(result.elo_upper)}] "
        f"avg_depth={result.avg_depth:.2f} "
        f"avg_nps={result.avg_nps:.0f} "
        f"avg_response_ms={result.avg_response_ms:.2f} "
        f"({result.decision})"
    )


def run_checkpoint_matches(
    root: Path,
    training_process: subprocess.Popen[str],
    experiment_root: Path,
    checkpoint_epochs: Sequence[int],
    engine_template_path: Path,
    reference_source_path: Path,
    candidate_id: str,
    match_bin: Path,
    book_path: Path,
    time_ms: int,
    checkpoint_game_count: int,
    dense_feature_mask: Sequence[int],
) -> None:
    if not checkpoint_epochs:
        return
    if checkpoint_game_count <= 0:
        status("trainer: skipping checkpoint arena matches (checkpoint_game_count <= 0)")
        return
    checkpoint_source = checkpoint_candidate_source_path(
        experiment_root,
        engine_template_path.stem,
        candidate_id,
    )
    checkpoint_source.parent.mkdir(parents=True, exist_ok=True)
    dispatched = 0

    def run_checkpoint(epoch: int) -> None:
        model_path = checkpoint_model_path(experiment_root, epoch)
        bake_runtime_source(engine_template_path, model_path, checkpoint_source, dense_feature_mask)
        result = run_match(
            root,
            match_bin,
            checkpoint_source,
            reference_source_path,
            book_path,
            time_ms,
            checkpoint_game_count,
        )
        print_checkpoint_result(epoch, model_path, result)

    while dispatched < len(checkpoint_epochs):
        epoch = checkpoint_epochs[dispatched]
        model_path = checkpoint_model_path(experiment_root, epoch)
        if model_path.is_file():
            run_checkpoint(epoch)
            dispatched += 1
            continue
        if training_process.poll() is not None:
            break
        time.sleep(1.0)
    for epoch in checkpoint_epochs[dispatched:]:
        model_path = checkpoint_model_path(experiment_root, epoch)
        if model_path.is_file():
            run_checkpoint(epoch)


def top_quantized_checkpoint_rows(model_report: dict[str, object], limit: int) -> list[dict[str, float | int]]:
    if limit <= 0:
        return []
    history = model_report.get("history")
    if not isinstance(history, list):
        return []
    rows: list[dict[str, float | int]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        epoch_value = entry.get("epoch")
        qval_value = entry.get("quantized_val_loss")
        if epoch_value is None or qval_value is None:
            continue
        try:
            epoch = int(epoch_value)
            quantized_val_loss = float(qval_value)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "epoch": epoch,
                "quantized_val_loss": quantized_val_loss,
                "val_loss": float(entry.get("val_loss", math.inf)),
            }
        )
    rows.sort(key=lambda row: (float(row["quantized_val_loss"]), int(row["epoch"])))
    return rows[:limit]


def print_top_checkpoint_result(
    rank: int,
    epoch: int,
    model_path: Path,
    quantized_val_loss: float,
    result: MatchResult,
) -> None:
    status(
        "top-checkpoint "
        f"rank={rank} "
        f"epoch={epoch} "
        f"quantized_val_loss={quantized_val_loss:.6f} "
        f"model_nnq={model_path} "
        f"run={result.run_path} "
        f"wdl={result.wins}-{result.draws}-{result.losses} "
        f"score_pct={result.score_pct:.2f} "
        f"elo(current-reference)={format_metric(result.elo)} "
        f"[{format_metric(result.elo_lower)}, {format_metric(result.elo_upper)}] "
        f"avg_depth={result.avg_depth:.2f} "
        f"avg_nps={result.avg_nps:.0f} "
        f"avg_response_ms={result.avg_response_ms:.2f} "
        f"({result.decision})"
    )


def run_top_checkpoint_matches(
    root: Path,
    experiment_root: Path,
    model_report: dict[str, object],
    engine_template_path: Path,
    reference_source_path: Path,
    candidate_id: str,
    match_bin: Path,
    book_path: Path,
    time_ms: int,
    game_count: int,
    top_k: int = DEFAULT_TOP_CHECKPOINT_COUNT,
    dense_feature_mask: Sequence[int] = (),
) -> list[dict[str, object]]:
    if top_k <= 0 or game_count <= 0:
        return []
    rows = top_quantized_checkpoint_rows(model_report, top_k)
    if not rows:
        status(
            "trainer: skipping top checkpoint arena matches "
            f"(top_k={top_k} no quantized history)"
        )
        return []
    engine_stem = engine_template_path.stem
    results: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        epoch = int(row["epoch"])
        model_path = checkpoint_model_path(experiment_root, epoch)
        if not model_path.is_file():
            fail(f"missing checkpoint model for top checkpoint match: {model_path}")
        checkpoint_source = ranked_checkpoint_candidate_source_path(
            experiment_root,
            engine_stem,
            candidate_id,
            epoch,
        )
        checkpoint_source.parent.mkdir(parents=True, exist_ok=True)
        bake_runtime_source(engine_template_path, model_path, checkpoint_source, dense_feature_mask)
        match = run_match(
            root,
            match_bin,
            checkpoint_source,
            reference_source_path,
            book_path,
            time_ms,
            game_count,
        )
        print_top_checkpoint_result(
            rank,
            epoch,
            model_path,
            float(row["quantized_val_loss"]),
            match,
        )
        results.append(
            {
                "rank": rank,
                "epoch": epoch,
                "val_loss": float(row["val_loss"]),
                "quantized_val_loss": float(row["quantized_val_loss"]),
                "model_nnq": str(model_path),
                "candidate_source": str(checkpoint_source),
                "match": serialize_match(match),
            }
        )
    return results


def run_active_learning_round_checkpoint_matches(
    root: Path,
    match_bin: Path,
    training_root: Path,
    model_report: dict[str, object],
    engine_template_path: Path,
    reference_source_path: Path,
    candidate_id: str,
    book_path: Path,
    time_ms: int,
    game_count: int,
    round_index: int,
    top_k: int,
    dense_feature_mask: Sequence[int],
) -> list[dict[str, object]]:
    if top_k <= 0 or game_count <= 0:
        return []
    rows = top_quantized_checkpoint_rows(model_report, top_k)
    if not rows:
        status(
            "trainer: skipping active-learning round checkpoint matches "
            f"(round={round_index} top_k={top_k} no quantized history)"
        )
        return []
    engine_stem = engine_template_path.stem
    results: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        epoch = int(row["epoch"])
        model_path = checkpoint_model_path(training_root, epoch)
        if not model_path.is_file():
            fail(f"missing checkpoint model for active-learning round {round_index}: {model_path}")
        checkpoint_source = ranked_checkpoint_candidate_source_path(
            training_root,
            engine_stem,
            candidate_id,
            epoch,
        )
        checkpoint_source.parent.mkdir(parents=True, exist_ok=True)
        bake_runtime_source(engine_template_path, model_path, checkpoint_source, dense_feature_mask)
        match = run_match(
            root,
            match_bin,
            checkpoint_source,
            reference_source_path,
            book_path,
            time_ms,
            game_count,
        )
        print_active_learning_round_checkpoint_result(
            round_index,
            rank,
            epoch,
            model_path,
            float(row["quantized_val_loss"]),
            match,
        )
        results.append(
            {
                "rank": rank,
                "epoch": epoch,
                "val_loss": float(row["val_loss"]),
                "quantized_val_loss": float(row["quantized_val_loss"]),
                "model_nnq": str(model_path),
                "candidate_source": str(checkpoint_source),
                "match": serialize_match(match),
            }
        )
    return results


def load_model_report(model_json_path: Path, metrics_json_path: Path) -> dict[str, object]:
    if metrics_json_path.is_file():
        metrics_report = load_json_file(metrics_json_path)
        if "best_val_loss" in metrics_report and "best_quantized_val_loss" in metrics_report:
            return metrics_report
    model_report = load_json_file(model_json_path)
    if "best_val_loss" in model_report and "best_quantized_val_loss" in model_report:
        return model_report
    fail(
        "training report is missing best-loss fields in both "
        f"{metrics_json_path} and {model_json_path}"
    )


def summarize_candidate_finding(report: dict[str, object]) -> str:
    if report["status"] == "failed":
        return f"Finding: {report['failure_summary']}"
    if report.get("beats_baseline"):
        return "Finding: This candidate cleared the confirm-match bar and beat the rc5 baseline."
    final_match = report.get("final_match")
    if final_match is None:
        return "Finding: Training completed cleanly, checkpoint screening ran, and the final baseline match was skipped."
    elo = float(final_match["elo"])
    if elo > 0:
        return "Finding: This candidate gained Elo over rc5, but it did not clear the final confidence bar."
    if elo < 0:
        return "Finding: This candidate lost Elo against rc5 despite completing training cleanly."
    return "Finding: This candidate was effectively flat against rc5."


def ensure_labbook_header() -> None:
    if LABBOOK_PATH.is_file():
        return
    LABBOOK_PATH.write_text(
        "# RC5 Experiment Labbook\n\n"
        "Append-only notes written by `tools/trainer.py`.\n",
        encoding="utf-8",
    )


def append_labbook_entry(title: str, lines: Sequence[str]) -> None:
    ensure_labbook_header()
    with LABBOOK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n")
        for line in lines:
            handle.write(f"{line}\n")


def append_candidate_labbook_entry(report: dict[str, object]) -> None:
    if report.get("labbook_logged"):
        return
    title = f"{report['completed_at']} - {report['candidate_id']}"
    lines = [
        f"- Timestamp: {report['completed_at']}",
        f"- Candidate ID: `{report['candidate_id']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Profile: `{report['profile']}`",
        f"- Tier: `{report['tier']}`",
    ]
    if report["status"] == "failed":
        lines.extend(
            [
                f"- Stage Reached: `{report['failure_stage']}`",
                f"- Error Summary: {report['failure_summary']}",
                f"- Runtime: {report['wall_clock_seconds']:.2f}s",
                f"- Artifacts: `{report['experiment_root']}`",
                f"- Finding: {report['failure_summary']}",
            ]
        )
    else:
        final_match = report.get("final_match")
        lines.extend(
            [
                (
                    "- Corpus Sizes: "
                    f"train={report['train_samples']} "
                    f"val={report['validation_samples']} "
                    f"diag_val={report['diagnostic_validation_samples']}"
                ),
                (
                    "- Hyperparameters: "
                    f"lr={report['learning_rate']} "
                    f"wd={report['weight_decay']} "
                    f"batch={report['batch_size']} "
                    f"epochs={report['epochs']} "
                    f"runtime_loss_interval={int(report.get('runtime_loss_interval', DEFAULT_RUNTIME_LOSS_INTERVAL))} "
                    f"lambda_mix={float(report.get('lambda_mix', {}).get('score', DEFAULT_LAMBDA_MIX[0])):.3f}/"
                    f"{float(report.get('lambda_mix', {}).get('result', DEFAULT_LAMBDA_MIX[1])):.3f}"
                ),
                (
                    "- Active Learning: "
                    + (
                        (
                            (
                                (
                                    f"requested_chunk_size={sample_count_label(int(report['active_learning']['requested_chunk_size']))} "
                                )
                                if report['active_learning'].get('requested_chunk_size') is not None
                                else ""
                            )
                            + f"chunks={','.join(sample_count_label(int(value)) for value in report['active_learning']['chunks'])} "
                            f"pool_train={int(report.get('pool_train_samples', report['train_samples']))} "
                            f"eligible_pool={int(report['active_learning']['eligible_pool_samples'])} "
                            f"patience={int(report['active_learning'].get('patience', ACTIVE_LEARNING_PATIENCE))} "
                            f"pool_cutoff_cp={report['active_learning'].get('pool_cutoff_cp', 'none')}"
                        )
                        if report.get("active_learning") is not None
                        else "disabled"
                    )
                ),
                (
                    "- Random-Val Loss: "
                    f"val={report['best_val_loss']:.6f} "
                    f"quantized={report['best_quantized_val_loss']:.6f}"
                ),
                f"- Diagnostic-Val Loss: {report['diagnostic_val_loss']:.6f}",
                (
                    "- Control Relative: "
                    f"control=`{report.get('control_candidate_id', 'n/a')}` "
                    f"delta_qval={report.get('delta_qval_vs_control', 0.0):+.6f} "
                    f"delta_diag={report.get('delta_diag_vs_control', 0.0):+.6f} "
                    f"delta_elo={report.get('delta_elo_vs_control', 0.0):+.2f} "
                    f"outcome={report.get('control_outcome', 'n/a')}"
                )
                if report.get("control_candidate_id") is not None
                else "- Control Relative: n/a",
                (
                    "- Match Result: "
                    + (
                        (
                            f"games={final_match['games']} "
                            f"wdl={final_match['wins']}-{final_match['draws']}-{final_match['losses']} "
                            f"elo={final_match['elo']:+.2f} "
                            f"[{final_match['elo_lower']:+.2f}, {final_match['elo_upper']:+.2f}]"
                        )
                        if final_match is not None
                        else "skipped (`final_match_games=0`)"
                    )
                ),
                f"- Runtime: {report['wall_clock_seconds']:.2f}s",
                (
                    "- Artifact Paths: "
                    f"corpus=`{report['corpus_dir']}` "
                    f"experiment=`{report['experiment_root']}` "
                    f"model=`{report['model_nnq']}` "
                    + (f"run=`{final_match['run_path']}`" if final_match is not None else "run=`skipped`")
                ),
                summarize_candidate_finding(report),
            ]
        )
    append_labbook_entry(title, lines)
    report["labbook_logged"] = True


def write_candidate_report(candidate_json_path: Path, report: dict[str, object]) -> None:
    write_json(candidate_json_path, report)


def load_candidate_report(candidate_json_path: Path) -> dict[str, object] | None:
    if not candidate_json_path.is_file():
        return None
    return load_json_file(candidate_json_path)


def leaderboard_json_path(sweep_dir: Path) -> Path:
    return sweep_dir / "leaderboard.json"


def leaderboard_md_path(sweep_dir: Path) -> Path:
    return sweep_dir / "leaderboard.md"


def stage_summary_path(sweep_dir: Path, stage_name: str) -> Path:
    return sweep_dir / f"stage_summary_{stage_name}.json"


def leaderboard_sort_key(report: dict[str, object]) -> tuple[float, float, float, str]:
    if report["status"] == "failed":
        return (-1.0, -math.inf, math.inf, report["candidate_id"])
    return (
        float(stage_order_value(str(report["tier"]))),
        float(report["final_match"]["elo"]),
        -float(report["best_quantized_val_loss"]),
        str(report["candidate_id"]),
    )


def update_leaderboard(sweep_dir: Path) -> None:
    candidates_dir = sweep_dir / "candidates"
    reports: list[dict[str, object]] = []
    if candidates_dir.is_dir():
        for path in sorted(candidates_dir.glob("*/candidate.json")):
            reports.append(load_json_file(path))
    reports.sort(key=leaderboard_sort_key, reverse=True)
    leaderboard = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": reports,
    }
    write_json(leaderboard_json_path(sweep_dir), leaderboard)
    lines = [
        "# RC5 Sweep Leaderboard",
        "",
        "| Candidate | Tier | Profile | Recipe | Status | Q-Val Loss | Diag Loss | Elo | Delta Q | Delta Diag | Delta Elo | Vs Control | CI | WDL |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for report in reports:
        if report["status"] == "failed":
            lines.append(
                f"| `{report['candidate_id']}` | {report['tier']} | {report['profile']} | {report['recipe']} | failed | - | - | - | - | - | - | - | - | - |"
            )
            continue
        final_match = report["final_match"]
        lines.append(
            "| "
            f"`{report['candidate_id']}` | "
            f"{report['tier']} | "
            f"{report['profile']} | "
            f"{report['recipe']} | "
            f"{report['status']} | "
            f"{report['best_quantized_val_loss']:.6f} | "
            f"{report['diagnostic_val_loss']:.6f} | "
            f"{final_match['elo']:+.2f} | "
            f"{float(report.get('delta_qval_vs_control', 0.0)):+.6f} | "
            f"{float(report.get('delta_diag_vs_control', 0.0)):+.6f} | "
            f"{float(report.get('delta_elo_vs_control', 0.0)):+.2f} | "
            f"{report.get('control_outcome', 'n/a')} | "
            f"[{final_match['elo_lower']:+.2f}, {final_match['elo_upper']:+.2f}] | "
            f"{final_match['wins']}-{final_match['draws']}-{final_match['losses']} |"
        )
    leaderboard_md_path(sweep_dir).write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_reports(reports: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [report for report in reports if report.get("status") == "complete"]


def select_control_report(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    complete = completed_reports(reports)
    for report in complete:
        if report["profile"] == "random" and report["recipe"] == "base":
            return report
    for report in complete:
        if report["profile"] == "random":
            return report
    fail("stage requires a completed random control candidate")


def control_outcome(report: dict[str, object], control: dict[str, object]) -> str:
    if report["status"] != "complete":
        return "failed"
    if report["candidate_id"] == control["candidate_id"]:
        return "control"
    delta_elo = float(report["final_match"]["elo"]) - float(control["final_match"]["elo"])
    report_lower = float(report["final_match"]["elo_lower"])
    report_upper = float(report["final_match"]["elo_upper"])
    control_elo = float(control["final_match"]["elo"])
    control_lower = float(control["final_match"]["elo_lower"])
    if delta_elo >= 10.0 or report_lower > control_lower:
        return "beat"
    if delta_elo <= -20.0 or report_upper < control_elo - 10.0:
        return "lost"
    return "tied"


def annotate_control_relative_fields(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    control = select_control_report(reports)
    control_qval = float(control["best_quantized_val_loss"])
    control_diag = float(control["diagnostic_val_loss"])
    control_elo = float(control["final_match"]["elo"])
    for report in reports:
        if report["status"] != "complete":
            report["control_candidate_id"] = control["candidate_id"]
            report["control_outcome"] = "failed"
            continue
        report["control_candidate_id"] = control["candidate_id"]
        report["delta_qval_vs_control"] = float(report["best_quantized_val_loss"]) - control_qval
        report["delta_diag_vs_control"] = float(report["diagnostic_val_loss"]) - control_diag
        report["delta_elo_vs_control"] = float(report["final_match"]["elo"]) - control_elo
        report["control_outcome"] = control_outcome(report, control)
        write_candidate_report(Path(str(report["experiment_root"])) / "candidate.json", report)
    return control


def fast_lane_rank_key(report: dict[str, object]) -> tuple[float, float, float, float, float, str]:
    if report["status"] != "complete":
        return (-1.0, -math.inf, -math.inf, -math.inf, -math.inf, str(report["candidate_id"]))
    outcome_rank = {"beat": 3.0, "tied": 2.0, "control": 1.0, "lost": 0.0}.get(
        str(report.get("control_outcome", "lost")),
        0.0,
    )
    return (
        outcome_rank,
        -float(report.get("delta_qval_vs_control", 0.0)),
        -float(report.get("delta_diag_vs_control", 0.0)),
        float(report.get("delta_elo_vs_control", 0.0)),
        -float(report["best_quantized_val_loss"]),
        str(report["candidate_id"]),
    )


def candidate_is_competitive(report: dict[str, object]) -> bool:
    if report["status"] != "complete":
        return False
    if report.get("control_outcome") == "lost":
        return False
    if report.get("control_outcome") == "beat":
        return True
    return (
        float(report.get("delta_qval_vs_control", 0.0)) < 0.0
        or float(report.get("delta_diag_vs_control", 0.0)) < 0.0
        or float(report.get("delta_elo_vs_control", 0.0)) > 0.0
    )


def choose_stage_candidates(
    reports: Sequence[dict[str, object]],
    count: int,
) -> tuple[list[dict[str, object]], list[str]]:
    control = annotate_control_relative_fields(reports)
    complete = completed_reports(reports)
    if not complete:
        fail("promotion requires at least one completed candidate")
    threshold = float(control["best_quantized_val_loss"]) * 1.02
    sorted_reports = sorted(complete, key=fast_lane_rank_key, reverse=True)
    winner = sorted_reports[0]
    challengers = [
        report
        for report in sorted_reports
        if report["candidate_id"] != control["candidate_id"]
        and float(report["best_quantized_val_loss"]) <= threshold
        and candidate_is_competitive(report)
    ]
    chosen: list[dict[str, object]] = [winner]
    for challenger in challengers:
        if len(chosen) >= count:
            break
        if challenger["candidate_id"] == winner["candidate_id"]:
            continue
        chosen.append(challenger)
    reasons: list[str] = []
    for report in chosen:
        if report["candidate_id"] == control["candidate_id"]:
            reasons.append(
                "no challenger beat the random control under the current guardrails, "
                "so the control remains the winner"
            )
            continue
        reasons.append(
            f"`{report['candidate_id']}` stayed competitive with the random control "
            f"(delta_qval {float(report['delta_qval_vs_control']):+.6f}, "
            f"delta_diag {float(report['delta_diag_vs_control']):+.6f}, "
            f"delta_elo {float(report['delta_elo_vs_control']):+.2f}, "
            f"outcome {report['control_outcome']})."
        )
    return chosen, reasons


def append_stage_summary(
    sweep_dir: Path,
    stage_name: str,
    reports: Sequence[dict[str, object]],
    promoted: Sequence[dict[str, object]],
    reasons: Sequence[str],
) -> None:
    summary_file = stage_summary_path(sweep_dir, stage_name)
    existing = load_candidate_report(summary_file)
    if existing is not None and existing.get("labbook_logged"):
        return
    control = annotate_control_relative_fields(reports)
    payload = {
        "stage": stage_name,
        "control_candidate": control["candidate_id"],
        "control_elo": control["final_match"]["elo"],
        "promoted_candidates": [report["candidate_id"] for report in promoted],
        "reasons": list(reasons),
        "labbook_logged": False,
    }
    write_json(summary_file, payload)
    lines = [
        f"- Control Result: `{control['candidate_id']}` at {control['final_match']['elo']:+.2f} Elo "
        f"with q-val {control['best_quantized_val_loss']:.6f}",
        "- Promoted Candidates: "
        + (", ".join(f"`{report['candidate_id']}`" for report in promoted) if promoted else "none"),
    ]
    if promoted and promoted[0]["candidate_id"] == control["candidate_id"]:
        lines.append("- Why: No challenger beat the control, so random remains the winner for this stage.")
    for reason in reasons:
        lines.append(f"- Why: {reason}")
    append_labbook_entry(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} - {stage_name} stage summary", lines)
    payload["labbook_logged"] = True
    write_json(summary_file, payload)


def maybe_confirm_match(
    root: Path,
    match_bin: Path,
    candidate_source: Path,
    reference_source: Path,
    book_path: Path,
    time_ms: int,
    final_match: MatchResult,
) -> MatchResult | None:
    if not (final_match.elo > 5.0 and final_match.elo_lower > -5.0):
        return None
    status("trainer: running 2000-game confirm match")
    return run_match(
        root,
        match_bin,
        candidate_source,
        reference_source,
        book_path,
        time_ms,
        DEFAULT_CONFIRM_GAMES,
    )


def print_final_result(
    spec: CandidateSpec,
    corpus: CorpusPaths,
    output_source_path: Path,
    model_json: Path,
    model_nnq: Path,
    loss_curve: Path,
    final_match: MatchResult | None,
    confirm_match: MatchResult | None,
) -> None:
    print(f"candidate_source: {output_source_path}")
    print(f"template_source: {spec.engine_template_path}")
    print(
        "reference_source: "
        f"{spec.reference_source_path if spec.reference_source_path is not None else spec.engine_template_path}"
    )
    print(f"dataset: {spec.dataset_path}")
    print(f"profile: {spec.profile.name}")
    print(f"tier: {spec.tier.name}")
    print(f"recipe: {spec.recipe.name}")
    print(f"train_dataset: {corpus.train_path}")
    print(f"val_dataset: {corpus.val_path}")
    print(f"diagnostic_val_dataset: {corpus.diagnostic_val_path}")
    print(f"manifest: {corpus.manifest_path}")
    print(f"model: {model_json}")
    print(f"model_nnq: {model_nnq}")
    print(f"loss_curve: {loss_curve}")
    print(f"book: {resolve_checkpoint_book(workspace_root())}")
    print(f"time_ms: {spec.time_ms}")
    if final_match is None:
        print("run: skipped")
        print("final_match: skipped (`final_match_games=0`)")
    else:
        print(f"run: {final_match.run_path}")
        print(f"wdl: {final_match.wins}-{final_match.draws}-{final_match.losses}")
        print(
            "elo(current-reference): "
            f"{format_metric(final_match.elo)} "
            f"[{format_metric(final_match.elo_lower)}, {format_metric(final_match.elo_upper)}] "
            f"({final_match.decision})"
        )
    if confirm_match is not None:
        print(f"confirm_run: {confirm_match.run_path}")
        print(
            "confirm_elo(current-reference): "
            f"{format_metric(confirm_match.elo)} "
            f"[{format_metric(confirm_match.elo_lower)}, {format_metric(confirm_match.elo_upper)}] "
            f"({confirm_match.decision})"
        )


def serialize_match(result: MatchResult) -> dict[str, object]:
    return {
        "run_path": str(result.run_path),
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "score_pct": result.score_pct,
        "elo": result.elo,
        "elo_lower": result.elo_lower,
        "elo_upper": result.elo_upper,
        "avg_depth": result.avg_depth,
        "avg_nps": result.avg_nps,
        "avg_response_ms": result.avg_response_ms,
        "decision": result.decision,
        "games": result.games,
    }


def deserialize_match(payload: dict[str, object] | None) -> MatchResult | None:
    if payload is None:
        return None
    return MatchResult(
        run_path=Path(str(payload["run_path"])),
        wins=int(payload["wins"]),
        draws=int(payload["draws"]),
        losses=int(payload["losses"]),
        score_pct=float(payload.get("score_pct", 0.0)),
        elo=float(payload["elo"]),
        elo_lower=float(payload["elo_lower"]),
        elo_upper=float(payload["elo_upper"]),
        avg_depth=float(payload.get("avg_depth", 0.0)),
        avg_nps=float(payload.get("avg_nps", 0.0)),
        avg_response_ms=float(payload.get("avg_response_ms", 0.0)),
        decision=str(payload["decision"]),
        games=int(payload["games"]),
    )


def corpus_paths_from_report(report: dict[str, object]) -> CorpusPaths:
    corpus_dir = Path(str(report["corpus_dir"]))
    return CorpusPaths(
        corpus_dir=corpus_dir,
        train_path=Path(str(report["train_dataset"])),
        val_path=Path(str(report["val_dataset"])),
        diagnostic_val_path=Path(str(report["diagnostic_val_dataset"])),
        manifest_path=Path(str(report["manifest"])),
        profile_path=corpus_dir / "profile.json",
        dataset_cache_dir=corpus_dir / "dataset_cache",
        train_samples=int(report["train_samples"]),
        val_samples=int(report["validation_samples"]),
        diagnostic_val_samples=int(report["diagnostic_validation_samples"]),
    )


def candidate_failure_report(
    spec: CandidateSpec,
    experiment_root: Path,
    corpus: CorpusPaths,
    stage: str,
    error: Exception,
    wall_clock_seconds: float,
) -> dict[str, object]:
    return {
        "status": "failed",
        "candidate_id": spec.candidate_id,
        "dataset": str(spec.dataset_path),
        "profile": spec.profile.name,
        "tier": spec.tier.name,
        "recipe": spec.recipe.name,
        "seed": spec.seed,
        "reference_source": str(
            spec.reference_source_path
            if spec.reference_source_path is not None
            else spec.engine_template_path
        ),
        "feature_set": spec.feature_set_name,
        "architecture": spec.architecture_spec,
        "activation": spec.activation_name,
        "dense_feature_mask": list(spec.dense_feature_mask),
        "repeat_occurrence_weight": spec.repeat_occurrence_weight,
        "class_weighting": spec.class_weighting,
        "ema_decay": spec.ema_decay,
        "lambda_mix": {
            "score": spec.lambda_score,
            "result": spec.lambda_result,
        },
        "failure_stage": stage,
        "failure_summary": str(error),
        "wall_clock_seconds": wall_clock_seconds,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_root": str(experiment_root),
        "corpus_dir": str(corpus.corpus_dir),
        "train_samples": spec.train_samples,
        "pool_train_samples": spec.pool_train_samples,
        "max_abs_score": spec.max_abs_score,
        "validation_samples": spec.validation_samples,
        "diagnostic_validation_samples": corpus.diagnostic_val_samples,
        "learning_rate": spec.recipe.learning_rate,
        "weight_decay": spec.recipe.weight_decay,
        "batch_size": spec.recipe.batch_size,
        "epochs": spec.epochs,
        "patience": patience_for_spec(spec),
        "runtime_loss_interval": runtime_loss_interval_for_spec(spec),
        "active_learning": (
            {
                "version": ACTIVE_LEARNING_VERSION,
                "chunks": (
                    list(spec.active_learning.manual_chunk_sizes)
                    if spec.active_learning.manual_chunk_sizes is not None
                    else None
                ),
                "chunk_size": spec.active_learning.auto_chunk_size,
                "pool_cutoff_cp": spec.active_learning.pool_cutoff_cp,
            }
            if spec.active_learning is not None
            else None
        ),
        "labbook_logged": False,
    }


def run_active_learning_candidate(
    root: Path,
    nnue_bin: Path,
    match_bin: Path,
    book_path: Path,
    spec: CandidateSpec,
    experiment_root: Path,
    pool_corpus: CorpusPaths,
) -> dict[str, object]:
    active_learning = spec.active_learning
    if active_learning is None:
        fail("internal error: active-learning runner requires an active-learning spec")
    checkpoint_epochs = checkpoint_epochs_for_candidate(spec)
    print_startup(spec, pool_corpus, experiment_root, book_path, checkpoint_epochs)
    active_root = active_learning_root_dir(experiment_root)
    active_root.mkdir(parents=True, exist_ok=True)
    pool_abs_scores = ensure_active_learning_pool_abs_scores(
        experiment_root=experiment_root,
        pool_path=pool_corpus.train_path,
        expected_samples=pool_corpus.train_samples,
    )
    eligible_mask = np.ones(pool_corpus.train_samples, dtype=bool)
    if active_learning.pool_cutoff_cp is not None:
        eligible_mask &= pool_abs_scores <= float(active_learning.pool_cutoff_cp)
    eligible_pool_samples = int(eligible_mask.sum())
    resolved_chunk_sizes = resolve_active_learning_chunk_sizes(active_learning, eligible_pool_samples)
    total_selected_samples = sum(resolved_chunk_sizes)
    round_match_top_k = active_learning.round_match_top_k
    round_patience = patience_for_spec(spec)

    round_reports: list[dict[str, object]] = []
    selected_mask = np.zeros(pool_corpus.train_samples, dtype=bool)
    final_round_corpus: CorpusPaths | None = None
    final_training_root: Path | None = None
    final_model_json: Path | None = None
    final_model_nnq: Path | None = None
    final_metrics_json: Path | None = None
    final_loss_curve: Path | None = None
    previous_model_nnq: Path | None = None
    final_round_match: MatchResult | None = None
    final_round_confirm_match: MatchResult | None = None

    for round_spec in active_learning_round_specs(resolved_chunk_sizes):
        round_root = active_learning_round_root(experiment_root, round_spec)
        round_root.mkdir(parents=True, exist_ok=True)
        added_ids_path = round_root / "added_ids.npy"
        selected_ids_path = round_root / "selected_ids.npy"
        selection_meta_path = round_root / "selection_meta.json"
        round_result_path = round_root / "round_result.json"
        training_root = round_root / "training"
        training_root.mkdir(parents=True, exist_ok=True)

        if round_result_path.is_file():
            if not added_ids_path.is_file():
                fail(f"missing active-learning selection file {added_ids_path}")
            added_ids = np.load(added_ids_path, allow_pickle=False).astype(np.uint32, copy=False)
            selected_mask[added_ids] = True
            selected_indices = np.flatnonzero(selected_mask).astype(np.uint32, copy=False)
            round_corpus = ensure_active_learning_round_corpus(
                round_root=round_root,
                pool_corpus=pool_corpus,
                round_spec=round_spec,
                selected_indices=selected_indices,
                profile=spec.profile,
                tier=spec.tier,
                seed=spec.seed,
                pool_cutoff_cp=active_learning.pool_cutoff_cp,
            )
            round_report = load_json_file(round_result_path)
            round_source = candidate_source_path(training_root, spec.engine_template_path.stem, round_candidate_id := f"{spec.candidate_id}_{round_spec.round_slug}")
            round_match = deserialize_match(
                None if round_report.get("round_match") is None else dict(round_report["round_match"])
            )
            round_confirm_match = deserialize_match(
                None if round_report.get("round_confirm_match") is None else dict(round_report["round_confirm_match"])
            )
            round_top_checkpoint_matches = round_report.get("round_top_checkpoint_matches")
            if round_match_top_k > 1 and not isinstance(round_top_checkpoint_matches, list):
                model_report = load_model_report(training_root / "model.json", training_root / "metrics.json")
                round_top_checkpoint_matches = run_active_learning_round_checkpoint_matches(
                    root=root,
                    match_bin=match_bin,
                    training_root=training_root,
                    model_report=model_report,
                    engine_template_path=spec.engine_template_path,
                    reference_source_path=(
                        spec.reference_source_path
                        if spec.reference_source_path is not None
                        else spec.engine_template_path
                    ),
                    candidate_id=round_candidate_id,
                    book_path=book_path,
                    time_ms=spec.time_ms,
                    game_count=spec.tier.final_match_games,
                    round_index=round_spec.index,
                    top_k=round_match_top_k,
                    dense_feature_mask=spec.dense_feature_mask,
                )
                round_report["round_top_checkpoint_matches"] = round_top_checkpoint_matches
            if round_match is None:
                round_match = run_model_match(
                    root=root,
                    match_bin=match_bin,
                    engine_template_path=spec.engine_template_path,
                    reference_source_path=(
                        spec.reference_source_path
                        if spec.reference_source_path is not None
                        else spec.engine_template_path
                    ),
                    output_source_path=round_source,
                    model_nnq=training_root / "model.nnq",
                    dense_feature_mask=spec.dense_feature_mask,
                    book_path=book_path,
                    time_ms=spec.time_ms,
                    game_count=spec.tier.final_match_games,
                )
                round_report["round_match"] = serialize_match(round_match)
            print_active_learning_round_result(round_spec.index, training_root / "model.nnq", round_match)
            if round_spec.index == len(resolved_chunk_sizes) and round_confirm_match is None:
                round_confirm_match = maybe_confirm_match(
                    root=root,
                    match_bin=match_bin,
                    candidate_source=round_source,
                    reference_source=(
                        spec.reference_source_path
                        if spec.reference_source_path is not None
                        else spec.engine_template_path
                    ),
                    book_path=book_path,
                    time_ms=spec.time_ms,
                    final_match=round_match,
                )
                round_report["round_confirm_match"] = (
                    serialize_match(round_confirm_match) if round_confirm_match is not None else None
                )
            if "patience" not in round_report:
                round_report["patience"] = round_patience
            write_json(round_result_path, round_report)
            round_reports.append(round_report)
            previous_model_nnq = Path(str(round_report["model_nnq"]))
            final_round_corpus = round_corpus
            final_training_root = training_root
            final_model_json = training_root / "model.json"
            final_model_nnq = training_root / "model.nnq"
            final_metrics_json = training_root / "metrics.json"
            final_loss_curve = training_root / "loss_curves.svg"
            if round_spec.index == len(resolved_chunk_sizes):
                final_round_match = round_match
                final_round_confirm_match = round_confirm_match
            continue

        if added_ids_path.is_file():
            added_ids = np.load(added_ids_path, allow_pickle=False).astype(np.uint32, copy=False)
        else:
            if round_spec.index == 1:
                added_ids = select_active_learning_random_ids(
                    eligible_mask=eligible_mask,
                    selected_mask=selected_mask,
                    count=round_spec.chunk_size,
                    seed=spec.seed,
                    tag=f"active-learning:{spec.candidate_id}:round{round_spec.index}",
                )
                selection_meta = {
                    "round": round_spec.index,
                    "strategy": "random",
                    "chunk_size": round_spec.chunk_size,
                    "selected_samples": round_spec.selected_samples,
                    "pool_cutoff_cp": active_learning.pool_cutoff_cp,
                }
            else:
                if previous_model_nnq is None:
                    fail("active-learning disagreement round is missing the previous model")
                added_ids, disagreement_meta = select_active_learning_disagreement_ids(
                    root=root,
                    nnue_bin=nnue_bin,
                    pool_path=pool_corpus.train_path,
                    model_path=previous_model_nnq,
                    teacher_root=round_root / "teacher_targets",
                    eligible_mask=eligible_mask,
                    selected_mask=selected_mask,
                    count=round_spec.chunk_size,
                )
                selection_meta = dict(disagreement_meta)
                selection_meta.update(
                    {
                        "round": round_spec.index,
                        "strategy": "largest_search_disagreement",
                        "chunk_size": round_spec.chunk_size,
                        "selected_samples": round_spec.selected_samples,
                        "model_path": str(previous_model_nnq),
                        "pool_cutoff_cp": active_learning.pool_cutoff_cp,
                    }
                )
            np.save(added_ids_path, added_ids, allow_pickle=False)
            write_json(selection_meta_path, selection_meta)
        selected_mask[added_ids] = True
        selected_indices = np.flatnonzero(selected_mask).astype(np.uint32, copy=False)
        np.save(selected_ids_path, selected_indices, allow_pickle=False)

        round_corpus = ensure_active_learning_round_corpus(
            round_root=round_root,
            pool_corpus=pool_corpus,
            round_spec=round_spec,
            selected_indices=selected_indices,
            profile=spec.profile,
            tier=spec.tier,
            seed=spec.seed,
            pool_cutoff_cp=active_learning.pool_cutoff_cp,
        )
        final_round = round_spec.index == len(resolved_chunk_sizes)
        round_candidate_id = f"{spec.candidate_id}_{round_spec.round_slug}"
        status(
            f"trainer: active-learning round {round_spec.index}/{len(resolved_chunk_sizes)} "
            f"chunk={round_spec.chunk_size} selected={round_spec.selected_samples}"
        )
        save_full_epoch_checkpoints = round_match_top_k > 1
        ensure_trained_model(
            root=root,
            nnue_bin=nnue_bin,
            corpus=round_corpus,
            experiment_root=training_root,
            spec=spec,
            match_bin=match_bin,
            book_path=book_path,
            candidate_id=round_candidate_id,
            checkpoint_epochs=[] if save_full_epoch_checkpoints else (checkpoint_epochs if final_round else []),
            enable_checkpoints=final_round or save_full_epoch_checkpoints,
            patience=round_patience,
            checkpoint_save_interval=1 if save_full_epoch_checkpoints else None,
        )
        model_json = training_root / "model.json"
        model_nnq = training_root / "model.nnq"
        metrics_json = training_root / "metrics.json"
        loss_curve = training_root / "loss_curves.svg"
        model_report = load_model_report(model_json, metrics_json)
        selection_meta = load_json_file(selection_meta_path) if selection_meta_path.is_file() else {}
        round_source = candidate_source_path(training_root, spec.engine_template_path.stem, round_candidate_id)
        round_top_checkpoint_matches = (
            run_active_learning_round_checkpoint_matches(
                root=root,
                match_bin=match_bin,
                training_root=training_root,
                model_report=model_report,
                engine_template_path=spec.engine_template_path,
                reference_source_path=(
                    spec.reference_source_path
                    if spec.reference_source_path is not None
                    else spec.engine_template_path
                ),
                candidate_id=round_candidate_id,
                book_path=book_path,
                time_ms=spec.time_ms,
                game_count=spec.tier.final_match_games,
                round_index=round_spec.index,
                top_k=round_match_top_k,
                dense_feature_mask=spec.dense_feature_mask,
            )
            if round_match_top_k > 1
            else []
        )
        round_match: MatchResult | None = None
        best_epoch = model_report.get("best_epoch")
        if isinstance(best_epoch, (int, float)):
            for entry in round_top_checkpoint_matches:
                if int(entry["epoch"]) == int(best_epoch):
                    round_match = deserialize_match(dict(entry["match"]))
                    break
        if round_match is None:
            round_match = run_model_match(
                root=root,
                match_bin=match_bin,
                engine_template_path=spec.engine_template_path,
                reference_source_path=(
                    spec.reference_source_path
                    if spec.reference_source_path is not None
                    else spec.engine_template_path
                ),
                output_source_path=round_source,
                model_nnq=model_nnq,
                dense_feature_mask=spec.dense_feature_mask,
                book_path=book_path,
                time_ms=spec.time_ms,
                game_count=spec.tier.final_match_games,
            )
        else:
            bake_runtime_source(spec.engine_template_path, model_nnq, round_source, spec.dense_feature_mask)
        print_active_learning_round_result(round_spec.index, model_nnq, round_match)
        round_confirm_match = (
            maybe_confirm_match(
                root=root,
                match_bin=match_bin,
                candidate_source=round_source,
                reference_source=(
                    spec.reference_source_path
                    if spec.reference_source_path is not None
                    else spec.engine_template_path
                ),
                book_path=book_path,
                time_ms=spec.time_ms,
                final_match=round_match,
            )
            if final_round
            else None
        )
        round_report = {
            "round": round_spec.index,
            "round_slug": round_spec.round_slug,
            "chunk_size": round_spec.chunk_size,
            "selected_samples": round_spec.selected_samples,
            "patience": round_patience,
            "selection_meta": selection_meta,
            "added_ids_path": str(added_ids_path),
            "selected_ids_path": str(selected_ids_path),
            "corpus_dir": str(round_corpus.corpus_dir),
            "train_dataset": str(round_corpus.train_path),
            "val_dataset": str(round_corpus.val_path),
            "diagnostic_val_dataset": str(round_corpus.diagnostic_val_path),
            "manifest": str(round_corpus.manifest_path),
            "training_dir": str(training_root),
            "model_json": str(model_json),
            "model_nnq": str(model_nnq),
            "metrics_json": str(metrics_json),
            "loss_curve": str(loss_curve),
            "best_epoch": model_report.get("best_epoch"),
            "best_val_loss": float(model_report["best_val_loss"]),
            "best_quantized_val_loss": float(model_report["best_quantized_val_loss"]),
            "round_candidate_source": str(round_source),
            "round_top_checkpoint_matches": round_top_checkpoint_matches,
            "round_match": serialize_match(round_match),
            "round_confirm_match": serialize_match(round_confirm_match) if round_confirm_match is not None else None,
        }
        write_json(round_result_path, round_report)
        round_reports.append(round_report)
        previous_model_nnq = model_nnq
        final_round_corpus = round_corpus
        final_training_root = training_root
        final_model_json = model_json
        final_model_nnq = model_nnq
        final_metrics_json = metrics_json
        final_loss_curve = loss_curve
        if final_round:
            final_round_match = round_match
            final_round_confirm_match = round_confirm_match

    if (
        final_round_corpus is None
        or final_training_root is None
        or final_model_json is None
        or final_model_nnq is None
        or final_metrics_json is None
        or final_loss_curve is None
        or final_round_match is None
    ):
        fail("active-learning run did not produce a final round")

    output_source = candidate_source_path(
        experiment_root,
        spec.engine_template_path.stem,
        spec.candidate_id,
    )
    diagnostic_report = run_runtime_loss(
        root,
        nnue_bin,
        final_model_nnq,
        final_round_corpus.diagnostic_val_path,
        spec.lambda_score,
    )
    status(f"trainer: baking {final_model_nnq} into {output_source}")
    bake_runtime_source(spec.engine_template_path, final_model_nnq, output_source, spec.dense_feature_mask)
    final_match = final_round_match
    confirm_match = final_round_confirm_match
    model_report = load_model_report(final_model_json, final_metrics_json)
    report = {
        "status": "complete",
        "candidate_id": spec.candidate_id,
        "mode": spec.mode,
        "sweep": spec.sweep_name,
        "dataset": str(spec.dataset_path),
        "profile": spec.profile.name,
        "tier": spec.tier.name,
        "recipe": spec.recipe.name,
        "seed": spec.seed,
        "backend": spec.backend,
        "reference_source": str(
            spec.reference_source_path
            if spec.reference_source_path is not None
            else spec.engine_template_path
        ),
        "time_ms": spec.time_ms,
        "lambda_mix": {
            "score": spec.lambda_score,
            "result": spec.lambda_result,
        },
        "dense_feature_mask": list(spec.dense_feature_mask),
        "repeat_occurrence_weight": spec.repeat_occurrence_weight,
        "ema_decay": spec.ema_decay,
        "train_samples": total_selected_samples,
        "pool_train_samples": spec.pool_train_samples,
        "max_abs_score": spec.max_abs_score,
        "validation_samples": spec.validation_samples,
        "diagnostic_validation_samples": final_round_corpus.diagnostic_val_samples,
        "epochs": spec.epochs,
        "batch_size": spec.recipe.batch_size,
        "learning_rate": spec.recipe.learning_rate,
        "weight_decay": spec.recipe.weight_decay,
        "dropout": spec.recipe.dropout,
        "runtime_loss_interval": runtime_loss_interval_for_spec(spec),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_epoch": model_report.get("best_epoch"),
        "best_val_loss": float(model_report["best_val_loss"]),
        "best_quantized_val_loss": float(model_report["best_quantized_val_loss"]),
        "diagnostic_val_loss": float(diagnostic_report["loss"]),
        "diagnostic_report": diagnostic_report,
        "final_match": serialize_match(final_match),
        "confirm_match": serialize_match(confirm_match) if confirm_match is not None else None,
        "beats_baseline": bool(confirm_match is not None and confirm_match.elo_lower > 0.0),
        "experiment_root": str(experiment_root),
        "corpus_dir": str(final_round_corpus.corpus_dir),
        "pool_corpus_dir": str(pool_corpus.corpus_dir),
        "train_dataset": str(final_round_corpus.train_path),
        "pool_train_dataset": str(pool_corpus.train_path),
        "val_dataset": str(final_round_corpus.val_path),
        "diagnostic_val_dataset": str(final_round_corpus.diagnostic_val_path),
        "manifest": str(final_round_corpus.manifest_path),
        "model_json": str(final_model_json),
        "model_nnq": str(final_model_nnq),
        "metrics_json": str(final_metrics_json),
        "loss_curve": str(final_loss_curve),
        "candidate_source": str(output_source),
        "active_learning": {
            "enabled": True,
            "version": ACTIVE_LEARNING_VERSION,
            "requested_chunks": (
                list(active_learning.manual_chunk_sizes)
                if active_learning.manual_chunk_sizes is not None
                else None
            ),
            "requested_chunk_size": active_learning.auto_chunk_size,
            "chunks": list(resolved_chunk_sizes),
            "patience": round_patience,
            "pool_cutoff_cp": active_learning.pool_cutoff_cp,
            "round_match_top_k": round_match_top_k,
            "eligible_pool_samples": eligible_pool_samples,
            "rounds": round_reports,
        },
        "labbook_logged": False,
    }
    return report


def run_candidate(
    root: Path,
    dataset: DatasetInput,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    nnue_bin: Path,
    match_bin: Path,
    book_path: Path,
    spec: CandidateSpec,
) -> dict[str, object]:
    experiment_root = experiment_root_for_candidate(root, spec)
    experiment_root.mkdir(parents=True, exist_ok=True)
    candidate_json_path = experiment_root / "candidate.json"
    existing = load_candidate_report(candidate_json_path)
    if existing is not None and existing.get("status") in {"complete", "failed"}:
        if spec.sweep_name is None and not existing.get("labbook_logged"):
            append_candidate_labbook_entry(existing)
            write_candidate_report(candidate_json_path, existing)
        return existing

    corpus = ensure_training_corpus(
        root=root,
        dataset=dataset,
        dataset_slug=canonical_dataset_slug(dataset),
        base_corpus=base_corpus,
        index=index,
        profile=spec.profile,
        tier=spec.tier,
        seed=spec.seed,
        max_abs_score=spec.max_abs_score,
    )
    started = time.monotonic()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failure_stage = "setup"
    try:
        if spec.active_learning is not None:
            failure_stage = "active_learning"
            report = run_active_learning_candidate(
                root=root,
                nnue_bin=nnue_bin,
                match_bin=match_bin,
                book_path=book_path,
                spec=spec,
                experiment_root=experiment_root,
                pool_corpus=corpus,
            )
        else:
            checkpoint_epochs = checkpoint_epochs_for_candidate(spec)
            print_startup(spec, corpus, experiment_root, book_path, checkpoint_epochs)
            output_source = candidate_source_path(experiment_root, spec.engine_template_path.stem, spec.candidate_id)
            model_json = experiment_root / "model.json"
            model_nnq = experiment_root / "model.nnq"
            metrics_json = experiment_root / "metrics.json"
            loss_curve = experiment_root / "loss_curves.svg"
            failure_stage = "training"
            ensure_trained_model(
                root=root,
                nnue_bin=nnue_bin,
                corpus=corpus,
                experiment_root=experiment_root,
                spec=spec,
                match_bin=match_bin,
                book_path=book_path,
                candidate_id=spec.candidate_id,
                checkpoint_epochs=checkpoint_epochs,
                enable_checkpoints=True,
            )

            model_report = load_model_report(model_json, metrics_json)

            failure_stage = "diagnostic_runtime_loss"
            diagnostic_report = run_runtime_loss(
                root,
                nnue_bin,
                model_nnq,
                corpus.diagnostic_val_path,
                spec.lambda_score,
            )

            failure_stage = "top_checkpoint_matches"
            top_checkpoint_matches = run_top_checkpoint_matches(
                root=root,
                experiment_root=experiment_root,
                model_report=model_report,
                engine_template_path=spec.engine_template_path,
                reference_source_path=(
                    spec.reference_source_path
                    if spec.reference_source_path is not None
                    else spec.engine_template_path
                ),
                candidate_id=spec.candidate_id,
                match_bin=match_bin,
                book_path=book_path,
                time_ms=spec.time_ms,
                game_count=spec.tier.checkpoint_match_games or 0,
                dense_feature_mask=spec.dense_feature_mask,
            )

            failure_stage = "source_bake"
            status(f"trainer: baking {model_nnq} into {output_source}")
            bake_runtime_source(spec.engine_template_path, model_nnq, output_source, spec.dense_feature_mask)

            final_match: MatchResult | None
            confirm_match: MatchResult | None
            if spec.tier.final_match_games > 0:
                failure_stage = "final_match"
                status(
                    "trainer: validating "
                    f"{output_source} against "
                    f"{spec.reference_source_path if spec.reference_source_path is not None else spec.engine_template_path} "
                    f"on {book_path} "
                    f"({spec.tier.final_match_games} games)"
                )
                final_match = run_match(
                    root=root,
                    match_bin=match_bin,
                    candidate_source=output_source,
                    reference_source=(
                        spec.reference_source_path
                        if spec.reference_source_path is not None
                        else spec.engine_template_path
                    ),
                    book_path=book_path,
                    time_ms=spec.time_ms,
                    game_count=spec.tier.final_match_games,
                )

                failure_stage = "confirm_match"
                confirm_match = maybe_confirm_match(
                    root=root,
                    match_bin=match_bin,
                    candidate_source=output_source,
                    reference_source=(
                        spec.reference_source_path
                        if spec.reference_source_path is not None
                        else spec.engine_template_path
                    ),
                    book_path=book_path,
                    time_ms=spec.time_ms,
                    final_match=final_match,
                )
            else:
                status("trainer: skipping final match (`final_match_games=0`)")
                final_match = None
                confirm_match = None

            report = {
                "status": "complete",
                "candidate_id": spec.candidate_id,
                "mode": spec.mode,
                "sweep": spec.sweep_name,
                "dataset": str(spec.dataset_path),
                "profile": spec.profile.name,
                "tier": spec.tier.name,
                "recipe": spec.recipe.name,
                "seed": spec.seed,
                "backend": spec.backend,
                "reference_source": str(
                    spec.reference_source_path
                    if spec.reference_source_path is not None
                    else spec.engine_template_path
                ),
                "feature_set": spec.feature_set_name,
                "architecture": spec.architecture_spec,
                "activation": spec.activation_name,
                "dense_feature_mask": list(spec.dense_feature_mask),
                "repeat_occurrence_weight": spec.repeat_occurrence_weight,
                "class_weighting": spec.class_weighting,
                "ema_decay": spec.ema_decay,
                "time_ms": spec.time_ms,
                "lambda_mix": {
                    "score": spec.lambda_score,
                    "result": spec.lambda_result,
                },
                "train_samples": spec.train_samples,
                "pool_train_samples": spec.pool_train_samples,
                "max_abs_score": spec.max_abs_score,
                "validation_samples": spec.validation_samples,
                "diagnostic_validation_samples": corpus.diagnostic_val_samples,
                "epochs": spec.epochs,
                "batch_size": spec.recipe.batch_size,
                "learning_rate": spec.recipe.learning_rate,
                "weight_decay": spec.recipe.weight_decay,
                "dropout": spec.recipe.dropout,
                "patience": patience_for_spec(spec),
                "runtime_loss_interval": runtime_loss_interval_for_spec(spec),
                "best_epoch": model_report.get("best_epoch"),
                "best_val_loss": float(model_report["best_val_loss"]),
                "best_quantized_val_loss": float(model_report["best_quantized_val_loss"]),
                "diagnostic_val_loss": float(diagnostic_report["loss"]),
                "diagnostic_report": diagnostic_report,
                "top_checkpoint_matches": top_checkpoint_matches,
                "final_match": serialize_match(final_match) if final_match is not None else None,
                "confirm_match": serialize_match(confirm_match) if confirm_match is not None else None,
                "beats_baseline": bool(confirm_match is not None and confirm_match.elo_lower > 0.0),
                "experiment_root": str(experiment_root),
                "corpus_dir": str(corpus.corpus_dir),
                "train_dataset": str(corpus.train_path),
                "val_dataset": str(corpus.val_path),
                "diagnostic_val_dataset": str(corpus.diagnostic_val_path),
                "manifest": str(corpus.manifest_path),
                "model_json": str(model_json),
                "model_nnq": str(model_nnq),
                "metrics_json": str(metrics_json),
                "loss_curve": str(loss_curve),
                "candidate_source": str(output_source),
                "active_learning": None,
                "labbook_logged": False,
            }
        report["started_at"] = started_at
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["wall_clock_seconds"] = time.monotonic() - started
        write_candidate_report(candidate_json_path, report)
        if spec.sweep_name is None:
            append_candidate_labbook_entry(report)
            write_candidate_report(candidate_json_path, report)
        print_final_result(
            spec=spec,
            corpus=corpus_paths_from_report(report),
            output_source_path=Path(str(report["candidate_source"])),
            model_json=Path(str(report["model_json"])),
            model_nnq=Path(str(report["model_nnq"])),
            loss_curve=Path(str(report["loss_curve"])),
            final_match=deserialize_match(
                None if report["final_match"] is None else dict(report["final_match"])
            ),
            confirm_match=deserialize_match(
                None if report["confirm_match"] is None else dict(report["confirm_match"])
            ),
        )
        return report
    except Exception as error:
        report = candidate_failure_report(
            spec=spec,
            experiment_root=experiment_root,
            corpus=corpus,
            stage=failure_stage,
            error=error,
            wall_clock_seconds=time.monotonic() - started,
        )
        write_candidate_report(candidate_json_path, report)
        if spec.sweep_name is None:
            append_candidate_labbook_entry(report)
            write_candidate_report(candidate_json_path, report)
        raise


def standalone_spec(
    args: CliArgs,
    dataset: DatasetInput,
    engine_template_path: Path,
    reference_source_path: Path | None = None,
) -> CandidateSpec:
    profile = PROFILE_SPECS[args.profile]
    tier = TIER_SPECS[args.tier]
    active_learning = (
        ActiveLearningSpec(
            chunk_sizes=args.active_learning_chunks,
            chunk_size=args.active_learning_chunk_size,
            pool_cutoff_cp=args.active_learning_pool_cutoff_cp,
            round_match_top_k=args.active_learning_match_top_k,
        )
        if args.active_learning_chunks is not None or args.active_learning_chunk_size is not None
        else None
    )
    recipe = RecipeSpec(
        name=DEFAULT_BASE_RECIPE,
        learning_rate=args.learning_rate if args.learning_rate is not None else DEFAULT_LEARNING_RATE,
        weight_decay=args.weight_decay if args.weight_decay is not None else DEFAULT_WEIGHT_DECAY,
        batch_size=args.batch_size,
        dropout=args.dropout if args.dropout is not None else DEFAULT_DROPOUT,
    )
    prepared_sample_count = (
        count_prepared_samples(dataset.prepared_path)
        if dataset.prepared_path.is_file()
        else args.validation_samples + tier.train_samples
    )
    default_full_train_samples = prepared_sample_count - args.validation_samples
    if default_full_train_samples <= 0:
        fail(
            f"{dataset.prepared_path} has no training samples left after reserving "
            f"{args.validation_samples} validation samples"
        )
    default_final_match_games = tier.final_match_games if active_learning is not None else 0
    if args.tier == "full":
        tier = TierSpec(
            name=tier.name,
            train_samples=(
                args.train_samples if args.train_samples is not None else default_full_train_samples
            ),
            fixed_validation_samples=args.validation_samples,
            diagnostic_validation_samples=(
                args.diagnostic_validation_samples
                if args.diagnostic_validation_samples is not None
                else DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES
            ),
            epochs=args.epochs if args.epochs is not None else tier.epochs,
            final_match_games=(
                args.final_match_games if args.final_match_games is not None else default_final_match_games
            ),
            checkpoint_interval_epochs=(
                args.checkpoint_interval if args.checkpoint_interval is not None else tier.checkpoint_interval_epochs
            ),
            checkpoint_match_games=(
                args.checkpoint_games if args.checkpoint_games is not None else tier.checkpoint_match_games
            ),
        )
    else:
        tier = TierSpec(
            name=tier.name,
            train_samples=args.train_samples if args.train_samples is not None else tier.train_samples,
            fixed_validation_samples=(
                args.validation_samples
                if args.validation_samples != DEFAULT_FIXED_VALIDATION_SAMPLES
                else tier.fixed_validation_samples
            ),
            diagnostic_validation_samples=(
                args.diagnostic_validation_samples
                if args.diagnostic_validation_samples is not None
                else DEFAULT_DIAGNOSTIC_VALIDATION_SAMPLES
            ),
            epochs=args.epochs if args.epochs is not None else tier.epochs,
            final_match_games=(
                args.final_match_games if args.final_match_games is not None else default_final_match_games
            ),
            checkpoint_interval_epochs=(
                args.checkpoint_interval if args.checkpoint_interval is not None else tier.checkpoint_interval_epochs
            ),
            checkpoint_match_games=(
                args.checkpoint_games if args.checkpoint_games is not None else tier.checkpoint_match_games
            ),
        )
    feature_set_name = args.feature_set if args.feature_set is not None else MODEL_INPUTS
    return CandidateSpec(
        mode="single",
        dataset_slug=canonical_dataset_slug(dataset),
        dataset_path=dataset.prepared_path,
        engine_template_path=engine_template_path,
        reference_source_path=reference_source_path,
        profile=profile,
        tier=tier,
        recipe=recipe,
        time_ms=args.time_ms,
        seed=args.seed,
        backend=resolve_backend(args),
        lambda_mix=args.lambda_mix,
        checkpoint_start_epoch=args.checkpoint_start,
        sweep_name=None,
        active_learning=active_learning,
        max_abs_score=args.max_abs_score,
        repeat_occurrence_weight=(
            (
                DEFAULT_REPEAT_OCCURRENCE_WEIGHT
                if args.repeat_occurrence_weight is None
                else None if args.repeat_occurrence_weight == "none" else args.repeat_occurrence_weight
            )
        ),
        class_weighting=(
            args.class_weighting if args.class_weighting is not None else DEFAULT_CLASS_WEIGHTING
        ),
        ema_decay=args.ema_decay if args.ema_decay is not None else DEFAULT_EMA_DECAY,
        feature_set_name=feature_set_name,
        architecture_spec=args.architecture if args.architecture is not None else ARCHITECTURE,
        activation_name=args.activation if args.activation is not None else "relu",
        dense_feature_mask=(
            args.dense_mask
            if args.dense_mask is not None
            else default_dense_feature_mask_for_feature_set(feature_set_name)
        ),
        runtime_loss_interval=(
            args.runtime_loss_interval
            if args.runtime_loss_interval is not None
            else (
                ACTIVE_LEARNING_RUNTIME_LOSS_INTERVAL
                if active_learning is not None
                else DEFAULT_RUNTIME_LOSS_INTERVAL
            )
        ),
        patience=args.patience,
    )


def run_candidate_batch(
    root: Path,
    dataset: DatasetInput,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    nnue_bin: Path,
    match_bin: Path,
    book_path: Path,
    specs: Sequence[CandidateSpec],
    sweep_dir: Path | None,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for spec in specs:
        status(f"trainer: candidate {spec.candidate_id}")
        try:
            report = run_candidate(
                root=root,
                dataset=dataset,
                base_corpus=base_corpus,
                index=index,
                nnue_bin=nnue_bin,
                match_bin=match_bin,
                book_path=book_path,
                spec=spec,
            )
        except Exception as error:
            status(f"trainer: candidate {spec.candidate_id} failed: {error}")
            candidate_json = experiment_root_for_candidate(root, spec) / "candidate.json"
            report = load_candidate_report(candidate_json)
            if report is None:
                raise
        reports.append(report)
        if sweep_dir is not None:
            update_leaderboard(sweep_dir)
    return reports


def probe_candidates(args: CliArgs, dataset: DatasetInput, engine_template_path: Path) -> list[CandidateSpec]:
    tier = TIER_SPECS["probe"]
    specs: list[CandidateSpec] = []
    for profile_name in ["random", "terminal_cap_only", "neutral_light", "neutral_medium", "neutral_hard"]:
        specs.append(
            CandidateSpec(
                mode="sweep",
                dataset_slug=canonical_dataset_slug(dataset),
                dataset_path=dataset.prepared_path,
                engine_template_path=engine_template_path,
                profile=PROFILE_SPECS[profile_name],
                tier=tier,
                recipe=RECIPE_SPECS["base"],
                time_ms=args.time_ms,
                seed=args.seed,
                backend=resolve_backend(args),
                lambda_mix=DEFAULT_LAMBDA_MIX,
                checkpoint_start_epoch=None,
                sweep_name="systematic_v1",
                active_learning=None,
            )
        )
    return specs


def screen_candidates(
    args: CliArgs,
    dataset: DatasetInput,
    engine_template_path: Path,
    promoted_profile: ProfileSpec,
) -> list[CandidateSpec]:
    tier = TIER_SPECS["screen"]
    recipe_names = ["base"] if promoted_profile.name != "random" else ["base", "low_lr", "high_lr", "low_wd"]
    seen: set[tuple[str, str]] = set()
    specs: list[CandidateSpec] = []
    for profile_name, recipe_name in [("random", "base")] + [(promoted_profile.name, name) for name in recipe_names]:
        key = (profile_name, recipe_name)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            CandidateSpec(
                mode="sweep",
                dataset_slug=canonical_dataset_slug(dataset),
                dataset_path=dataset.prepared_path,
                engine_template_path=engine_template_path,
                profile=PROFILE_SPECS[profile_name],
                tier=tier,
                recipe=RECIPE_SPECS[recipe_name],
                time_ms=args.time_ms,
                seed=args.seed,
                backend=resolve_backend(args),
                lambda_mix=DEFAULT_LAMBDA_MIX,
                checkpoint_start_epoch=None,
                sweep_name="systematic_v1",
                active_learning=None,
            )
        )
    return specs


def full_candidates(
    args: CliArgs,
    dataset: DatasetInput,
    engine_template_path: Path,
    promoted_reports: Sequence[dict[str, object]],
) -> list[CandidateSpec]:
    tier = TIER_SPECS["full"]
    specs: list[CandidateSpec] = []
    seen: set[tuple[str, str]] = set()
    seed_specs = [("random", "base")]
    for report in promoted_reports:
        recipe = RECIPE_SPECS.get(str(report["recipe"]))
        if recipe is None:
            fail(f"unknown promoted recipe {report['recipe']}")
        seed_specs.append((str(report["profile"]), recipe.name))
    for profile_name, recipe_name in seed_specs:
        key = (profile_name, recipe_name)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            CandidateSpec(
                mode="sweep",
                dataset_slug=canonical_dataset_slug(dataset),
                dataset_path=dataset.prepared_path,
                engine_template_path=engine_template_path,
                profile=PROFILE_SPECS[profile_name],
                tier=tier,
                recipe=RECIPE_SPECS[recipe_name],
                time_ms=args.time_ms,
                seed=args.seed,
                backend=resolve_backend(args),
                lambda_mix=DEFAULT_LAMBDA_MIX,
                checkpoint_start_epoch=None,
                sweep_name="systematic_v1",
                active_learning=None,
            )
        )
    return specs


def run_systematic_v1(
    root: Path,
    args: CliArgs,
    dataset: DatasetInput,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    nnue_bin: Path,
    match_bin: Path,
    book_path: Path,
    engine_template_path: Path,
) -> int:
    sweep_dir = sweep_root(root, canonical_dataset_slug(dataset), "systematic_v1", args.seed)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    probe_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=probe_candidates(args, dataset, engine_template_path),
        sweep_dir=sweep_dir,
    )
    promoted_probe, probe_reasons = choose_stage_candidates(probe_reports, 1)
    append_stage_summary(sweep_dir, "probe", probe_reports, promoted_probe, probe_reasons)
    update_leaderboard(sweep_dir)

    promoted_profile = PROFILE_SPECS[str(promoted_probe[0]["profile"])]
    screen_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=screen_candidates(args, dataset, engine_template_path, promoted_profile),
        sweep_dir=sweep_dir,
    )
    promoted_screen, screen_reasons = choose_stage_candidates(screen_reports, 2)
    append_stage_summary(sweep_dir, "screen", screen_reports, promoted_screen, screen_reasons)
    update_leaderboard(sweep_dir)

    full_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=full_candidates(args, dataset, engine_template_path, promoted_screen),
        sweep_dir=sweep_dir,
    )
    promoted_full, full_reasons = choose_stage_candidates(full_reports, min(2, len([r for r in full_reports if r["profile"] != "random"])))
    append_stage_summary(sweep_dir, "full", full_reports, promoted_full, full_reasons)
    update_leaderboard(sweep_dir)
    status(f"trainer: sweep leaderboard {leaderboard_md_path(sweep_dir)}")
    return 0


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note_file_path(sweep_dir: Path, note_name: str) -> Path:
    return sweep_dir / f"{slugify(note_name)}.json"


def append_once_note(
    sweep_dir: Path,
    note_name: str,
    title: str,
    lines: Sequence[str],
    payload: dict[str, object],
) -> None:
    path = note_file_path(sweep_dir, note_name)
    existing = load_candidate_report(path)
    if existing is not None and existing.get("labbook_logged"):
        return
    note_payload = dict(payload)
    note_payload["labbook_logged"] = False
    write_json(path, note_payload)
    append_labbook_entry(f"{now_utc()} - {title}", lines)
    note_payload["labbook_logged"] = True
    write_json(path, note_payload)


def finalize_stage_reports(sweep_dir: Path, reports: Sequence[dict[str, object]]) -> dict[str, object]:
    control = annotate_control_relative_fields(reports)
    for report in reports:
        if not report.get("labbook_logged"):
            append_candidate_labbook_entry(report)
            write_candidate_report(Path(str(report["experiment_root"])) / "candidate.json", report)
    update_leaderboard(sweep_dir)
    return control


def make_sweep_candidate(
    args: CliArgs,
    dataset: DatasetInput,
    engine_template_path: Path,
    profile_name: str,
    tier_name: str,
    recipe_name: str,
    sweep_name: str,
    seed: int,
) -> CandidateSpec:
    return CandidateSpec(
        mode="sweep",
        dataset_slug=canonical_dataset_slug(dataset),
        dataset_path=dataset.prepared_path,
        engine_template_path=engine_template_path,
        profile=PROFILE_SPECS[profile_name],
        tier=TIER_SPECS[tier_name],
        recipe=RECIPE_SPECS[recipe_name],
        time_ms=args.time_ms,
        seed=seed,
        backend=resolve_backend(args),
        lambda_mix=DEFAULT_LAMBDA_MIX,
        checkpoint_start_epoch=None,
        sweep_name=sweep_name,
        active_learning=None,
        max_abs_score=args.max_abs_score,
    )


def unique_specs(specs: Sequence[CandidateSpec]) -> list[CandidateSpec]:
    unique: list[CandidateSpec] = []
    seen: set[tuple[str, str, str, int, int | None]] = set()
    for spec in specs:
        key = (spec.tier.name, spec.profile.name, spec.recipe.name, spec.seed, spec.max_abs_score)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def config_key(profile: str, recipe: str) -> tuple[str, str]:
    return profile, recipe


def report_config_key(report: dict[str, object]) -> tuple[str, str]:
    return str(report["profile"]), str(report["recipe"])


def best_stage_reports(reports: Sequence[dict[str, object]], limit: int) -> list[dict[str, object]]:
    complete = completed_reports(reports)
    return sorted(complete, key=fast_lane_rank_key, reverse=True)[:limit]


def infer_recipe_from_candidate_id(candidate_id: str) -> str | None:
    for recipe_name in sorted(RECIPE_SPECS, key=len, reverse=True):
        if f"_{recipe_name}_seed" in candidate_id:
            return recipe_name
    return None


def parse_labbook_candidate_entries(path: Path) -> list[dict[str, object]]:
    import re

    if not path.is_file():
        return []
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    entries: list[dict[str, object]] = []
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        entry: dict[str, object] = {"title": title.strip()}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if line.startswith("- Candidate ID: `") and line.endswith("`"):
                entry["candidate_id"] = line[len("- Candidate ID: `") : -1]
            elif line.startswith("- Profile: `") and line.endswith("`"):
                entry["profile"] = line[len("- Profile: `") : -1]
            elif line.startswith("- Tier: `") and line.endswith("`"):
                entry["tier"] = line[len("- Tier: `") : -1]
            elif line.startswith("- Random-Val Loss: "):
                match = re.search(r"quantized=([0-9.]+)", line)
                if match is not None:
                    entry["best_quantized_val_loss"] = float(match.group(1))
            elif line.startswith("- Diagnostic-Val Loss: "):
                entry["diagnostic_val_loss"] = float(line.split(": ", 1)[1])
            elif line.startswith("- Match Result: "):
                match = re.search(r"elo=([+-]?[0-9.]+)", line)
                if match is not None:
                    entry["elo"] = float(match.group(1))
        candidate_id = entry.get("candidate_id")
        if candidate_id is None:
            continue
        if "profile" not in entry or "tier" not in entry:
            continue
        if "best_quantized_val_loss" not in entry or "diagnostic_val_loss" not in entry or "elo" not in entry:
            continue
        recipe_name = infer_recipe_from_candidate_id(str(candidate_id))
        if recipe_name is None:
            continue
        entry["recipe"] = recipe_name
        entries.append(entry)
    return entries


def choose_historical_anchor(entries: Sequence[dict[str, object]], profile: str, recipe: str) -> dict[str, object] | None:
    matching = [
        entry
        for entry in entries
        if entry.get("profile") == profile
        and entry.get("recipe") == recipe
        and entry.get("tier") != "micro"
    ]
    if not matching:
        return None
    matching.sort(
        key=lambda entry: (
            stage_order_value(str(entry["tier"])),
            str(entry["title"]),
        ),
        reverse=True,
    )
    return matching[0]


def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_ranks = np.array(average_ranks(x), dtype=np.float64)
    y_ranks = np.array(average_ranks(y), dtype=np.float64)
    if np.allclose(x_ranks, x_ranks[0]) or np.allclose(y_ranks, y_ranks[0]):
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def metric_text(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:+.3f}"


def calibration_metrics(
    fast_reports: Sequence[dict[str, object]],
    history_entries: Sequence[dict[str, object]],
) -> dict[str, object]:
    historical_by_config: dict[tuple[str, str], dict[str, object]] = {}
    for report in fast_reports:
        key = report_config_key(report)
        historical = choose_historical_anchor(history_entries, key[0], key[1])
        if historical is not None:
            historical_by_config[key] = historical
    aligned = [
        report
        for report in completed_reports(fast_reports)
        if report_config_key(report) in historical_by_config
    ]
    larger_elos = [float(historical_by_config[report_config_key(report)]["elo"]) for report in aligned]
    neg_qvals = [-float(report["best_quantized_val_loss"]) for report in aligned]
    neg_diags = [-float(report["diagnostic_val_loss"]) for report in aligned]
    fast_elos = [float(report["final_match"]["elo"]) for report in aligned]
    metrics = {
        "anchor_count": len(aligned),
        "qval_corr": spearman_rank_correlation(neg_qvals, larger_elos),
        "diag_corr": spearman_rank_correlation(neg_diags, larger_elos),
        "elo_corr": spearman_rank_correlation(fast_elos, larger_elos),
    }
    by_config = {report_config_key(report): report for report in completed_reports(fast_reports)}
    random_base = by_config.get(config_key("random", "base"))
    neutral_base = by_config.get(config_key("neutral_light", "base"))
    metrics["random_beats_neutral_base_on_qval"] = bool(
        random_base is not None
        and neutral_base is not None
        and float(random_base["best_quantized_val_loss"]) <= float(neutral_base["best_quantized_val_loss"])
    )
    metrics["random_beats_neutral_base_on_elo"] = bool(
        random_base is not None
        and neutral_base is not None
        and float(random_base["final_match"]["elo"]) >= float(neutral_base["final_match"]["elo"])
    )
    neutral_recipe_reports = [
        report
        for report in completed_reports(fast_reports)
        if report["profile"] == "neutral_light"
        and report["recipe"] in {"base", "low_lr", "high_lr", "low_wd"}
    ]
    neutral_recipe_reports.sort(key=fast_lane_rank_key, reverse=True)
    metrics["neutral_best_recipe"] = neutral_recipe_reports[0]["recipe"] if neutral_recipe_reports else None
    loser_keys = [
        config_key("terminal_cap_only", "base"),
        config_key("neutral_medium", "base"),
        config_key("neutral_hard", "base"),
    ]
    metrics["losers_still_lose"] = bool(
        random_base is not None
        and all(
            key not in by_config or fast_lane_rank_key(by_config[key]) <= fast_lane_rank_key(random_base)
            for key in loser_keys
        )
    )
    return metrics


def choose_fast_tier_from_calibration(metrics: dict[str, object]) -> tuple[str, str, list[str]]:
    qval_corr = metrics.get("qval_corr")
    diag_corr = metrics.get("diag_corr")
    elo_corr = metrics.get("elo_corr")
    notes: list[str] = []
    good_loss_signal = (
        (qval_corr is not None and qval_corr >= 0.45)
        or (diag_corr is not None and diag_corr >= 0.45)
    )
    preserves_key_order = bool(metrics.get("random_beats_neutral_base_on_qval")) and bool(
        metrics.get("losers_still_lose")
    )
    if good_loss_signal and preserves_key_order:
        metric = "qval" if (qval_corr or -math.inf) >= (diag_corr or -math.inf) else "diag"
        notes.append("micro preserved the loss-signal ordering well enough to use as the discovery lane")
        if elo_corr is None or elo_corr < 0.25:
            notes.append("micro Elo is noisy at 120 games and should stay a guardrail rather than the main selector")
        return "micro", metric, notes
    notes.append("micro did not preserve the larger-run ordering cleanly enough, so the next pass should fall back to probe")
    return "probe", "qval", notes


def relative_summary_lines(reports: Sequence[dict[str, object]], limit: int) -> list[str]:
    lines: list[str] = []
    for report in best_stage_reports(reports, limit):
        lines.append(
            f"- `{report['candidate_id']}` qval={float(report['best_quantized_val_loss']):.6f} "
            f"diag={float(report['diagnostic_val_loss']):.6f} "
            f"elo={float(report['final_match']['elo']):+.2f} "
            f"delta_q={float(report.get('delta_qval_vs_control', 0.0)):+.6f} "
            f"delta_diag={float(report.get('delta_diag_vs_control', 0.0)):+.6f} "
            f"delta_elo={float(report.get('delta_elo_vs_control', 0.0)):+.2f} "
            f"outcome={report.get('control_outcome', 'n/a')}"
        )
    return lines


def profile_diagnostics_summary_lines(reports: Sequence[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for report in completed_reports(reports):
        if not str(report["profile"]).endswith("_v1"):
            continue
        profile_json = Path(str(report["corpus_dir"])) / "profile.json"
        if not profile_json.is_file():
            continue
        payload = load_json_file(profile_json)
        diagnostics = payload.get("train_selection_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        lines.append(
            f"- `{report['profile']}` terminal={float(diagnostics['terminal_percentage']):.2f}% "
            f"openings={int(diagnostics['unique_openings'])} "
            f"opening_median={float(diagnostics['median_samples_per_opening']):.2f} "
            f"opening_p95={float(diagnostics['p95_samples_per_opening']):.2f} "
            f"opening_max={int(diagnostics['max_samples_per_opening'])} "
            f"disjoint={diagnostics['disjoint_train_diagnostic']}"
        )
    return lines


def select_best_profiles(
    profile_reports: Sequence[dict[str, object]],
) -> list[str]:
    complete = sorted(completed_reports(profile_reports), key=fast_lane_rank_key, reverse=True)
    profiles: list[str] = []
    for report in complete:
        profile = str(report["profile"])
        if profile in profiles:
            continue
        profiles.append(profile)
        if len(profiles) >= 2:
            break
    if "random" not in profiles:
        profiles.insert(0, "random")
    return profiles[:2]


def select_finalists(
    reports: Sequence[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    finalists = []
    for report in sorted(completed_reports(reports), key=fast_lane_rank_key, reverse=True):
        finalists.append(report)
        if len(finalists) >= count:
            break
    return finalists


def run_subset_lab_v1(
    root: Path,
    args: CliArgs,
    dataset: DatasetInput,
    base_corpus: BaseCorpus,
    index: SelectionIndex,
    nnue_bin: Path,
    match_bin: Path,
    book_path: Path,
    engine_template_path: Path,
) -> int:
    sweep_dir = sweep_root(root, canonical_dataset_slug(dataset), SUBSET_SWEEP_NAME, args.seed)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    append_once_note(
        sweep_dir,
        "harness_correction_note",
        "harness correction note",
        [
            "- Random control now participates in ranking and is allowed to remain the stage winner.",
            "- No non-random promotion is now a valid outcome; the harness prefers no-promotion over force-promoting a loser.",
            "- Profile comparison and recipe comparison are now separate stages, so random recipes can be swept when random remains best.",
            f"- Selection index schema bumped to `selection_index_v{SELECTION_INDEX_VERSION}.npz` and corpus layout to `{SELECTION_CORPUS_VERSION}` to avoid stale cache reuse.",
        ],
        {
            "kind": "harness_correction_note",
            "selection_index_version": SELECTION_INDEX_VERSION,
            "selection_corpus_version": SELECTION_CORPUS_VERSION,
        },
    )

    calibration_specs = unique_specs(
        [
            make_sweep_candidate(args, dataset, engine_template_path, "random", "micro", "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_light", "micro", "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_light", "micro", "low_lr", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_light", "micro", "high_lr", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_light", "micro", "low_wd", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "terminal_cap_only", "micro", "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_medium", "micro", "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "neutral_hard", "micro", "base", SUBSET_SWEEP_NAME, args.seed),
        ]
    )
    calibration_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=calibration_specs,
        sweep_dir=sweep_dir,
    )
    finalize_stage_reports(sweep_dir, calibration_reports)
    history_entries = parse_labbook_candidate_entries(LABBOOK_PATH)
    calibration = calibration_metrics(calibration_reports, history_entries)
    fast_tier_name, trusted_metric, calibration_notes = choose_fast_tier_from_calibration(calibration)
    append_once_note(
        sweep_dir,
        "fast_tier_calibration_summary",
        "fast-tier calibration summary",
        [
            f"- Anchor Count: {calibration['anchor_count']}",
            f"- Rank Correlation (lower q-val better): {metric_text(calibration['qval_corr'])}",
            f"- Rank Correlation (lower diag better): {metric_text(calibration['diag_corr'])}",
            f"- Rank Correlation (fast-tier Elo): {metric_text(calibration['elo_corr'])}",
            f"- random > neutral_light/base preserved on q-val: {calibration['random_beats_neutral_base_on_qval']}",
            f"- random > neutral_light/base preserved on Elo: {calibration['random_beats_neutral_base_on_elo']}",
            f"- neutral_light best recipe on fast tier: {calibration['neutral_best_recipe']}",
            f"- terminal_cap_only / neutral_medium / neutral_hard still look like losers: {calibration['losers_still_lose']}",
            f"- Recommended Fast Tier: `{fast_tier_name}`",
            f"- Recommended Main Metric: `{trusted_metric}`",
            *[f"- Note: {note}" for note in calibration_notes],
        ],
        {
            "kind": "fast_tier_calibration_summary",
            "metrics": calibration,
            "recommended_tier": fast_tier_name,
            "recommended_metric": trusted_metric,
        },
    )

    random_recipe_specs = unique_specs(
        [
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "low_lr", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "high_lr", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "low_wd", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "very_low_lr", SUBSET_SWEEP_NAME, args.seed),
        ]
    )
    random_recipe_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=random_recipe_specs,
        sweep_dir=sweep_dir,
    )
    random_control = finalize_stage_reports(sweep_dir, random_recipe_reports)
    append_once_note(
        sweep_dir,
        "random_control_recipe_sweep_summary",
        "random-control recipe sweep summary",
        [
            f"- Control: `{random_control['candidate_id']}`",
            f"- Trusted Fast Tier: `{fast_tier_name}` with primary signal `{trusted_metric}`",
            *relative_summary_lines(random_recipe_reports, 5),
        ],
        {
            "kind": "random_control_recipe_sweep_summary",
            "tier": fast_tier_name,
            "metric": trusted_metric,
            "reports": [report["candidate_id"] for report in random_recipe_reports],
        },
    )

    profile_specs = unique_specs(
        [
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "score_bucket_flat_v1", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "ply_bucket_flat_v1", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "result_balanced_v1", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "opening_diverse_cap_v1", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
            make_sweep_candidate(args, dataset, engine_template_path, "score_ply_hybrid_v1", fast_tier_name, "base", SUBSET_SWEEP_NAME, args.seed),
        ]
    )
    profile_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=profile_specs,
        sweep_dir=sweep_dir,
    )
    profile_control = finalize_stage_reports(sweep_dir, profile_reports)
    append_once_note(
        sweep_dir,
        "corpus_diagnostics_note",
        "corpus diagnostics note",
        profile_diagnostics_summary_lines(profile_reports),
        {
            "kind": "corpus_diagnostics_note",
            "profiles": [report["candidate_id"] for report in profile_reports if str(report["profile"]).endswith("_v1")],
        },
    )
    append_once_note(
        sweep_dir,
        "new_profile_sweep_summary",
        "new profile sweep summary",
        [
            f"- Control: `{profile_control['candidate_id']}`",
            f"- Trusted Fast Tier: `{fast_tier_name}` with primary signal `{trusted_metric}`",
            *relative_summary_lines(profile_reports, 6),
            (
                "- Note: No challenger beat the random control."
                if all(
                    report["candidate_id"] == profile_control["candidate_id"] or not candidate_is_competitive(report)
                    for report in completed_reports(profile_reports)
                )
                else "- Note: At least one challenger stayed competitive with random."
            ),
        ],
        {
            "kind": "new_profile_sweep_summary",
            "tier": fast_tier_name,
            "metric": trusted_metric,
            "reports": [report["candidate_id"] for report in profile_reports],
        },
    )

    finalist_profiles = select_best_profiles(profile_reports)
    interaction_specs: list[CandidateSpec] = []
    for profile_name in finalist_profiles:
        for recipe_name in ["base", "low_lr", "high_lr", "low_wd"]:
            interaction_specs.append(
                make_sweep_candidate(
                    args,
                    dataset,
                    engine_template_path,
                    profile_name,
                    fast_tier_name,
                    recipe_name,
                    SUBSET_SWEEP_NAME,
                    args.seed,
                )
            )
    interaction_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=unique_specs(interaction_specs),
        sweep_dir=sweep_dir,
    )
    interaction_control = finalize_stage_reports(sweep_dir, interaction_reports)
    finalists = select_finalists(interaction_reports, 2)
    append_once_note(
        sweep_dir,
        "interaction_check_summary",
        "interaction check summary",
        [
            f"- Control: `{interaction_control['candidate_id']}`",
            f"- Finalist Profiles: {', '.join(f'`{profile}`' for profile in finalist_profiles)}",
            *relative_summary_lines(interaction_reports, min(8, len(completed_reports(interaction_reports)))),
        ],
        {
            "kind": "interaction_check_summary",
            "finalist_profiles": finalist_profiles,
            "finalists": [report["candidate_id"] for report in finalists],
        },
    )

    seed2_specs = unique_specs(
        [
            make_sweep_candidate(args, dataset, engine_template_path, "random", fast_tier_name, "base", SUBSET_SWEEP_NAME, 2),
            *[
                make_sweep_candidate(
                    args,
                    dataset,
                    engine_template_path,
                    str(report["profile"]),
                    fast_tier_name,
                    str(report["recipe"]),
                    SUBSET_SWEEP_NAME,
                    2,
                )
                for report in finalists
            ],
        ]
    )
    seed2_reports = run_candidate_batch(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        specs=seed2_specs,
        sweep_dir=sweep_dir,
    )
    seed2_control = finalize_stage_reports(sweep_dir, seed2_reports)
    append_once_note(
        sweep_dir,
        "finalists_seed2_replication_summary",
        "finalists / seed2 replication summary",
        [
            f"- Control: `{seed2_control['candidate_id']}`",
            *relative_summary_lines(seed2_reports, len(completed_reports(seed2_reports))),
        ],
        {
            "kind": "finalists_seed2_replication_summary",
            "reports": [report["candidate_id"] for report in seed2_reports],
        },
    )

    seed2_lookup = {report_config_key(report): report for report in completed_reports(seed2_reports)}
    next_run_candidates = []
    for report in finalists:
        seed2_match = seed2_lookup.get(report_config_key(report))
        if seed2_match is not None:
            next_run_candidates.append(seed2_match)
    append_once_note(
        sweep_dir,
        "next_iteration_recommendation",
        "next iteration recommendation",
        [
            f"- 1. Trusted fast tier next: `{fast_tier_name}`",
            f"- 2. Optimize on `{trusted_metric}` first, with fast-tier Elo only as a guardrail.",
            "- 3. Next larger run configs: "
            + (
                ", ".join(
                    f"`{report['profile']}/{report['recipe']}`"
                    for report in next_run_candidates[:2]
                )
                if next_run_candidates
                else "`random/base`"
            ),
            "- 4. Dead ideas right now: `terminal_cap_only/base`, `neutral_medium/base`, `neutral_hard/base` unless later evidence contradicts today’s calibration.",
            "- 5. Harness caveats remain: micro Elo is still noisy at low game counts, calibration is anchored to mixed historical probe/screen/full results, and any new profile semantics should keep bumping corpus/index versions.",
        ],
        {
            "kind": "next_iteration_recommendation",
            "fast_tier": fast_tier_name,
            "metric": trusted_metric,
            "next_candidates": [report["candidate_id"] for report in next_run_candidates[:2]],
        },
    )
    status(f"trainer: sweep leaderboard {leaderboard_md_path(sweep_dir)}")
    return 0


def run_self_tests() -> int:
    class TrainerSelfTests(unittest.TestCase):
        def setUp(self) -> None:
            self.tempdir = Path(tempfile.mkdtemp(prefix="trainer-selftest-"))

        def tearDown(self) -> None:
            shutil.rmtree(self.tempdir, ignore_errors=True)

        def sample(self, score: float, result_bucket: int) -> PreparedSample:
            return PreparedSample(
                black_bits=1,
                white_bits=2,
                side_to_move="b",
                position_key=int(abs(score) * 10) + result_bucket + 100,
                mean_score=score,
                mean_clipped_score=score,
                mean_result=float(result_bucket),
                mean_ply=12.0,
                effective_game_turns_played=12.0,
                occurrence_count=1,
                sample_weight=1.0,
                win_count=1 if result_bucket == 1 else 0,
                draw_count=1 if result_bucket == 0 else 0,
                loss_count=1 if result_bucket == -1 else 0,
                result_bucket=result_bucket,
                mean_completed_depth=10.0,
                mean_no_progress_plies=2.0,
                ejection_rate=0.0,
                recorded_mean_score=score,
                label_source=None,
                label_budget_ms=None,
                label_depth=None,
                source_dataset=None,
            )

        def fixture_path(self) -> Path:
            return self.tempdir / "fixture.abapack"

        def write_teacher_targets(self, path: Path, entries: Sequence[tuple[int, float]]) -> Path:
            with path.open("wb") as handle:
                handle.write(TEACHER_TARGET_MAGIC)
                write_u16(handle, TEACHER_TARGET_VERSION)
                write_u64(handle, len(entries))
                for position_key, score in entries:
                    write_u64(handle, position_key)
                    write_f32(handle, score)
            return path

        def make_candidate_spec(self) -> CandidateSpec:
            return CandidateSpec(
                mode="single",
                dataset_slug="fixture",
                dataset_path=self.fixture_path(),
                engine_template_path=self.tempdir / "engine.rs",
                profile=PROFILE_SPECS["random"],
                tier=TIER_SPECS["micro"],
                recipe=RECIPE_SPECS["base"],
                time_ms=DEFAULT_TIME_MS,
                seed=1,
                backend=DEFAULT_BACKEND,
                lambda_mix=DEFAULT_LAMBDA_MIX,
                checkpoint_start_epoch=None,
                sweep_name=None,
                active_learning=None,
            )

        def make_corpus_paths(self) -> CorpusPaths:
            corpus_dir = self.tempdir / "corpus"
            return CorpusPaths(
                corpus_dir=corpus_dir,
                train_path=corpus_dir / "train.abapack",
                val_path=corpus_dir / "val.abapack",
                diagnostic_val_path=corpus_dir / "diagnostic_val.abapack",
                manifest_path=corpus_dir / "manifest.json",
                profile_path=corpus_dir / "profile.json",
                dataset_cache_dir=corpus_dir / "dataset_cache",
                train_samples=123,
                val_samples=45,
                diagnostic_val_samples=45,
            )

        def make_index(
            self,
            abs_scores: np.ndarray,
            *,
            ply_values: np.ndarray | None = None,
            result_values: np.ndarray | None = None,
            opening_values: np.ndarray | None = None,
        ) -> SelectionIndex:
            count = int(abs_scores.shape[0])
            if ply_values is None:
                ply_values = np.arange(count, dtype=np.float32)
            if result_values is None:
                result_values = np.zeros(count, dtype=np.int8)
            if opening_values is None:
                opening_values = np.arange(count, dtype=np.uint64)
            return SelectionIndex(
                key_hashes=np.arange(count, dtype=np.uint64),
                abs_scores=abs_scores.astype(np.float32, copy=False),
                terminal_mask=(abs_scores >= TERMINAL_SCORE_THRESHOLD),
                score_buckets=np.array(
                    [score_bucket_for_abs_score(float(value)) for value in abs_scores.tolist()],
                    dtype=np.int32,
                ),
                ply_buckets=np.array(
                    [ply_bucket_for_turns(float(value)) for value in ply_values.tolist()],
                    dtype=np.int32,
                ),
                result_buckets=result_values.astype(np.int8, copy=False),
                opening_hashes=opening_values.astype(np.uint64, copy=False),
            )

        def write_fixture(self) -> Path:
            writer = PreparedWriter(self.fixture_path())
            chain = PreparedChain(
                run_file="fixture.json",
                game_index=1,
                opening_name="sample",
                opening_position="aba-v1;stm=b;black=A1;white=I5",
                opening_hash=1234,
                split="train",
                samples=[
                    self.sample(-6000.0, -1),
                    self.sample(-50.0, -1),
                    self.sample(0.0, 0),
                    self.sample(300.0, 1),
                    self.sample(7000.0, 1),
                ],
            )
            writer.write_chain(chain)
            writer.close()
            return self.fixture_path()

        def test_roundtrip_prepared_abapack(self) -> None:
            path = self.write_fixture()
            chains = list(iter_prepared_chains(path))
            self.assertEqual(len(chains), 1)
            self.assertEqual(len(chains[0].samples), 5)
            self.assertEqual(chains[0].samples[0].mean_clipped_score, -6000.0)

        def test_splitmix_and_priority_determinism(self) -> None:
            key_hashes = np.array([1, 2, 3, 4], dtype=np.uint64)
            weights = np.ones(4, dtype=np.float64)
            first = priority_array(key_hashes, 1, "diag:random", weights)
            second = priority_array(key_hashes, 1, "diag:random", weights)
            self.assertTrue(np.allclose(first, second))

        def test_parse_active_learning_chunks_supports_variable_sizes(self) -> None:
            self.assertEqual(parse_active_learning_chunks("250k,500k,1m"), (250_000, 500_000, 1_000_000))

        def test_active_learning_round_specs_accumulate_selected_samples(self) -> None:
            specs = active_learning_round_specs((100_000, 250_000, 650_000))
            self.assertEqual([spec.chunk_size for spec in specs], [100_000, 250_000, 650_000])
            self.assertEqual([spec.selected_samples for spec in specs], [100_000, 350_000, 1_000_000])

        def test_auto_active_learning_chunk_size_fills_remaining_pool(self) -> None:
            active = ActiveLearningSpec(chunk_size=250_000)
            self.assertEqual(
                resolve_active_learning_chunk_sizes(active, 920_000),
                (250_000, 250_000, 250_000, 170_000),
            )

        def test_build_train_command_allows_patience_override(self) -> None:
            command = build_train_command(
                Path("/tmp/nnue"),
                self.make_corpus_paths(),
                self.tempdir / "experiment",
                self.make_candidate_spec(),
                enable_checkpoints=False,
                patience=ACTIVE_LEARNING_PATIENCE,
            )
            index = command.index("--patience")
            self.assertEqual(command[index + 1], str(ACTIVE_LEARNING_PATIENCE))

        def test_active_learning_uses_runtime_loss_every_epoch(self) -> None:
            spec = replace(
                self.make_candidate_spec(),
                active_learning=ActiveLearningSpec(chunk_size=100_000),
            )
            command = build_train_command(
                Path("/tmp/nnue"),
                self.make_corpus_paths(),
                self.tempdir / "experiment",
                spec,
                enable_checkpoints=False,
            )
            index = command.index("--runtime-loss-interval")
            self.assertEqual(command[index + 1], "1")

        def test_build_train_command_omits_selection_knobs_for_full_recipe(self) -> None:
            command = build_train_command(
                Path("/tmp/nnue"),
                self.make_corpus_paths(),
                self.tempdir / "experiment",
                self.make_candidate_spec(),
                enable_checkpoints=False,
            )
            for flag in (
                "--selection-profile",
                "--qat-selection-profile",
                "--teacher-hard-threshold-cp",
                "--selection-thin-mod",
                "--late-limit-weight",
                "--drawish-weight",
                "--max-abs-clipped-score",
            ):
                self.assertNotIn(flag, command)

        def test_standalone_spec_allows_final_match_games_override(self) -> None:
            args = parse_args(
                [
                    "fixture.abapack",
                    "engine.rs",
                    "--tier",
                    "full",
                    "--final-match-games",
                    "1000",
                ]
            )
            dataset = DatasetInput(
                source_path=self.fixture_path(),
                prepared_path=self.fixture_path(),
                dataset_stem="fixture",
                dataset_slug="fixture",
                sibling_unique_shard=False,
            )
            spec = standalone_spec(args, dataset, self.tempdir / "engine.rs")
            self.assertEqual(spec.tier.final_match_games, 1000)

        def test_standalone_spec_skips_final_match_by_default(self) -> None:
            args = parse_args(
                [
                    "fixture.abapack",
                    "engine.rs",
                    "--tier",
                    "full",
                ]
            )
            dataset = DatasetInput(
                source_path=self.fixture_path(),
                prepared_path=self.fixture_path(),
                dataset_stem="fixture",
                dataset_slug="fixture",
                sibling_unique_shard=False,
            )
            spec = standalone_spec(args, dataset, self.tempdir / "engine.rs")
            self.assertEqual(spec.tier.final_match_games, 0)
            self.assertEqual(spec.tier.checkpoint_match_games, DEFAULT_CHECKPOINT_GAMES)

        def test_active_learning_keeps_default_final_match_games(self) -> None:
            args = parse_args(
                [
                    "fixture.abapack",
                    "engine.rs",
                    "--tier",
                    "full",
                    "--active-learning-chunk-size",
                    "100000",
                ]
            )
            dataset = DatasetInput(
                source_path=self.fixture_path(),
                prepared_path=self.fixture_path(),
                dataset_stem="fixture",
                dataset_slug="fixture",
                sibling_unique_shard=False,
            )
            spec = standalone_spec(args, dataset, self.tempdir / "engine.rs")
            self.assertIsNotNone(spec.active_learning)
            self.assertEqual(spec.tier.final_match_games, TIER_SPECS["full"].final_match_games)
            self.assertEqual(spec.tier.checkpoint_match_games, DEFAULT_CHECKPOINT_GAMES)

        def test_exact_selection_counts_and_disjointness(self) -> None:
            abs_scores = np.linspace(0, 8000, 100, dtype=np.float32)
            index = self.make_index(abs_scores)
            eligible = np.ones(100, dtype=bool)
            diag = select_profiled_indices(index, eligible, 10, 1, "diag:neutral_medium", PROFILE_SPECS["neutral_medium"])
            eligible[diag] = False
            train = select_profiled_indices(index, eligible, 50, 1, "train:neutral_medium:probe", PROFILE_SPECS["neutral_medium"])
            self.assertEqual(diag.shape[0], 10)
            self.assertEqual(train.shape[0], 50)
            self.assertEqual(len(set(diag.tolist()).intersection(set(train.tolist()))), 0)

        def test_random_diag_repeatable(self) -> None:
            abs_scores = np.linspace(0, 4000, 50, dtype=np.float32)
            index = self.make_index(abs_scores)
            eligible = np.ones(50, dtype=bool)
            first = select_profiled_indices(index, eligible, 10, 7, "diag:random", PROFILE_SPECS["random"])
            second = select_profiled_indices(index, eligible, 10, 7, "diag:random", PROFILE_SPECS["random"])
            self.assertEqual(first.tolist(), second.tolist())

        def test_terminal_cap_is_respected(self) -> None:
            abs_scores = np.concatenate(
                [
                    np.full(950, 100.0, dtype=np.float32),
                    np.full(50, 6000.0, dtype=np.float32),
                ]
            )
            index = self.make_index(abs_scores)
            eligible = np.ones(1000, dtype=bool)
            selected = select_profiled_indices(
                index,
                eligible,
                200,
                1,
                "train:terminal_cap_only:probe",
                PROFILE_SPECS["terminal_cap_only"],
            )
            terminal_count = int(index.terminal_mask[selected].sum())
            self.assertLessEqual(terminal_count, 10)

        def test_terminal_backfill_when_terminal_pool_is_small(self) -> None:
            abs_scores = np.concatenate(
                [
                    np.full(96, 100.0, dtype=np.float32),
                    np.full(4, 7000.0, dtype=np.float32),
                ]
            )
            index = self.make_index(abs_scores)
            eligible = np.ones(100, dtype=bool)
            selected = select_profiled_indices(
                index,
                eligible,
                50,
                1,
                "train:neutral_hard:probe",
                PROFILE_SPECS["neutral_hard"],
            )
            self.assertEqual(selected.shape[0], 50)
            self.assertEqual(int(index.terminal_mask[selected].sum()), 1)

        def test_bucket_flat_selection_is_deterministic(self) -> None:
            abs_scores = np.linspace(0, 5000, 120, dtype=np.float32)
            ply_values = np.linspace(0, 120, 120, dtype=np.float32)
            index = self.make_index(abs_scores, ply_values=ply_values)
            eligible = np.ones(120, dtype=bool)
            first = select_profiled_indices(
                index,
                eligible,
                48,
                11,
                "train:score_bucket_flat_v1:micro",
                PROFILE_SPECS["score_bucket_flat_v1"],
            )
            second = select_profiled_indices(
                index,
                eligible,
                48,
                11,
                "train:score_bucket_flat_v1:micro",
                PROFILE_SPECS["score_bucket_flat_v1"],
            )
            self.assertEqual(first.tolist(), second.tolist())

        def test_bucket_flat_selection_exact_counts(self) -> None:
            abs_scores = np.concatenate(
                [
                    np.full(20, 10.0, dtype=np.float32),
                    np.full(20, 600.0, dtype=np.float32),
                    np.full(20, 1100.0, dtype=np.float32),
                    np.full(20, 1600.0, dtype=np.float32),
                ]
            )
            index = self.make_index(abs_scores)
            selected = select_profiled_indices(
                index,
                np.ones(80, dtype=bool),
                32,
                3,
                "train:score_bucket_flat_v1:micro",
                PROFILE_SPECS["score_bucket_flat_v1"],
            )
            _, counts = np.unique(index.score_buckets[selected], return_counts=True)
            self.assertEqual(selected.shape[0], 32)
            self.assertEqual(sorted(counts.tolist()), [8, 8, 8, 8])

        def test_opening_cap_selection_keeps_exact_count_and_disjointness(self) -> None:
            abs_scores = np.linspace(0, 4000, 120, dtype=np.float32)
            openings = np.repeat(np.arange(12, dtype=np.uint64), 10)
            index = self.make_index(abs_scores, opening_values=openings)
            eligible = np.ones(120, dtype=bool)
            diag = select_profiled_indices(
                index,
                eligible,
                24,
                5,
                "diag:opening_diverse_cap_v1",
                PROFILE_SPECS["opening_diverse_cap_v1"],
            )
            eligible[diag] = False
            train = select_profiled_indices(
                index,
                eligible,
                60,
                5,
                "train:opening_diverse_cap_v1:micro",
                PROFILE_SPECS["opening_diverse_cap_v1"],
            )
            self.assertEqual(diag.shape[0], 24)
            self.assertEqual(train.shape[0], 60)
            self.assertEqual(len(set(diag.tolist()).intersection(set(train.tolist()))), 0)

        def test_control_can_remain_winner(self) -> None:
            reports = [
                {
                    "status": "complete",
                    "candidate_id": "random_base",
                    "profile": "random",
                    "recipe": "base",
                    "best_quantized_val_loss": 0.0035,
                    "diagnostic_val_loss": 0.0060,
                    "final_match": {"elo": -100.0, "elo_lower": -120.0, "elo_upper": -80.0},
                    "experiment_root": str(self.tempdir / "random_base"),
                },
                {
                    "status": "complete",
                    "candidate_id": "neutral_base",
                    "profile": "neutral_light",
                    "recipe": "base",
                    "best_quantized_val_loss": 0.0038,
                    "diagnostic_val_loss": 0.0064,
                    "final_match": {"elo": -150.0, "elo_lower": -170.0, "elo_upper": -130.0},
                    "experiment_root": str(self.tempdir / "neutral_base"),
                },
            ]
            for report in reports:
                path = Path(str(report["experiment_root"]))
                path.mkdir(parents=True, exist_ok=True)
                write_json(path / "candidate.json", report)
            chosen, _ = choose_stage_candidates(reports, 1)
            self.assertEqual([report["candidate_id"] for report in chosen], ["random_base"])

        def test_no_promotion_path_is_valid(self) -> None:
            reports = [
                {
                    "status": "complete",
                    "candidate_id": "random_base",
                    "profile": "random",
                    "recipe": "base",
                    "best_quantized_val_loss": 0.0035,
                    "diagnostic_val_loss": 0.0060,
                    "final_match": {"elo": -100.0, "elo_lower": -120.0, "elo_upper": -80.0},
                    "experiment_root": str(self.tempdir / "random_base_2"),
                },
                {
                    "status": "complete",
                    "candidate_id": "challenger_a",
                    "profile": "score_bucket_flat_v1",
                    "recipe": "base",
                    "best_quantized_val_loss": 0.0036,
                    "diagnostic_val_loss": 0.0061,
                    "final_match": {"elo": -145.0, "elo_lower": -165.0, "elo_upper": -125.0},
                    "experiment_root": str(self.tempdir / "challenger_a"),
                },
                {
                    "status": "complete",
                    "candidate_id": "challenger_b",
                    "profile": "ply_bucket_flat_v1",
                    "recipe": "base",
                    "best_quantized_val_loss": 0.0037,
                    "diagnostic_val_loss": 0.0062,
                    "final_match": {"elo": -160.0, "elo_lower": -180.0, "elo_upper": -140.0},
                    "experiment_root": str(self.tempdir / "challenger_b"),
                },
            ]
            for report in reports:
                path = Path(str(report["experiment_root"]))
                path.mkdir(parents=True, exist_ok=True)
                write_json(path / "candidate.json", report)
            chosen, reasons = choose_stage_candidates(reports, 2)
            self.assertEqual([report["candidate_id"] for report in chosen], ["random_base"])
            self.assertTrue(any("control remains the winner" in reason for reason in reasons))

        def test_write_selected_corpus_keeps_exact_target(self) -> None:
            path = self.write_fixture()
            selected = np.array([1, 3], dtype=np.uint32)
            summary = write_selected_corpus(path, self.tempdir / "subset.abapack", selected, "train")
            self.assertEqual(summary.samples, 2)
            subset = list(iter_prepared_chains(self.tempdir / "subset.abapack"))
            self.assertEqual(len(subset[0].samples), 2)

        def test_active_learning_random_selection_is_deterministic(self) -> None:
            eligible = np.array([True, True, False, True, True, True], dtype=bool)
            selected = np.array([False, True, False, False, False, False], dtype=bool)
            first = select_active_learning_random_ids(eligible, selected, 2, 7, "active:test")
            second = select_active_learning_random_ids(eligible, selected, 2, 7, "active:test")
            self.assertEqual(first.tolist(), second.tolist())
            self.assertTrue(all(eligible[index] and not selected[index] for index in first.tolist()))

        def test_active_learning_disagreement_selection_prefers_largest_gaps(self) -> None:
            path = self.write_fixture()
            chains = list(iter_prepared_chains(path))
            entries = []
            teacher_scores = [-6000.0, 100.0, 20.0, -500.0, 6800.0]
            for chain in chains:
                for sample, teacher_score in zip(chain.samples, teacher_scores):
                    entries.append((sample.position_key, teacher_score))
            teacher_path = self.write_teacher_targets(self.tempdir / "fixture.teacher.bin", entries)
            eligible = np.ones(5, dtype=bool)
            selected = np.array([True, False, False, False, False], dtype=bool)
            chosen, meta = select_top_disagreement_ids_from_teacher_targets(
                pool_path=path,
                teacher_path=teacher_path,
                eligible_mask=eligible,
                selected_mask=selected,
                count=2,
            )
            self.assertEqual(chosen.tolist(), [3, 4])
            self.assertEqual(meta["selected_samples"], 2)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainerSelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Sequence[str]) -> int:
    if len(argv) == 1 and argv[0] == "--self-test":
        return run_self_tests()
    args = parse_args(argv)
    if args.sweep is not None and args.sweep not in {"systematic_v1", SUBSET_SWEEP_NAME}:
        fail(f"unsupported sweep {args.sweep}")
    if args.active_learning_chunks is not None and args.active_learning_chunk_size is not None:
        fail("use either --active-learning-chunks or --active-learning-chunk-size, not both")
    if (
        args.active_learning_pool_cutoff_cp is not None
        and args.active_learning_chunks is None
        and args.active_learning_chunk_size is None
    ):
        fail("--active-learning-pool-cutoff requires active-learning chunks or chunk size")
    if args.active_learning_match_top_k <= 0:
        fail("--active-learning-match-top-k must be positive")
    if (args.active_learning_chunks is not None or args.active_learning_chunk_size is not None) and args.sweep is not None:
        fail("active-learning runs are only supported for standalone candidates right now")
    if args.patience is not None and args.patience <= 0:
        fail("--patience must be positive")
    if args.runtime_loss_interval is not None and args.runtime_loss_interval <= 0:
        fail("--runtime-loss-interval must be positive")

    root = workspace_root()
    dataset = resolve_dataset(root, args.dataset_arg)
    engine_template_path = resolve_engine_path(root, args.engine_arg)
    reference_source_path = (
        resolve_engine_path(root, args.reference_source)
        if args.reference_source is not None
        else engine_template_path
    )
    book_path = (
        ensure_wall_variations_book(root)
        if args.checkpoint_book is None and args.tier == "fast"
        else resolve_checkpoint_book(root, args.checkpoint_book)
    )
    nnue_bin = ensure_repo_binary(root, root / "nnue/Cargo.toml", "nnue")
    match_bin = ensure_repo_binary(root, root / "arena/Cargo.toml", "codingame_source_match")

    base_corpus = ensure_base_random_corpus(
        root,
        nnue_bin,
        dataset,
        args.seed,
        args.validation_samples,
        args.max_abs_score,
    )
    index = ensure_selection_index(base_corpus)

    if args.sweep == "systematic_v1":
        return run_systematic_v1(
            root=root,
            args=args,
            dataset=dataset,
            base_corpus=base_corpus,
            index=index,
            nnue_bin=nnue_bin,
            match_bin=match_bin,
            book_path=book_path,
            engine_template_path=engine_template_path,
        )
    if args.sweep == SUBSET_SWEEP_NAME:
        return run_subset_lab_v1(
            root=root,
            args=args,
            dataset=dataset,
            base_corpus=base_corpus,
            index=index,
            nnue_bin=nnue_bin,
            match_bin=match_bin,
            book_path=book_path,
            engine_template_path=engine_template_path,
        )

    spec = standalone_spec(args, dataset, engine_template_path, reference_source_path)
    run_candidate(
        root=root,
        dataset=dataset,
        base_corpus=base_corpus,
        index=index,
        nnue_bin=nnue_bin,
        match_bin=match_bin,
        book_path=book_path,
        spec=spec,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
