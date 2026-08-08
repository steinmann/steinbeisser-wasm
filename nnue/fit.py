#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

NNUE_DIR = Path(__file__).resolve().parent
WORKSPACE = NNUE_DIR.parent


def read_json(path: Path, default: object | None = None) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    row_lengths: tuple[int, ...]
    sparse_count: int
    dense_count: int
    max_active_features: int
    dense_feature_names: tuple[str, ...]
    dense_feature_scales: tuple[float, ...]

    @property
    def cell_count(self) -> int:
        return sum(self.row_lengths)

    @property
    def input_count(self) -> int:
        return self.sparse_count + self.dense_count


def current_feature_schema() -> FeatureSchema:
    return schema_from_rust_source(NNUE_DIR / "src/core.rs")


def schema_from_rust_source(path: Path) -> FeatureSchema:
    text = path.read_text(encoding="utf-8")
    row_lengths = tuple(parse_rust_array(text, "FEATURE_SCHEMA_ROW_LENGTHS", int))
    dense_names = tuple(parse_rust_array(text, "DENSE_FEATURE_NAMES", str))
    dense_scales = tuple(parse_rust_array(text, "DENSE_FEATURE_SCALES", float, source=text))
    name = parse_rust_string_const(text, "FEATURE_SET_NAME")
    max_active = parse_rust_usize_const(text, "MAX_ACTIVE_FEATURES")
    return FeatureSchema(
        name=name,
        row_lengths=row_lengths,
        sparse_count=sum(row_lengths) * 2,
        dense_count=len(dense_names),
        max_active_features=max_active,
        dense_feature_names=dense_names,
        dense_feature_scales=dense_scales,
    )


def parse_rust_string_const(text: str, name: str) -> str:
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', text)
    if not match:
        raise ValueError(f"missing Rust const {name}")
    return match.group(1)


def parse_rust_usize_const(text: str, name: str) -> int:
    match = re.search(rf"pub const {name}: usize = ([^;]+);", text)
    if not match:
        raise ValueError(f"missing Rust const {name}")
    expression = match.group(1).strip()
    if re.fullmatch(r"[0-9_]+", expression):
        return int(expression.replace("_", ""))
    max_pieces = parse_engine_max_pieces()
    expression = expression.replace("Position::MAX_PIECES_PER_SIDE", str(max_pieces))
    if re.fullmatch(r"[0-9_ ]+\*[0-9_ ]+", expression):
        left, right = expression.split("*", 1)
        return int(left.replace("_", "").strip()) * int(right.replace("_", "").strip())
    raise ValueError(f"unsupported Rust const expression for {name}: {match.group(1)}")


def parse_engine_max_pieces() -> int:
    text = (WORKSPACE / "engine/src/board.rs").read_text(encoding="utf-8")
    match = re.search(r"pub const MAX_PIECES_PER_SIDE: usize = ([0-9_]+);", text)
    if not match:
        raise ValueError("missing Position::MAX_PIECES_PER_SIDE source constant")
    return int(match.group(1).replace("_", ""))


def parse_rust_array(text: str, name: str, item_type: type, source: str | None = None) -> list:
    match = re.search(
        rf"pub const {name}: \[[^]]+\] = \[(.*?)\];",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing Rust array const {name}")
    body = re.sub(r"//.*", "", match.group(1))
    values = [value.strip() for value in body.split(",") if value.strip()]
    if item_type is str:
        return [value.strip('"') for value in values]
    if item_type is int:
        return [int(value.replace("_", "")) for value in values]
    if item_type is float:
        return [parse_rust_float(value, source or text) for value in values]
    raise TypeError(item_type)


def parse_rust_float(value: str, source: str) -> float:
    normalized = value.rstrip("f32").replace("_", "")
    try:
        return float(normalized)
    except ValueError:
        match = re.search(rf"pub const {re.escape(value)}: f32 = ([0-9_.]+);", source)
        if not match:
            raise
        return float(match.group(1).replace("_", ""))


def huber_loss(error: float, delta: float = 1.0) -> float:
    absolute = abs(error)
    if absolute <= delta:
        return 0.5 * error * error
    return delta * (absolute - 0.5 * delta)


def relu(value: float) -> float:
    return value if value > 0.0 else 0.0


FEATURE_SCHEMA = current_feature_schema()
ROW_LENGTHS = list(FEATURE_SCHEMA.row_lengths)
FEATURE_SET_NAME = FEATURE_SCHEMA.name
DEFAULT_FEATURE_SET_NAME = FEATURE_SET_NAME
CELL_COUNT = FEATURE_SCHEMA.cell_count
SPARSE_FEATURE_COUNT = FEATURE_SCHEMA.sparse_count
MAX_ACTIVE_FEATURES = FEATURE_SCHEMA.max_active_features
DENSE_FEATURE_COUNT = FEATURE_SCHEMA.dense_count
TURNS_TO_LIMIT_SCALE = 1.0
NO_PROGRESS_PLIES_SCALE = 64.0

DENSE_FEATURE_NAMES = list(FEATURE_SCHEMA.dense_feature_names)
DENSE_FEATURE_SCALES = list(FEATURE_SCHEMA.dense_feature_scales)


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    sparse_count: int
    dense_count: int
    max_active_features: int

    @property
    def input_count(self) -> int:
        return self.sparse_count + self.dense_count


FEATURE_SPEC = FeatureSetSpec(
    FEATURE_SET_NAME,
    SPARSE_FEATURE_COUNT,
    DENSE_FEATURE_COUNT,
    MAX_ACTIVE_FEATURES,
)

def feature_spec(name: str) -> FeatureSetSpec:
    if name != FEATURE_SET_NAME:
        raise ValueError(f"unsupported feature set {name!r}")
    return FEATURE_SPEC


def dense_feature_names_for_set(feature_set_name: str) -> list[str]:
    feature_spec(feature_set_name)
    return list(DENSE_FEATURE_NAMES)


def resolve_dense_feature_offsets(feature_set_name: str, normalization: dict | None) -> list[float]:
    scales = _default_dense_feature_scales(feature_set_name)
    offsets = [0.0 for _ in scales]
    provided = (normalization or {}).get("dense_feature_offsets")
    if isinstance(provided, list):
        for index, value in enumerate(provided[: len(offsets)]):
            offsets[index] = float(value)
    return offsets


def resolve_dense_feature_scales(feature_set_name: str, normalization: dict | None) -> list[float]:
    scales = _default_dense_feature_scales(feature_set_name)
    provided = (normalization or {}).get("dense_feature_scales")
    if isinstance(provided, list):
        for index, value in enumerate(provided[: len(scales)]):
            scales[index] = max(float(value), 1.0)
        return scales

    edge_abs_max = float((normalization or {}).get("edge_abs_max", scales[0]))
    compact_abs_max = float((normalization or {}).get("compact_abs_max", scales[1]))
    liberty_abs_max = float((normalization or {}).get("liberty_abs_max", scales[4]))
    isolated_abs_max = float((normalization or {}).get("isolated_abs_max", scales[5]))
    scales[0] = max(edge_abs_max, 1.0)
    scales[1] = max(compact_abs_max, 1.0)
    scales[4] = max(liberty_abs_max, 1.0)
    scales[5] = max(isolated_abs_max, 1.0)
    return scales


def _default_dense_feature_scales(feature_set_name: str) -> list[float]:
    feature_spec(feature_set_name)
    return list(DENSE_FEATURE_SCALES)

# Trainer --------------------------------------------------------------------

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None
jax = None
jnp = None


def ensure_jax_loaded() -> None:
    global JAX_AVAILABLE, jax, jnp
    if not JAX_AVAILABLE or jax is not None:
        return
    try:
        import jax as loaded_jax
        import jax.numpy as loaded_jnp
    except ImportError:
        JAX_AVAILABLE = False
        return
    jax = loaded_jax
    jnp = loaded_jnp

import numpy as np

DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_MIN_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_PATIENCE = 5
GRADIENT_CLIP_NORM = 1.0
NNQ_MAGIC = b"NNQ1"
NNQ_VERSION = 7
CLIPPED_RELU_ACTIVATION = 1
SCALAR_BACKEND_ID = 1
TARGET_TRANSFORM_LINEAR_CLIP_V1 = 1
SPARSE_WEIGHT_QUANT_RANGE = 32_760.0
DENSE_WEIGHT_QUANT_RANGE = 127.0
ACTIVATION_QUANT_RANGE = 127.0
RUNTIME_ACTIVATION_MARGIN = 1.10
RUNTIME_ACTIVATION_SAMPLE_LIMIT = 8_192
TRAIN_EVAL_MONITOR_SAMPLE_LIMIT = 100_000
NNUE_CLI_ENV = "STEINBEISSER_NNUE_CLI"
NNUE_BOARD_RADIUS_MARKER = 23

@dataclass
class TrainingConfig:
    train_path: str
    val_path: str
    manifest_path: str
    output_dir: str
    feature_set: str = DEFAULT_FEATURE_SET_NAME
    architecture: str | None = None
    lambda_mix: float = 0.5
    epoch_size: int | None = None
    epochs: int = 10
    batch_size: int = 256
    threads: int | None = None
    loader_workers: int | None = None
    dataset_cache_dir: str | None = None
    learning_rate: float = DEFAULT_LEARNING_RATE
    min_learning_rate: float = DEFAULT_MIN_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    patience: int = DEFAULT_PATIENCE
    screen_checkpoint_count: int = 0
    selection_interval: int = 1
    runtime_loss_interval: int = 1
    ema_decay: float | None = None
    seed: int = 1
    nnue_cli: str | None = None


@dataclass
class NumpyDataset:
    inputs: np.ndarray
    targets: np.ndarray
    train_weights: np.ndarray
    eval_weights: np.ndarray
    keep_probabilities: np.ndarray

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])


@dataclass
class DatasetLoadResult:
    dataset: NumpyDataset
    selection_report: dict | None = None






















def resolved_thread_count(config: TrainingConfig) -> int:
    if config.threads is not None:
        return max(int(config.threads), 1)
    return max(int(os.cpu_count() or 1), 1)




def manifest_sample_limit(manifest: dict, *, train: bool) -> int | None:
    split = "train" if train else "val"
    samples = manifest.get(split, {}).get("samples")
    return int(samples) if samples is not None else None


def run_training(config: TrainingConfig) -> dict:
    ensure_jax_loaded()
    if not JAX_AVAILABLE:
        raise RuntimeError("jax is required for training")

    manifest = read_json(Path(config.manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f"{config.manifest_path} must contain a JSON object")
    feature_name = config.feature_set or manifest.get("feature_set", DEFAULT_FEATURE_SET_NAME)
    spec = feature_spec(feature_name)
    architecture = resolve_architecture(config.architecture, spec.input_count)
    validate_training_config(config, "jax_mlp")
    normalization = resolve_training_dense_normalization(
        manifest,
        feature_name,
    )
    thread_count = resolved_thread_count(config)
    print(f"training threads: resolved={thread_count} dataset_loader=rust_prepare")

    train_sample_limit = manifest_sample_limit(manifest, train=True)
    val_sample_limit = manifest_sample_limit(manifest, train=False)
    train_load_started = time.perf_counter()
    print(f"dataset_load=start split=train path={config.train_path}", flush=True)
    train_load = load_dataset(
        config.train_path,
        manifest,
        spec.name,
        normalization,
        config,
        train=True,
        sample_limit=train_sample_limit,
    )
    print(
        "dataset_load=done "
        f"split=train samples={train_load.dataset.size} "
        f"elapsed_s={time.perf_counter() - train_load_started:.1f}",
        flush=True,
    )
    val_load_started = time.perf_counter()
    print(f"dataset_load=start split=val path={config.val_path}", flush=True)
    val_load = load_dataset(
        config.val_path,
        manifest,
        spec.name,
        normalization,
        config,
        train=False,
        sample_limit=val_sample_limit,
    )
    print(
        "dataset_load=done "
        f"split=val samples={val_load.dataset.size} "
        f"elapsed_s={time.perf_counter() - val_load_started:.1f}",
        flush=True,
    )
    train_data = train_load.dataset
    val_data = val_load.dataset
    train_dataset_size = train_data.size
    resolved_epoch_size = resolve_epoch_sample_limit(train_data, config)

    model = JaxMlpModel(
        architecture=architecture,
        sparse_input_count=spec.sparse_count,
        dense_input_count=spec.dense_count,
        seed=config.seed,
        weight_decay=config.weight_decay,
        ema_decay=config.ema_decay,
    )
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_loss_binary = resolve_runtime_loss_binary(config)
    provenance = build_training_provenance(
        manifest,
        config,
        spec.name,
    )

    history: list[dict] = []
    best_val_loss = math.inf
    best_quantized_val_loss = None
    best_epoch = 0
    best_state: dict | None = None
    best_selection_metric = math.inf
    epochs_without_improvement = 0
    stopped_epoch = 0
    float_best_epoch = 0
    float_best_val_loss = math.inf
    float_best_quantized_val_loss = None
    selection_states: dict[int, dict] = {}

    calibration_inputs = calibration_inputs_from_dataset(
        val_data if val_data.size else train_data,
        limit=RUNTIME_ACTIVATION_SAMPLE_LIMIT,
    )

    epoch_cap = config.epochs if config.epochs > 0 else None
    epoch_limit_label = str(epoch_cap) if epoch_cap is not None else "patience"

    def is_selection_candidate(epoch_value: int | float) -> bool:
        epoch_number = int(epoch_value)
        if epoch_cap is not None and epoch_number == epoch_cap:
            return True
        return config.selection_interval <= 1 or epoch_number % config.selection_interval == 0

    def selection_metric_name() -> str:
        if runtime_loss_binary:
            if config.runtime_loss_interval <= 1:
                return "quantized_val_loss"
            return f"quantized_val_loss_every_{config.runtime_loss_interval}_epochs"
        return "val_loss"

    def selection_metric_for_epoch(
        val_loss: float, quantized_val_metrics: dict | None
    ) -> float | None:
        if runtime_loss_binary:
            if quantized_val_metrics is None:
                return None
            return float(quantized_val_metrics["loss"])
        return float(val_loss)

    epoch = 1
    while True:
        learning_rate = epoch_learning_rate(epoch, config)
        train_metrics = train_epoch(model, train_data, config, rng, learning_rate)
        train_eval_metrics = evaluate_subset(
            model,
            train_data,
            config,
            train_metrics["monitor_indices"],
        )
        val_metrics = evaluate(model, val_data, config)
        current_state = model.export_state(spec.name, normalization)
        current_state.update(provenance)
        current_state["runtime_activation_scales"] = calibrate_runtime_activation_scales(
            current_state,
            calibration_inputs,
            margin=RUNTIME_ACTIVATION_MARGIN,
        )
        quantized_val_metrics = None
        if should_run_runtime_validation(epoch, config):
            quantized_val_metrics = evaluate_quantized_runtime_loss(
                current_state,
                config.val_path,
                config,
                output_dir,
                runtime_loss_binary,
            )
        selection_metric = selection_metric_for_epoch(
            val_metrics["loss"], quantized_val_metrics
        )
        row = {
            "epoch": epoch,
            "phase": "float",
            "learning_rate": learning_rate,
            "train_loss": train_eval_metrics["loss"],
            "train_samples": train_metrics["samples"],
            "train_eval_samples": train_eval_metrics["samples"],
            "train_batch_loss": train_metrics["loss"],
            "train_batch_samples": train_metrics["samples"],
            "train_weight_mass": train_metrics["weight_mass"],
            "val_loss": val_metrics["loss"],
            "val_samples": val_metrics["samples"],
            "quantized_val_loss": (
                quantized_val_metrics["loss"] if quantized_val_metrics is not None else None
            ),
            "quantized_val_samples": (
                quantized_val_metrics["samples"] if quantized_val_metrics is not None else None
            ),
            "runtime_loss_evaluated": quantized_val_metrics is not None,
            "selection_metric": selection_metric,
            "selection_candidate": is_selection_candidate(epoch),
        }
        history.append(row)
        if row["selection_candidate"] and selection_metric is not None:
            selection_states[int(row["epoch"])] = current_state
        print(
            f"epoch {epoch}/{epoch_limit_label}: "
            f"lr={learning_rate:.6g} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"quantized_val_loss={format_optional_loss(row['quantized_val_loss'])} "
            f"train_samples={row['train_samples']} "
            f"train_eval_samples={row['train_eval_samples']} "
            f"val_samples={row['val_samples']}"
        )

        if row["val_loss"] < float_best_val_loss:
            float_best_epoch = epoch
            float_best_val_loss = row["val_loss"]
            float_best_quantized_val_loss = row["quantized_val_loss"]

        if (
            row["selection_candidate"]
            and selection_metric is not None
            and selection_metric < best_selection_metric
        ):
            best_selection_metric = selection_metric
            best_epoch = epoch
            best_val_loss = row["val_loss"]
            best_quantized_val_loss = row["quantized_val_loss"]
            best_state = current_state
            epochs_without_improvement = 0
        elif row["selection_candidate"] and selection_metric is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_epoch = epoch
                break
        if epoch_cap is not None and epoch >= epoch_cap:
            stopped_epoch = epoch
            break
        epoch += 1

    if best_state is None:
        best_state = model.export_state(spec.name, normalization)
        best_epoch = history[-1]["epoch"] if history else 0
        best_val_loss = history[-1]["val_loss"] if history else math.inf
        best_quantized_val_loss = history[-1].get("quantized_val_loss") if history else None
        if history:
            last_selection_metric = history[-1].get("selection_metric")
            best_selection_metric = (
                float(last_selection_metric)
                if last_selection_metric is not None
                else float(history[-1]["val_loss"])
            )
        else:
            best_selection_metric = math.inf
    if stopped_epoch == 0:
        stopped_epoch = history[-1]["epoch"] if history else 0

    best_state["runtime_activation_scales"] = calibrate_runtime_activation_scales(
        best_state,
        calibration_inputs,
        margin=RUNTIME_ACTIVATION_MARGIN,
    )

    screen_checkpoint_rows: list[dict] = []
    screen_checkpoint_count = max(0, int(config.screen_checkpoint_count))
    if screen_checkpoint_count > 0:
        ranked_screen_rows = sorted(
            [
                row
                for row in history
                if row.get("quantized_val_loss") is not None
                and int(row.get("epoch", 0)) in selection_states
            ],
            key=lambda row: (float(row["quantized_val_loss"]), int(row["epoch"])),
        )
        checkpoint_dir = output_dir / "screen_checkpoints"
        for rank, row in enumerate(ranked_screen_rows[:screen_checkpoint_count], start=1):
            epoch_number = int(row["epoch"])
            state = selection_states[epoch_number]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_json_path = checkpoint_dir / f"rank_{rank:02}_epoch_{epoch_number:04d}_model.json"
            checkpoint_nnq_path = checkpoint_dir / f"rank_{rank:02}_epoch_{epoch_number:04d}_model.nnq"
            with checkpoint_json_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
            write_quantized_runtime_model(state, checkpoint_nnq_path)
            row["screen_checkpoint_rank"] = rank
            row["screen_checkpoint_file"] = str(checkpoint_nnq_path)
            screen_checkpoint_rows.append(row)

    selection_report_payload = train_load.selection_report
    summary = {
        "feature_set": spec.name,
        "backend": model.backend_name,
        "architecture": architecture,
        "optimizer": "adamw",
        "class_weighting": "inverse_sqrt",
        "lambda": config.lambda_mix,
        "epoch_size": config.epoch_size,
        "epoch_mode": "full_dataset" if config.epoch_size is None else "sample_budget",
        "epochs": config.epochs,
        "epoch_cap": config.epochs if config.epochs > 0 else None,
        "learning_rate_schedule_horizon": (
            config.epochs if config.epochs > 0 else max(config.warmup_epochs + config.patience * 8, 40)
        ),
        "batch_size": config.batch_size,
        "selection_report": selection_report_payload,
        "threads": thread_count,
        "dataset_cache_dir": str(resolve_dataset_cache_dir(config)) if config.dataset_cache_dir else None,
        "repeat_occurrence_weight": "sqrt_cap2",
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "patience": config.patience,
        "screen_checkpoint_count": config.screen_checkpoint_count,
        "selection_interval": config.selection_interval,
        "runtime_loss_interval": config.runtime_loss_interval,
        "ema_decay": config.ema_decay,
        "activation": "relu",
        "norm": "none",
        "block_type": "plain",
        "float_best_epoch": float_best_epoch,
        "float_best_val_loss": float_best_val_loss,
        "best_epoch": best_epoch,
        "screen_checkpoint_epochs": [int(row["epoch"]) for row in screen_checkpoint_rows],
        "best_val_loss": best_val_loss,
        "best_quantized_val_loss": best_quantized_val_loss,
        "best_selection_metric": best_selection_metric,
        "selection_metric_name": selection_metric_name(),
        "float_best_quantized_val_loss": float_best_quantized_val_loss,
        "runtime_loss_binary": runtime_loss_binary,
        "stopped_epoch": stopped_epoch,
        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_data.size,
        "resolved_epoch_size": resolved_epoch_size,
        "effective_float_dataset_passes": (
            sum(float(row.get("train_samples", 0)) for row in history if row.get("phase") == "float")
            / max(train_dataset_size, 1)
        ),
        "history": history,
    }
    summary.update(provenance)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def resolve_runtime_loss_binary(config: TrainingConfig) -> str | None:
    if config.nnue_cli:
        return str(config.nnue_cli)
    env_value = os.environ.get(NNUE_CLI_ENV)
    if env_value:
        return env_value
    return None


def build_training_provenance(
    manifest: dict,
    config: TrainingConfig,
    feature_set_name: str,
) -> dict:
    return {
        "dataset_path": config.train_path,
        "val_dataset_path": config.val_path,
        "manifest_path": config.manifest_path,
        "manifest_feature_set": str(manifest.get("feature_set", "")),
        "feature_set_used": feature_set_name,
        "dataset_phase_bucket_count": manifest.get("phase_bucket_count"),
    }


def evaluate_quantized_runtime_loss(
    state: dict,
    dataset_path: str,
    config: TrainingConfig,
    output_dir: Path,
    runtime_loss_binary: str | None,
) -> dict | None:
    if runtime_loss_binary is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix="_runtime_validation_model_",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(state, handle)
        handle.write("\n")
        model_path = Path(handle.name)

    command = [
        runtime_loss_binary,
        "runtime-loss",
        "--model",
        str(model_path),
        "--dataset",
        dataset_path,
        "--lambda",
        str(config.lambda_mix),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"failed to launch runtime-loss binary {runtime_loss_binary}: {error}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "runtime-loss command failed:\n"
            f"command: {' '.join(command)}\n"
            f"stdout: {error.stdout}\n"
            f"stderr: {error.stderr}"
        ) from error
    finally:
        model_path.unlink(missing_ok=True)


def format_optional_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_quantized_runtime_model(state: dict, path: Path) -> None:
    feature_set_name = str(state["feature_set"])
    if feature_set_name != FEATURE_SET_NAME:
        raise ValueError(f"unsupported runtime feature set {feature_set_name}")

    hidden_sizes = [int(value) for value in state.get("hidden_sizes", [])]
    if not hidden_sizes:
        hidden_sizes = [int(state["hidden_size"])]
    if not hidden_sizes:
        raise ValueError("runtime model requires at least one hidden layer")

    sparse_input_count = int(state.get("input_count_sparse", state["input_count"]))
    dense_input_count = int(state.get("input_count_dense", 0))
    dense_normalization = load_dense_normalization(state)
    dense_feature_offsets = [
        float(value) for value in dense_normalization.get("dense_feature_offsets", [])
    ]
    dense_feature_scales = resolve_dense_feature_scales(feature_set_name, dense_normalization)
    if str(state.get("activation")) != "relu":
        raise ValueError("runtime model export requires relu activation")
    activation_id = CLIPPED_RELU_ACTIVATION
    target_transform = str(state.get("target_transform"))
    if target_transform != "linear_clip_v1":
        raise ValueError(f"unsupported runtime target transform {target_transform}")
    target_transform_id = TARGET_TRANSFORM_LINEAR_CLIP_V1

    sparse_weights_matrix = state["w1_sparse"]
    sparse_weights_flat = [
        float(value)
        for row in sparse_weights_matrix
        for value in row
    ]
    max_abs_sparse = max((abs(value) for value in sparse_weights_flat), default=0.0)
    sparse_weight_scale = (
        max_abs_sparse / SPARSE_WEIGHT_QUANT_RANGE
        if max_abs_sparse > 0.0
        else 1.0 / SPARSE_WEIGHT_QUANT_RANGE
    )
    quantized_sparse = [
        quantize_i16(value, sparse_weight_scale)
        for value in sparse_weights_flat
    ]
    quantized_biases = [
        int(round(float(value) / sparse_weight_scale))
        for value in state["b1"]
    ]
    activation_scales = parse_runtime_activation_scales(state, hidden_sizes)
    architecture = state_architecture(state, sparse_input_count + dense_input_count, hidden_sizes)
    weights, biases = state_layers_to_python(
        state,
        architecture,
        sparse_input_count,
        dense_input_count,
    )

    runtime_bytes = bytearray()
    runtime_bytes.extend(NNQ_MAGIC)
    runtime_bytes.extend(
        struct.pack(
            "<HBBBBIII",
            NNQ_VERSION,
            NNUE_BOARD_RADIUS_MARKER,
            activation_id,
            SCALAR_BACKEND_ID,
            target_transform_id,
            sparse_input_count,
            dense_input_count,
            len(hidden_sizes),
        )
    )
    for size in hidden_sizes:
        runtime_bytes.extend(struct.pack("<I", int(size)))
    runtime_bytes.extend(
        struct.pack(
            "<fI",
            float(sparse_weight_scale),
            len(dense_feature_scales),
        )
    )
    if dense_feature_scales:
        runtime_bytes.extend(struct.pack(f"<{len(dense_feature_scales)}f", *dense_feature_scales))
    runtime_bytes.extend(struct.pack("<I", len(dense_feature_offsets)))
    if dense_feature_offsets:
        runtime_bytes.extend(struct.pack(f"<{len(dense_feature_offsets)}f", *dense_feature_offsets))
    runtime_bytes.extend(struct.pack("<f", float(parse_scalar_field(state[f"b{len(hidden_sizes) + 1}"]))))
    runtime_bytes.extend(struct.pack(f"<{len(activation_scales)}f", *activation_scales))
    if quantized_biases:
        runtime_bytes.extend(struct.pack(f"<{len(quantized_biases)}i", *quantized_biases))
    if quantized_sparse:
        runtime_bytes.extend(struct.pack(f"<{len(quantized_sparse)}h", *quantized_sparse))

    dense_weights = flatten_matrix(state.get("w1_dense", []))
    if dense_weights:
        runtime_bytes.extend(struct.pack(f"<{len(dense_weights)}f", *dense_weights))

    for layer_weights, layer_biases in zip(weights[1:-1], biases[1:-1]):
        padded_input_size, weight_scale, quantized_weights = quantize_dense_runtime_layer(
            layer_weights
        )
        runtime_bytes.extend(struct.pack("<If", padded_input_size, weight_scale))
        runtime_bytes.extend(struct.pack(f"<{len(layer_biases)}f", *layer_biases))
        runtime_bytes.extend(struct.pack(f"<{len(quantized_weights)}b", *quantized_weights))

    output_padded_input_size, output_weight_scale, quantized_output_weights = (
        quantize_output_runtime_layer(parse_output_weights(state[f"w{len(hidden_sizes) + 1}"]))
    )
    runtime_bytes.extend(struct.pack("<If", output_padded_input_size, output_weight_scale))
    runtime_bytes.extend(struct.pack(f"<{len(quantized_output_weights)}b", *quantized_output_weights))

    path.write_bytes(runtime_bytes)


def flatten_matrix(values: list[list[float]]) -> list[float]:
    return [float(value) for row in values for value in row]


def parse_output_weights(values: list[float] | list[list[float]]) -> list[float]:
    if values and isinstance(values[0], list):
        return [float(row[0]) for row in values]
    return [float(value) for value in values]


def parse_scalar_field(value: float | list[float]) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected scalar output bias")
        return float(value[0])
    return float(value)


def quantize_i16(value: float, scale: float) -> int:
    quantized = round(value / scale)
    return int(max(-32_768, min(32_767, quantized)))


def quantize_i8(value: float, scale: float) -> int:
    quantized = round(value / scale)
    return int(max(-127, min(127, quantized)))


def padded_width(width: int) -> int:
    if width <= 0:
        return 0
    return ((width + 15) // 16) * 16


def state_architecture(state: dict, input_count: int, hidden_sizes: list[int]) -> list[int]:
    explicit = [int(value) for value in state.get("architecture", [])]
    if explicit:
        return explicit
    return [input_count, *hidden_sizes, 1]


def quantized_weight_scale(values: list[float]) -> float:
    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs > 0.0:
        return max_abs / DENSE_WEIGHT_QUANT_RANGE
    return 1.0 / DENSE_WEIGHT_QUANT_RANGE


def quantize_dense_runtime_layer(
    layer_weights: list[list[float]],
) -> tuple[int, float, list[int]]:
    input_size = len(layer_weights)
    output_size = len(layer_weights[0]) if layer_weights else 0
    padded_input_size = padded_width(input_size)
    flattened = [float(value) for row in layer_weights for value in row]
    weight_scale = quantized_weight_scale(flattened)
    quantized_weights = [0 for _ in range(output_size * padded_input_size)]
    for output_index in range(output_size):
        row_start = output_index * padded_input_size
        for input_index in range(input_size):
            quantized_weights[row_start + input_index] = quantize_i8(
                float(layer_weights[input_index][output_index]),
                weight_scale,
            )
    return padded_input_size, weight_scale, quantized_weights


def quantize_output_runtime_layer(output_weights: list[float]) -> tuple[int, float, list[int]]:
    input_size = len(output_weights)
    padded_input_size = padded_width(input_size)
    weight_scale = quantized_weight_scale([float(value) for value in output_weights])
    quantized = [0 for _ in range(padded_input_size)]
    for index, value in enumerate(output_weights):
        quantized[index] = quantize_i8(float(value), weight_scale)
    return padded_input_size, weight_scale, quantized


def parse_runtime_activation_scales(
    state: dict,
    hidden_sizes: list[int],
) -> list[float]:
    raw = state.get("runtime_activation_scales")
    if raw is None:
        raise ValueError("runtime_activation_scales is required")
    if len(raw) != len(hidden_sizes):
        raise ValueError(
            f"runtime_activation_scales length {len(raw)} does not match hidden layer count {len(hidden_sizes)}"
        )
    scales = [max(float(value), 1.0 / ACTIVATION_QUANT_RANGE) for value in raw]
    if not all(math.isfinite(value) and value > 0.0 for value in scales):
        raise ValueError("runtime_activation_scales must contain finite positive values")
    return scales








































































def resolve_dataset_cache_dir(config: TrainingConfig) -> Path | None:
    if config.dataset_cache_dir:
        return Path(config.dataset_cache_dir)
    return None


def dataset_cache_file_signature(path: str | None) -> dict | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dataset_cache_key_payload(
    path: str,
    manifest_path: str,
    feature_set_name: str,
    normalization: dict,
    config: TrainingConfig,
    sample_limit: int | None,
) -> dict:
    return {
        "version": 12,
        "array_format": "raw-f32-v1",
        "prepared_by": "nnue prepare-dataset",
        "dataset": dataset_cache_file_signature(path),
        "manifest": dataset_cache_file_signature(manifest_path),
        "feature_set": feature_set_name,
        "dense_feature_offsets": [float(value) for value in normalization.get("dense_feature_offsets", [])],
        "dense_feature_scales": [float(value) for value in normalization.get("dense_feature_scales", [])],
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "lambda_mix": float(config.lambda_mix),
        "class_weighting": "inverse_sqrt",
        "repeat_occurrence_weight": "sqrt_cap2",
    }


def dataset_cache_key(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]




















DATASET_CACHE_ARRAY_NAMES = (
    "inputs",
    "targets",
    "train_weights",
    "eval_weights",
    "keep_probabilities",
)


def dataset_cache_format() -> str:
    return "raw-f32-v1"


def dataset_cache_mmap_mode() -> str | None:
    value = os.environ.get("STEINBEISSER_NNUE_DATASET_CACHE_MMAP", "1").strip().lower()
    return None if value in {"0", "false", "no", "off"} else "r"


def dataset_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key


def load_raw_f32_arrays(cache_path: Path, metadata: dict) -> dict[str, np.ndarray]:
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, dict):
        raise ValueError("raw dataset cache metadata is missing arrays")
    mmap_mode = dataset_cache_mmap_mode()
    arrays: dict[str, np.ndarray] = {}
    for name in DATASET_CACHE_ARRAY_NAMES:
        entry = array_metadata.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"raw dataset cache metadata is missing {name}")
        shape = tuple(int(value) for value in entry.get("shape", []))
        file_name = str(entry.get("file") or f"{name}.f32")
        file_path = cache_path / file_name
        if mmap_mode is None:
            arrays[name] = np.fromfile(file_path, dtype=np.float32).reshape(shape)
        else:
            arrays[name] = np.memmap(file_path, dtype=np.float32, mode=mmap_mode, shape=shape)
    return arrays


def load_cached_dataset(cache_path: Path, meta_path: Path) -> DatasetLoadResult | None:
    if not meta_path.is_file():
        return None
    try:
        metadata = read_json(meta_path)
        if not isinstance(metadata, dict):
            return None
        if not (cache_path.is_dir() and metadata.get("array_format") == "raw-f32-v1"):
            return None
        arrays = load_raw_f32_arrays(cache_path, metadata)
        dataset = NumpyDataset(
            inputs=arrays["inputs"],
            targets=arrays["targets"],
            train_weights=arrays["train_weights"],
            eval_weights=arrays["eval_weights"],
            keep_probabilities=arrays["keep_probabilities"],
        )
        report = metadata.get("selection_report")
        if report is not None:
            report = json.loads(json.dumps(report))
            report["cache_hit"] = True
            report["cache_file"] = str(cache_path)
            report["cache_format"] = "raw-f32-v1"
            report["cache_key"] = metadata.get("cache_key")
        return DatasetLoadResult(dataset=dataset, selection_report=report)
    except Exception:
        return None




def rust_dataset_prep_supported(config: TrainingConfig) -> bool:
    return resolve_runtime_loss_binary(config) is not None


def build_rust_cached_numpy_dataset(
    cache_path: Path,
    meta_path: Path,
    key: str,
    path: str,
    config: TrainingConfig,
    feature_set_name: str,
    normalization: dict,
    sample_limit: int | None,
) -> DatasetLoadResult:
    nnue_cli = resolve_runtime_loss_binary(config)
    if not nnue_cli:
        raise RuntimeError("STEINBEISSER_NNUE_CLI is required for Rust dataset preparation")

    tmp_path = cache_path.with_name(f".{cache_path.name}.rust-tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    command = [
        nnue_cli,
        "prepare-dataset",
        "--dataset",
        path,
        "--out-dir",
        str(tmp_path),
        "--manifest",
        config.manifest_path,
        "--feature-set",
        feature_set_name,
        "--normalization-json",
        json.dumps(normalization, sort_keys=True, separators=(",", ":")),
        "--lambda",
        str(config.lambda_mix),
    ]
    if sample_limit is not None:
        command.extend(["--sample-limit", str(int(sample_limit))])

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=WORKSPACE,
        )
        report = json.loads(completed.stdout)
        if not isinstance(report, dict):
            raise ValueError("prepare-dataset did not return a JSON object")
        if cache_path.exists():
            shutil.rmtree(cache_path)
        tmp_path.rename(cache_path)
        report["cache_build_seconds"] = float(report.get("cache_build_seconds", 0.0))
        report["cache_hit"] = False
        report["cache_key"] = key
        report["cache_file"] = str(cache_path)
        report["cache_format"] = "raw-f32-v1"
        write_json(
            meta_path,
            {
                "cache_key": key,
                "selection_report": report,
                "cache_format": "raw-f32-v1",
                "array_format": "raw-f32-v1",
                "cache_path": cache_path.name,
                "arrays": report.get("arrays", {}),
            },
        )
        loaded = load_cached_dataset(cache_path, meta_path)
        if loaded is None:
            raise RuntimeError(f"failed to load Rust-prepared dataset cache {cache_path}")
        if loaded.selection_report is not None:
            loaded.selection_report["cache_hit"] = False
        return loaded
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def load_dataset(
    path: str,
    manifest: dict,
    feature_set_name: str,
    normalization: dict,
    config: TrainingConfig,
    *,
    train: bool,
    sample_limit: int | None = None,
    allow_shards: bool = True,
) -> DatasetLoadResult:
    if not rust_dataset_prep_supported(config):
        raise RuntimeError("dataset preparation is Rust-only; set STEINBEISSER_NNUE_CLI")
    if config.dataset_cache_dir is None:
        raise RuntimeError("STEINBEISSER_NNUE_DATASET_CACHE_DIR is required")

    if allow_shards:
        shards = manifest_split_shards(manifest, config.manifest_path, train=train)
        if shards:
            return load_sharded_dataset(
                shards,
                manifest,
                feature_set_name,
                normalization,
                config,
                train=train,
                sample_limit=sample_limit,
            )

    cache_started = time.perf_counter()
    payload = dataset_cache_key_payload(
        path,
        config.manifest_path,
        feature_set_name,
        normalization,
        config,
        sample_limit,
    )
    key = dataset_cache_key(payload)
    cache_dir = resolve_dataset_cache_dir(config)
    assert cache_dir is not None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_cache_path(cache_dir, key)
    meta_path = cache_dir / f"{key}.json"
    cached = load_cached_dataset(cache_path, meta_path)
    split = "train" if train else "val"
    if cached is not None:
        if cached.selection_report is not None:
            cached.selection_report["cache_key"] = key
            cached.selection_report["cache_file"] = str(cache_path)
        print(
            "dataset_cache=hit "
            f"split={split} key={key} "
            f"samples={cached.dataset.size} "
            f"elapsed_s={time.perf_counter() - cache_started:.1f}",
            flush=True,
        )
        return cached

    print(f"dataset_cache=miss split={split} key={key}", flush=True)
    built = build_rust_cached_numpy_dataset(
        cache_path,
        meta_path,
        key,
        path,
        config,
        feature_set_name,
        normalization,
        sample_limit,
    )
    print(
        "dataset_cache=saved "
        f"split={split} key={key} "
        f"samples={built.dataset.size} "
        f"format=raw-f32-v1 "
        f"elapsed_s={time.perf_counter() - cache_started:.1f}",
        flush=True,
    )
    return built


def manifest_split_shards(
    manifest: dict,
    manifest_path: str,
    *,
    train: bool,
) -> list[dict[str, object]]:
    split_name = "train" if train else "val"
    split = manifest.get(split_name)
    if not isinstance(split, dict):
        return []
    raw_shards = split.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        return []
    base = Path(manifest_path).resolve().parent
    shards: list[dict[str, object]] = []
    for raw in raw_shards:
        if not isinstance(raw, dict):
            continue
        raw_file = raw.get("file")
        if not isinstance(raw_file, str) or not raw_file:
            continue
        path = Path(raw_file)
        if not path.is_absolute():
            path = base / path
        samples = int(raw.get("samples") or 0)
        if samples <= 0:
            continue
        shards.append({"path": str(path), "samples": samples})
    return shards


def load_sharded_dataset(
    shards: list[dict[str, object]],
    manifest: dict,
    feature_set_name: str,
    normalization: dict,
    config: TrainingConfig,
    *,
    train: bool,
    sample_limit: int | None,
) -> DatasetLoadResult:
    split = "train" if train else "val"
    remaining = sample_limit
    plans: list[tuple[str, int | None]] = []
    started = time.perf_counter()
    for shard in shards:
        if remaining is not None and remaining <= 0:
            break
        shard_samples = int(shard["samples"])
        shard_limit = None
        if remaining is not None:
            shard_limit = min(shard_samples, remaining)
            remaining -= shard_limit
        plans.append((str(shard["path"]), shard_limit))
    if not plans:
        raise RuntimeError(f"manifest has no usable {split} dataset shards")

    def load_plan(plan: tuple[str, int | None]) -> DatasetLoadResult:
        path, shard_limit = plan
        return load_dataset(
            path,
            manifest,
            feature_set_name,
            normalization,
            config,
            train=train,
            sample_limit=shard_limit,
            allow_shards=False,
        )

    first = load_plan(plans[0])
    total_samples = first.dataset.size + sum(
        int(limit) if limit is not None else int(shards[index]["samples"])
        for index, (_path, limit) in enumerate(plans[1:], start=1)
    )
    combined = allocate_combined_dataset(first.dataset, total_samples)
    reports = []
    if first.selection_report is not None:
        reports.append(first.selection_report)
    offset = copy_dataset_slice(combined.dataset, first.dataset, 0)
    del first
    gc.collect()

    for plan in plans[1:]:
        loaded = load_plan(plan)
        if loaded.selection_report is not None:
            reports.append(loaded.selection_report)
        offset = copy_dataset_slice(combined.dataset, loaded.dataset, offset)
        del loaded
        gc.collect()

    if offset != combined.dataset.size:
        combined = DatasetLoadResult(
            dataset=NumpyDataset(
                inputs=combined.dataset.inputs[:offset],
                targets=combined.dataset.targets[:offset],
                train_weights=combined.dataset.train_weights[:offset],
                eval_weights=combined.dataset.eval_weights[:offset],
                keep_probabilities=combined.dataset.keep_probabilities[:offset],
            )
        )

    combined.selection_report = {
        "profile": "full",
        "split": split,
        "sharded_cache": True,
        "shard_count": len(plans),
        "samples_kept": combined.dataset.size,
        "cache_build_seconds": sum(float(report.get("cache_build_seconds", 0.0)) for report in reports),
        "shards": reports,
    }
    print(
        "dataset_cache=sharded "
        f"split={split} shards={len(plans)} samples={combined.dataset.size} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return combined


def allocate_like(array: np.ndarray, samples: int) -> np.ndarray:
    return np.empty((samples, *array.shape[1:]), dtype=array.dtype)


def allocate_combined_dataset(dataset: NumpyDataset, samples: int) -> DatasetLoadResult:
    return DatasetLoadResult(
        dataset=NumpyDataset(
            inputs=allocate_like(dataset.inputs, samples),
            targets=allocate_like(dataset.targets, samples),
            train_weights=allocate_like(dataset.train_weights, samples),
            eval_weights=allocate_like(dataset.eval_weights, samples),
            keep_probabilities=allocate_like(dataset.keep_probabilities, samples),
        )
    )


def copy_dataset_slice(target: NumpyDataset, source: NumpyDataset, start: int) -> int:
    end = start + source.size
    target.inputs[start:end] = source.inputs
    target.targets[start:end] = source.targets
    target.train_weights[start:end] = source.train_weights
    target.eval_weights[start:end] = source.eval_weights
    target.keep_probabilities[start:end] = source.keep_probabilities
    return end


def calibration_inputs_from_dataset(
    dataset: NumpyDataset,
    *,
    limit: int,
):
    if dataset.size == 0:
        return np.empty((0, dataset.inputs.shape[1] if dataset.inputs.ndim == 2 else 0), dtype=np.float32)
    return sample_calibration_rows(dataset.inputs, limit)


def sample_calibration_rows(rows, limit: int):
    total = len(rows)
    if total <= limit:
        return rows
    indices = np.linspace(0, total - 1, num=limit, dtype=np.int64)
    return rows[indices]


def calibrate_runtime_activation_scales(
    state: dict,
    calibration_inputs,
    *,
    margin: float,
) -> list[float]:
    hidden_sizes = [int(value) for value in state.get("hidden_sizes", [])]
    if not hidden_sizes:
        hidden_sizes = [int(state["hidden_size"])]
    if not hidden_sizes:
        raise ValueError("runtime activation calibration requires at least one hidden layer")

    if calibration_inputs is None or len(calibration_inputs) == 0:
        return [1.0 / ACTIVATION_QUANT_RANGE for _ in hidden_sizes]

    sparse_input_count = int(state.get("input_count_sparse", state["input_count"]))
    dense_input_count = int(state.get("input_count_dense", 0))
    architecture = state_architecture(
        state,
        sparse_input_count + dense_input_count,
        hidden_sizes,
    )

    if np is not None:
        inputs = np.asarray(calibration_inputs, dtype=np.float64)
        weights, biases = state_layers_to_numpy(
            state,
            architecture,
            sparse_input_count,
            dense_input_count,
        )
        activations = inputs
        scales: list[float] = []
        for weight_matrix, bias_vector in zip(weights[:-1], biases[:-1]):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                outputs = activations @ weight_matrix.astype(np.float64) + bias_vector.astype(np.float64)
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
            activations = np.maximum(outputs, 0.0)
            max_value = float(np.max(np.abs(activations))) if activations.size else 0.0
            scales.append(max(max_value * margin / ACTIVATION_QUANT_RANGE, 1.0 / ACTIVATION_QUANT_RANGE))
        return scales

    weights, biases = state_layers_to_python(
        state,
        architecture,
        sparse_input_count,
        dense_input_count,
    )
    scales = [0.0 for _ in hidden_sizes]
    for values in calibration_inputs:
        activations = [float(value) for value in values]
        for layer_index, (layer_weights, layer_biases) in enumerate(zip(weights[:-1], biases[:-1])):
            outputs = layer_biases[:]
            for input_index, input_value in enumerate(activations):
                row = layer_weights[input_index]
                for output_index, weight in enumerate(row):
                    outputs[output_index] += input_value * weight
            activations = [relu(value) for value in outputs]
            layer_max = max((abs(value) for value in activations), default=0.0)
            if layer_max > scales[layer_index]:
                scales[layer_index] = layer_max
    return [
        max(scale * margin / ACTIVATION_QUANT_RANGE, 1.0 / ACTIVATION_QUANT_RANGE)
        for scale in scales
    ]


def train_epoch(
    model: "JaxMlpModel",
    dataset: NumpyDataset,
    config: TrainingConfig,
    rng: random.Random,
    learning_rate: float,
) -> dict:
    indices = select_epoch_indices(dataset, config.epoch_size, rng)
    if indices.size == 0:
        return {
            "loss": 0.0,
            "samples": 0,
            "weight_mass": 0.0,
            "monitor_indices": np.empty((0,), dtype=np.int64),
        }

    total_loss = 0.0
    total_samples = 0
    total_weight_mass = 0.0
    batch_size = max(config.batch_size, 1)
    for start in range(0, int(indices.size), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_weights = dataset.train_weights[batch_indices]
        batch_loss = model.train_batch(
            dataset.inputs[batch_indices],
            dataset.targets[batch_indices],
            batch_weights,
            learning_rate,
        )
        batch_weight_mass = float(np.sum(batch_weights, dtype=np.float64))
        total_loss += batch_loss * batch_weight_mass
        total_samples += int(batch_indices.size)
        total_weight_mass += batch_weight_mass
    monitor_limit = min(total_samples, TRAIN_EVAL_MONITOR_SAMPLE_LIMIT)
    return {
        "loss": total_loss / max(total_weight_mass, 1.0),
        "samples": total_samples,
        "weight_mass": total_weight_mass,
        "monitor_indices": indices[:monitor_limit].copy(),
    }


def evaluate(model: "JaxMlpModel", dataset: NumpyDataset, config: TrainingConfig) -> dict:
    if dataset.size == 0:
        return {"loss": 0.0, "samples": 0}
    predictions = np.asarray(model.predict_batch(dataset.inputs), dtype=np.float32)
    errors = predictions - dataset.targets
    losses = huber_loss_vector(errors)
    weighted_loss = float(np.sum(dataset.eval_weights * losses, dtype=np.float64))
    total_weight = float(np.sum(dataset.eval_weights, dtype=np.float64))
    return {
        "loss": weighted_loss / max(total_weight, 1.0),
        "samples": dataset.size,
    }


def evaluate_subset(
    model: "JaxMlpModel",
    dataset: NumpyDataset,
    config: TrainingConfig,
    indices: np.ndarray,
) -> dict:
    if indices.size == 0:
        return {"loss": 0.0, "samples": 0}
    subset = NumpyDataset(
        inputs=dataset.inputs[indices],
        targets=dataset.targets[indices],
        train_weights=dataset.train_weights[indices],
        eval_weights=dataset.eval_weights[indices],
        keep_probabilities=np.ones((int(indices.size),), dtype=np.float32),
    )
    return evaluate(model, subset, config)


def resolve_epoch_sample_limit(dataset: NumpyDataset, config: TrainingConfig) -> int:
    if config.epoch_size is None:
        return dataset.size
    return max(1, min(int(config.epoch_size), dataset.size))


def should_run_runtime_validation(epoch: int, config: TrainingConfig) -> bool:
    interval = max(int(config.runtime_loss_interval), 1)
    if interval <= 1:
        return True
    return epoch == 1 or (epoch % interval) == 0


def select_epoch_indices(
    dataset: NumpyDataset, epoch_size: int | None, rng: random.Random
) -> np.ndarray:
    if dataset.keep_probabilities.size == 0:
        return np.empty((0,), dtype=np.int64)
    np_rng = np.random.default_rng(rng.getrandbits(64))
    if bool(np.all(dataset.keep_probabilities >= 1.0)):
        indices = np_rng.permutation(dataset.size).astype(np.int64, copy=False)
    else:
        mask = np_rng.random(dataset.size, dtype=np.float32) <= dataset.keep_probabilities
        indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if indices.size > 1:
            np_rng.shuffle(indices)
    if epoch_size is not None and indices.size > epoch_size:
        return indices[:epoch_size]
    return indices


def resolve_architecture(raw: str | None, input_count: int) -> list[int]:
    if raw in {None, ""}:
        return [input_count, 128, 64, 1]
    parts = [int(part.strip()) for part in raw.replace("x", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("architecture must define at least one hidden layer")
    if parts[0] == input_count and parts[-1] == 1:
        architecture = parts
    else:
        architecture = [input_count, *parts, 1]
    validate_architecture(architecture, input_count)
    return architecture


def validate_architecture(architecture: list[int], input_count: int) -> None:
    hidden_layers = len(architecture) - 2
    if hidden_layers < 1 or hidden_layers > 2:
        raise ValueError("architecture must have one or two hidden layers")
    if architecture[0] != input_count:
        raise ValueError(f"architecture input {architecture[0]} does not match expected {input_count}")
    if architecture[-1] != 1:
        raise ValueError("architecture output must be 1")


def epoch_learning_rate(epoch: int, config: TrainingConfig) -> float:
    if epoch <= config.warmup_epochs:
        return config.learning_rate * epoch / max(config.warmup_epochs, 1)
    schedule_horizon = (
        config.epochs
        if config.epochs > 0
        else max(config.warmup_epochs + config.patience * 8, 40)
    )
    span = max(schedule_horizon - config.warmup_epochs, 1)
    progress = min(max((epoch - config.warmup_epochs) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def load_dense_normalization(container: dict) -> dict:
    if "dense_feature_offsets" in container:
        return {
            "dense_feature_offsets": [
                float(value) for value in container.get("dense_feature_offsets", [])
            ],
            "dense_feature_scales": [
                float(value) for value in container.get("dense_feature_scales", [])
            ],
        }
    if "dense_feature_scales" in container:
        nested = container.get("dense_feature_scales")
        if isinstance(nested, dict):
            return {
                "dense_feature_offsets": [
                    float(value) for value in nested.get("dense_feature_offsets", [])
                ],
                "dense_feature_scales": [
                    float(value) for value in nested.get("dense_feature_scales", [])
                ],
            }
        return {
            "dense_feature_offsets": [],
            "dense_feature_scales": [float(value) for value in nested or []],
        }
    return {}


def resolve_training_dense_normalization(
    manifest: dict,
    feature_set_name: str,
) -> dict:
    base = load_dense_normalization(manifest)
    return {
        "dense_feature_offsets": resolve_dense_feature_offsets(feature_set_name, base),
        "dense_feature_scales": resolve_dense_feature_scales(feature_set_name, base),
    }


def dense_feature_names_for_model(feature_set_name: str, dense_input_count: int) -> list[str]:
    if dense_input_count <= 0:
        return []
    names = dense_feature_names_for_set(feature_set_name)
    if len(names) != dense_input_count:
        raise ValueError(
            f"feature set {feature_set_name} exposes {len(names)} dense features but model expects {dense_input_count}"
        )
    return names


def validate_training_config(config: TrainingConfig, backend_name: str = "jax_mlp") -> None:
    if backend_name != "jax_mlp":
        raise ValueError("the trainer is JAX-only")
    if config.feature_set != FEATURE_SET_NAME:
        raise ValueError(f"unsupported feature set {config.feature_set}")
    if config.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if config.epoch_size is not None and config.epoch_size <= 0:
        raise ValueError("epoch_size must be positive")
    if config.threads is not None and config.threads <= 0:
        raise ValueError("threads must be positive")
    if config.loader_workers is not None and config.loader_workers <= 0:
        raise ValueError("loader_workers must be positive")
    if config.learning_rate <= 0.0 or config.min_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    if config.screen_checkpoint_count < 0:
        raise ValueError("screen checkpoint count must be non-negative")
    if config.selection_interval <= 0:
        raise ValueError("selection_interval must be positive")
    if config.runtime_loss_interval <= 0:
        raise ValueError("runtime_loss_interval must be positive")
    if config.ema_decay is not None and not (0.0 < config.ema_decay < 1.0):
        raise ValueError("ema_decay must be in (0, 1)")
    if not config.nnue_cli and not os.environ.get(NNUE_CLI_ENV):
        raise ValueError("STEINBEISSER_NNUE_CLI is required for Rust dataset preparation/runtime validation")


class JaxMlpModel:
    backend_name = "jax_mlp"

    def __init__(
        self,
        architecture: list[int],
        sparse_input_count: int,
        dense_input_count: int,
        seed: int,
        weight_decay: float,
        ema_decay: float | None,
    ) -> None:
        if jax is None or jnp is None:
            raise RuntimeError("jax backend requested without jax")
        self.architecture = architecture
        self.sparse_input_count = sparse_input_count
        self.dense_input_count = dense_input_count
        self.weight_decay = float(weight_decay)
        self.ema_decay = ema_decay
        self.norm = "none"
        self.block_type = "plain"
        self.step = 0

        key = jax.random.PRNGKey(seed)
        weights = []
        biases = []
        for layer_index in range(len(self.architecture) - 1):
            key, subkey = jax.random.split(key)
            weights.append(
                jax.random.normal(
                    subkey,
                    (
                        self.architecture[layer_index],
                        self.architecture[layer_index + 1],
                    ),
                    dtype=jnp.float32,
                )
                * 0.05
            )
            biases.append(jnp.zeros((self.architecture[layer_index + 1],), dtype=jnp.float32))

        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.m_weights = tuple(jnp.zeros_like(weight) for weight in self.weights)
        self.v_weights = tuple(jnp.zeros_like(weight) for weight in self.weights)
        self.m_biases = tuple(jnp.zeros_like(bias) for bias in self.biases)
        self.v_biases = tuple(jnp.zeros_like(bias) for bias in self.biases)
        self.ema_weights = tuple(jnp.array(weight) for weight in self.weights) if ema_decay is not None else None
        self.ema_biases = tuple(jnp.array(bias) for bias in self.biases) if ema_decay is not None else None

        self._predict_fn = jax.jit(
            lambda weights, biases, inputs: _jax_predict_batch(weights, biases, inputs)
        )
        self._train_step_fn = jax.jit(
            lambda weights, biases, m_weights, v_weights, m_biases, v_biases, inputs, targets, sample_weights, learning_rate, weight_decay, step: _jax_train_step(
                weights,
                biases,
                m_weights,
                v_weights,
                m_biases,
                v_biases,
                inputs,
                targets,
                sample_weights,
                learning_rate,
                weight_decay,
                step,
            )
        )

    def predict_batch(self, inputs):
        inputs_array = jnp.asarray(inputs, dtype=jnp.float32)
        weights = self.ema_weights if self.ema_weights is not None else self.weights
        biases = self.ema_biases if self.ema_biases is not None else self.biases
        outputs = self._predict_fn(weights, biases, inputs_array)
        return np.asarray(jax.device_get(outputs), dtype=np.float32).tolist()

    def train_batch(self, inputs, targets, weights, learning_rate: float) -> float:
        inputs_array = jnp.asarray(inputs, dtype=jnp.float32)
        loss, new_weights, new_biases, new_m_weights, new_v_weights, new_m_biases, new_v_biases = self._train_step_fn(
            self.weights,
            self.biases,
            self.m_weights,
            self.v_weights,
            self.m_biases,
            self.v_biases,
            inputs_array,
            jnp.asarray(targets, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.asarray(float(learning_rate), dtype=jnp.float32),
            jnp.asarray(float(self.weight_decay), dtype=jnp.float32),
            jnp.asarray(self.step + 1, dtype=jnp.int32),
        )
        self.weights = new_weights
        self.biases = new_biases
        self.m_weights = new_m_weights
        self.v_weights = new_v_weights
        self.m_biases = new_m_biases
        self.v_biases = new_v_biases
        self.step += 1
        self._update_ema()
        return float(jax.device_get(loss))

    def export_state(self, feature_set_name: str, normalization: dict) -> dict:
        weights = self.ema_weights if self.ema_weights is not None else self.weights
        biases = self.ema_biases if self.ema_biases is not None else self.biases
        hidden_sizes = self.architecture[1:-1]
        first_layer = np.asarray(jax.device_get(weights[0]), dtype=np.float32)
        first_bias = np.asarray(jax.device_get(biases[0]), dtype=np.float32)
        export_weights = [np.asarray(jax.device_get(weight), dtype=np.float32) for weight in weights[1:]]
        export_biases = [np.asarray(jax.device_get(bias), dtype=np.float32) for bias in biases[1:]]
        return {
            "backend": self.backend_name,
            "feature_set": feature_set_name,
            "input_count": self.architecture[0],
            "input_count_sparse": self.sparse_input_count,
            "input_count_dense": self.dense_input_count,
            "dense_feature_names": dense_feature_names_for_model(
                feature_set_name, self.dense_input_count
            ),
            "dense_feature_offsets": [
                float(value) for value in normalization.get("dense_feature_offsets", [])
            ],
            "dense_feature_scales": resolve_dense_feature_scales(
                feature_set_name, normalization
            ),
            "target_transform": "linear_clip_v1",
            "activation": "relu",
            "norm": self.norm,
            "block_type": self.block_type,
            "hidden_sizes": hidden_sizes,
            "architecture": self.architecture,
            "w1_sparse": first_layer[: self.sparse_input_count].tolist(),
            "w1_dense": first_layer[self.sparse_input_count :].tolist(),
            "b1": first_bias.tolist(),
            **export_dense_layers(
                [weight.tolist() for weight in export_weights],
                [bias.tolist() for bias in export_biases],
                start_index=2,
            ),
        }

    def _update_ema(self) -> None:
        if self.ema_weights is None or self.ema_biases is None or self.ema_decay is None:
            return
        decay = jnp.asarray(float(self.ema_decay), dtype=jnp.float32)
        keep = jnp.asarray(1.0, dtype=jnp.float32) - decay
        self.ema_weights = tuple(
            decay * ema + keep * weight
            for ema, weight in zip(self.ema_weights, self.weights)
        )
        self.ema_biases = tuple(
            decay * ema + keep * bias
            for ema, bias in zip(self.ema_biases, self.biases)
        )








def export_dense_layers(weights: list, biases: list, start_index: int) -> dict:
    payload = {}
    for offset, (weight, bias) in enumerate(zip(weights, biases), start=start_index):
        if offset == start_index + len(weights) - 1:
            payload[f"w{offset}"] = weight if isinstance(weight[0], (float, int)) else [
                row[0] if len(row) == 1 else row for row in weight
            ]
            payload[f"b{offset}"] = bias[0] if len(bias) == 1 else bias
        else:
            payload[f"w{offset}"] = weight
            payload[f"b{offset}"] = bias
    return payload


def _jax_forward_batch(weights, biases, inputs):
    activations = inputs
    last_index = len(weights) - 1
    for layer_index, (weight_matrix, bias_vector) in enumerate(zip(weights, biases)):
        outputs = activations @ weight_matrix + bias_vector
        activations = outputs if layer_index == last_index else jax.nn.relu(outputs)
    return activations


def _jax_predict_batch(weights, biases, inputs):
    return _jax_forward_batch(weights, biases, inputs)[:, 0]




def _jax_huber_loss(errors):
    absolute = jnp.abs(errors)
    return jnp.where(
        absolute <= 1.0,
        0.5 * errors * errors,
        absolute - 0.5,
    )


def _jax_global_norm(leaves) -> object:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        total = total + jnp.sum(jnp.square(leaf))
    return jnp.sqrt(total)


def _jax_loss_and_grads(weights, biases, inputs, targets, sample_weights):
    def loss_fn(weight_params, bias_params):
        predictions = _jax_predict_batch(weight_params, bias_params, inputs)
        total_weight = jnp.maximum(jnp.sum(sample_weights), 1.0)
        losses = _jax_huber_loss(predictions - targets)
        return jnp.sum(sample_weights * losses) / total_weight

    return jax.value_and_grad(loss_fn, argnums=(0, 1))(weights, biases)


def _jax_train_step(
    weights,
    biases,
    m_weights,
    v_weights,
    m_biases,
    v_biases,
    inputs,
    targets,
    sample_weights,
    learning_rate,
    weight_decay,
    step,
):
    beta1 = jnp.asarray(0.9, dtype=jnp.float32)
    beta2 = jnp.asarray(0.999, dtype=jnp.float32)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)

    loss, (grad_weights, grad_biases) = _jax_loss_and_grads(
        weights,
        biases,
        inputs,
        targets,
        sample_weights,
    )
    grad_norm = _jax_global_norm((*grad_weights, *grad_biases))
    clip_scale = jnp.where(
        jnp.logical_and(grad_norm > GRADIENT_CLIP_NORM, grad_norm > 0.0),
        jnp.asarray(GRADIENT_CLIP_NORM, dtype=jnp.float32) / grad_norm,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    grad_weights = tuple(gradient * clip_scale for gradient in grad_weights)
    grad_biases = tuple(gradient * clip_scale for gradient in grad_biases)

    correction1 = 1.0 - jnp.power(beta1, step)
    correction2 = 1.0 - jnp.power(beta2, step)

    new_weights = []
    new_biases = []
    new_m_weights = []
    new_v_weights = []
    new_m_biases = []
    new_v_biases = []

    for weight_matrix, grad_matrix, m_matrix, v_matrix in zip(
        weights, grad_weights, m_weights, v_weights
    ):
        next_m = beta1 * m_matrix + (1.0 - beta1) * grad_matrix
        next_v = beta2 * v_matrix + (1.0 - beta2) * jnp.square(grad_matrix)
        m_hat = next_m / correction1
        v_hat = next_v / correction2
        next_weight = weight_matrix * (1.0 - learning_rate * weight_decay)
        next_weight = next_weight - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        new_weights.append(next_weight)
        new_m_weights.append(next_m)
        new_v_weights.append(next_v)

    for bias_vector, grad_vector, m_vector, v_vector in zip(
        biases, grad_biases, m_biases, v_biases
    ):
        next_m = beta1 * m_vector + (1.0 - beta1) * grad_vector
        next_v = beta2 * v_vector + (1.0 - beta2) * jnp.square(grad_vector)
        m_hat = next_m / correction1
        v_hat = next_v / correction2
        next_bias = bias_vector - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        new_biases.append(next_bias)
        new_m_biases.append(next_m)
        new_v_biases.append(next_v)

    return (
        loss,
        tuple(new_weights),
        tuple(new_biases),
        tuple(new_m_weights),
        tuple(new_v_weights),
        tuple(new_m_biases),
        tuple(new_v_biases),
    )








def state_layers_to_python(
    state: dict,
    architecture: list[int],
    sparse_input_count: int,
    dense_input_count: int,
) -> tuple[list[list[list[float]]], list[list[float]]]:
    validate_state_compatibility(state, architecture, sparse_input_count, dense_input_count)
    first_sparse = [[float(value) for value in row] for row in state["w1_sparse"]]
    first_dense = [[float(value) for value in row] for row in state.get("w1_dense", [])]
    weights: list[list[list[float]]] = [first_sparse + first_dense]
    biases: list[list[float]] = [[float(value) for value in state["b1"]]]
    for layer_index in range(2, len(architecture)):
        raw_weights = state[f"w{layer_index}"]
        raw_biases = state[f"b{layer_index}"]
        if layer_index == len(architecture) - 1:
            weights.append([[float(value)] for value in raw_weights])
            biases.append([float(raw_biases)])
        else:
            weights.append([[float(value) for value in row] for row in raw_weights])
            biases.append([float(value) for value in raw_biases])
    return weights, biases


def state_layers_to_numpy(
    state: dict,
    architecture: list[int],
    sparse_input_count: int,
    dense_input_count: int,
):
    if np is None:
        raise RuntimeError("numpy state load requested without numpy")
    weights, biases = state_layers_to_python(
        state, architecture, sparse_input_count, dense_input_count
    )
    return (
        [np.asarray(layer, dtype=np.float32) for layer in weights],
        [np.asarray(layer, dtype=np.float32) for layer in biases],
    )


def validate_state_compatibility(
    state: dict,
    architecture: list[int],
    sparse_input_count: int,
    dense_input_count: int,
) -> None:
    state_architecture = [int(value) for value in state.get("architecture", [])]
    if state_architecture and state_architecture != architecture:
        raise ValueError(
            f"model architecture {state_architecture} does not match requested {architecture}"
        )
    if int(state.get("input_count_sparse", sparse_input_count)) != sparse_input_count:
        raise ValueError("model sparse input count does not match requested feature set")
    if int(state.get("input_count_dense", dense_input_count)) != dense_input_count:
        raise ValueError("model dense input count does not match requested feature set")










def huber_loss_vector(errors):
    if np is None:
        return [huber_loss(float(error)) for error in errors]
    absolute = np.abs(errors)
    return np.where(
        absolute <= 1.0,
        0.5 * errors * errors,
        absolute - 0.5,
    )






def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required; run train or python3 nnue/train.py")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def trainer_main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("embedded trainer is env-driven; run train or python3 nnue/train.py")
    ensure_jax_loaded()
    if not JAX_AVAILABLE:
        raise SystemExit("jax is required for the Steinbeisser training recipe")

    config = TrainingConfig(
        train_path=required_env("STEINBEISSER_NNUE_TRAIN_PATH"),
        val_path=required_env("STEINBEISSER_NNUE_VAL_PATH"),
        manifest_path=required_env("STEINBEISSER_NNUE_MANIFEST_PATH"),
        output_dir=required_env("STEINBEISSER_NNUE_OUTPUT_DIR"),
        feature_set=required_env("STEINBEISSER_NNUE_FEATURE_SET"),
        architecture=required_env("STEINBEISSER_NNUE_ARCHITECTURE"),
        lambda_mix=0.99,
        epoch_size=None,
        epochs=env_int("STEINBEISSER_NNUE_EPOCHS", 100),
        batch_size=256,
        threads=env_int(
            "STEINBEISSER_NNUE_THREADS",
            env_int("STEINBEISSER_TRAIN_THREADS", 1),
        ),
        loader_workers=env_int(
            "STEINBEISSER_NNUE_LOADER_WORKERS",
            env_int("STEINBEISSER_LOADER_WORKERS", 1),
        ),
        dataset_cache_dir=required_env("STEINBEISSER_NNUE_DATASET_CACHE_DIR"),
        learning_rate=0.0019,
        min_learning_rate=0.00003,
        weight_decay=0.0007,
        warmup_epochs=5,
        patience=env_int("STEINBEISSER_NNUE_PATIENCE", 20),
        screen_checkpoint_count=env_int("STEINBEISSER_TRAIN_SCREEN_CHECKPOINTS", 3),
        runtime_loss_interval=1,
        ema_decay=0.9999,
        nnue_cli=required_env("STEINBEISSER_NNUE_CLI"),
    )
    print(json.dumps(run_training(config), indent=2))
    return 0
