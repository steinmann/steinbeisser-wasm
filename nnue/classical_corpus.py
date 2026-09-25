"""Build a Classical finetuning corpus with an independent validation split.

The two input corpora must come from separate self-play streams. Validation
records are accepted only when their exact board-and-side key was absent from
training and has not already been accepted into validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


MAGIC = b"SBSMP01\n"
RECORD_BYTES = 70
KEY_BYTES = 25


def records(path: Path):
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"invalid sample header: {path}")
        while row := source.read(RECORD_BYTES):
            if len(row) != RECORD_BYTES:
                raise ValueError(f"truncated sample record: {path}")
            yield row


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def assemble(train_corpus: Path, heldout_corpus: Path, destination: Path, val_samples: int) -> dict:
    if val_samples <= 0:
        raise ValueError("validation sample count must be positive")
    source_train = train_corpus / "train.sbin"
    train_keys = {row[:KEY_BYTES] for row in records(source_train)}
    train_count = (source_train.stat().st_size - len(MAGIC)) // RECORD_BYTES
    destination.mkdir(parents=True, exist_ok=True)
    val_keys: set[bytes] = set()
    excluded = 0
    val_path = destination / "val.sbin"
    temporary_val = destination / "val.sbin.pending"
    try:
        with temporary_val.open("wb") as output:
            output.write(MAGIC)
            for source in (heldout_corpus / "val.sbin", heldout_corpus / "train.sbin"):
                for row in records(source):
                    key = row[:KEY_BYTES]
                    if key in train_keys or key in val_keys:
                        excluded += 1
                        continue
                    output.write(row)
                    val_keys.add(key)
                    if len(val_keys) == val_samples:
                        break
                if len(val_keys) == val_samples:
                    break
        if len(val_keys) != val_samples:
            raise ValueError(f"only {len(val_keys)} leakage-free validation samples available")
        temporary_val.replace(val_path)
    finally:
        temporary_val.unlink(missing_ok=True)

    shutil.copyfile(source_train, destination / "train.sbin")
    manifest = json.loads((train_corpus / "manifest.json").read_text())
    manifest["train"] = {"file": "train.sbin", "samples": train_count}
    manifest["val"] = {"file": "val.sbin", "samples": val_samples}
    manifest["corpus_dir"] = str(destination.resolve())
    manifest["canonical_corpus_dir"] = str(destination.resolve())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "train_samples": train_count,
        "validation_samples": val_samples,
        "excluded_validation_rows": excluded,
        "shared_exact_keys": len(train_keys & val_keys),
        "train_sha256": digest(destination / "train.sbin"),
        "val_sha256": digest(val_path),
        "manifest_sha256": digest(destination / "manifest.json"),
    }
    (destination / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-samples", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(assemble(args.train, args.heldout, args.out, args.val_samples), indent=2))


if __name__ == "__main__":
    main()
