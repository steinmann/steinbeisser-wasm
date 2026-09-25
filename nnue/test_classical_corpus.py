"""Leakage and retry tests for Classical corpus assembly."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from nnue.classical_corpus import MAGIC, RECORD_BYTES, assemble, records


def sample(key: int) -> bytes:
    return bytes([key]) * 25 + bytes(RECORD_BYTES - 25)


def write_samples(path: Path, keys: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + b"".join(sample(key) for key in keys))


class ClassicalCorpusTests(unittest.TestCase):
    def test_exact_keys_are_excluded_and_retries_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, heldout, output = (root / name for name in ("train", "heldout", "out"))
            write_samples(train / "train.sbin", [1, 2, 3])
            (train / "manifest.json").write_text(json.dumps({"feature_set": "test"}))
            write_samples(heldout / "val.sbin", [1, 4, 4])
            write_samples(heldout / "train.sbin", [2, 5, 6])
            first = assemble(train, heldout, output, 2)
            self.assertEqual(first["train_samples"], 3)
            self.assertEqual(first["validation_samples"], 2)
            self.assertEqual(first["shared_exact_keys"], 0)
            self.assertEqual([row[0] for row in records(output / "val.sbin")], [4, 5])
            self.assertEqual(assemble(train, heldout, output, 2), first)

    def test_short_heldout_does_not_publish_partial_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, heldout, output = (root / name for name in ("train", "heldout", "out"))
            write_samples(train / "train.sbin", [1])
            write_samples(heldout / "val.sbin", [1, 2])
            write_samples(heldout / "train.sbin", [1, 2])
            with self.assertRaisesRegex(ValueError, "only 1 leakage-free"):
                assemble(train, heldout, output, 2)
            self.assertFalse((output / "val.sbin").exists())
            self.assertFalse((output / "val.sbin.pending").exists())


if __name__ == "__main__":
    unittest.main()
