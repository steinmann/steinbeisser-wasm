"""Qualification safety checks without launching games."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from nnue.classical_train import read_match


class ClassicalTrainTests(unittest.TestCase):
    def test_match_requires_complete_zero_error_non_stopped_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "match.log"
            fields = {
                "games": "4", "w_d_l": "2-1-1", "elo": "+8.0",
                "elo_95_ci": "-2.0,18.0", "early_stopped": "false",
                "github_illegal": "0", "github_errors": "0",
                "local_illegal": "0", "local_errors": "0",
            }

            def write() -> None:
                log.write_text("".join(f"{key},{value}\n" for key, value in fields.items()))

            write()
            self.assertEqual(read_match(log, 4)["wins"], 2)
            fields["games"] = "3"
            write()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_match(log, 4)
            fields["games"] = "4"
            fields["local_errors"] = "1"
            write()
            with self.assertRaisesRegex(ValueError, "engine error"):
                read_match(log, 4)
            fields["local_errors"] = "0"
            fields["early_stopped"] = "true"
            write()
            with self.assertRaisesRegex(ValueError, "early stop"):
                read_match(log, 4)


if __name__ == "__main__":
    unittest.main()
