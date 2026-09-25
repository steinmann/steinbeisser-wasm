"""Tournament arithmetic and ranking checks without running matches."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from nnue import general_qualify
from nnue.general_qualify import CANDIDATES, GAMES_PER_MATCHUP, summarize


class GeneralQualificationTests(unittest.TestCase):
    def test_sixteen_entrants_each_play_7500_games(self) -> None:
        selected = [{"id": f"candidate-{number}"} for number in range(CANDIDATES)]
        players = [row["id"] for row in selected] + ["reference:v2.6.1"]
        search = {"moves": 100, "nodes": 100_000, "engine_ms": 50, "depth_sum": 900}
        rows = [
            {
                "left": left,
                "right": right,
                "result": {
                    "wins": 0, "draws": GAMES_PER_MATCHUP, "losses": 0,
                    "search": {"local": search, "github": search},
                },
            }
            for index, left in enumerate(players)
            for right in players[index + 1:]
        ]
        self.assertEqual(len(rows), 120)
        standings = summarize(selected, "v2.6.1", rows)
        self.assertEqual(len(standings), 16)
        self.assertTrue(all(row["games"] == 7_500 for row in standings))
        self.assertTrue(all(row["elo_vs_release"] == 0 for row in standings))
        self.assertTrue(all(row["avg_depth"] == 9 for row in standings))
        self.assertTrue(all(row["avg_nps"] == 2_000_000 for row in standings))

    def test_qualification_runs_all_pairings_then_the_independent_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "release"
            reference.write_bytes(b"release")
            tournament, gate = root / "tournament.fen", root / "gate.fen"
            tournament.write_text("tournament")
            gate.write_text("gate")
            selected = []
            for number in range(CANDIDATES):
                model, binary = root / f"model-{number}.nnq", root / f"candidate-{number}-bin"
                model.write_bytes(bytes([number]))
                binary.write_bytes(bytes([number]))
                selected.append({
                    "id": f"candidate-{number}", "model": str(model), "source_bin": str(binary),
                    "model_sha256": general_qualify.digest(model),
                    "source_bin_sha256": general_qualify.digest(binary),
                })

            def fake_match(_repo, _selfplay, _reference, candidate, _openings, games, _ms, _parallel, _seed, _log):
                wins = games if candidate.name == "candidate-0-bin" else 0
                return {
                    "wins": wins, "draws": games - wins, "losses": 0,
                    "games": games, "elo": 0, "elo_95_ci": "0,0",
                    "search": {
                        side: {"moves": 100, "nodes": 100_000, "engine_ms": 50, "depth_sum": 900}
                        for side in ("local", "github")
                    },
                }

            with patch.object(general_qualify, "select", return_value=selected), patch.object(general_qualify, "match", side_effect=fake_match) as played:
                report = general_qualify.qualify(
                    [], repo=root, run_root=root, reference_ref="v2.6.1",
                    reference_bin=reference, selfplay_bin=reference,
                    tournament_openings=tournament, gate_openings=gate,
                )
            self.assertEqual(played.call_count, 121)
            self.assertEqual(report["pairings"], 120)
            self.assertEqual(report["winner"], "candidate-0")
            self.assertTrue(report["release_gate_passed"])
            self.assertEqual((Path(report["output_dir"]) / "winner.nnq").read_bytes(), b"\0")


if __name__ == "__main__":
    unittest.main()
