from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnue"))
import fit
import train
from run_identity import implementation_sources, preserve_manifest


def trainer_env(output="output"):
    return {"STEINBEISSER_NNUE_" + name: value for name, value in {
        "TRAIN_PATH": "train.bin", "VAL_PATH": "val.bin", "MANIFEST_PATH": "manifest.json",
        "OUTPUT_DIR": output, "FEATURE_SET": fit.FEATURE_SET_NAME, "ARCHITECTURE": "130,84,1",
        "DATASET_CACHE_DIR": "cache", "CLI": "nnue", "EPOCHS": "3", "THREADS": "2",
    }.items()}


class TrainingInterfaces(unittest.TestCase):
    def test_screen_match_rejects_forfeit_even_with_complete_wdl(self):
        payload = {
            "wins": 0, "draws": 1000, "losses": 0,
            "elo": 0.0, "elo_lower": -1.0, "elo_upper": 1.0,
            "forfeit": True,
        }
        with patch.object(train, "run_json_command", return_value=payload):
            with self.assertRaisesRegex(SystemExit, "forfeit"):
                train.run_selfplay_match(Path("candidate"), Path("baseline"))

    def test_v27_finish_uses_random_only_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = train.RunState(screened_candidates=[{"id": "screened"}])
            summary = {
                "winner": "candidate-1",
                "standings": [{
                    "player": "candidate-1", "elo_vs_release": 2.0,
                    "games": 7500, "wins": 20, "draws": 7470, "losses": 10,
                }],
                "release_gate": {
                    "games": 10000, "wins": 30, "draws": 9950,
                    "losses": 20, "elo": 0.3, "elo_95_ci": "0.1,0.5",
                },
                "release_gate_passed": True,
            }
            with patch.object(train, "WORK_DIR", root), patch.object(train, "RUN_ROOT", root), patch.object(train, "prepare_tournament_openings", return_value=(root / "tournament.fen", [])) as tournament, patch.object(train, "load_unique_book_openings", return_value=["fen"] * 5000) as openings, patch("nnue.general_qualify.qualify", return_value=summary) as qualify, patch.object(state, "save"), patch.object(train, "emit_tournament_table"), patch.object(train, "emit"):
                train.finish_training(state)
            tournament.assert_called_once_with(train.OPENING_CONFIG, 250)
            self.assertEqual(openings.call_args.kwargs["skip"], 750)
            self.assertEqual(qualify.call_args.kwargs["tournament_openings"], root / "tournament.fen")
            self.assertTrue(state.tournament_completed)
            self.assertTrue(state.tournament_summary["release_gate_passed"])

    def test_config_is_explicit_and_roundtrips(self):
        before = dict(os.environ)
        config = fit.training_config_from_env(trainer_env())
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.threads, 2)
        self.assertEqual(config, fit.TrainingConfig(**json.loads(json.dumps(dataclasses.asdict(config)))))
        with self.assertRaises(ValueError):
            fit.training_config_from_env({})

    def test_trainer_subprocess_never_mutates_parent_environment_or_streams(self):
        for status in (0, 7):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                child_env = trainer_env(str(root / "output with spaces"))
                before_env = dict(os.environ)
                before_streams = (sys.stdout, sys.stderr)

                def child(command, **kwargs):
                    self.assertEqual(command[:2], [sys.executable, str(train.NNUE_MODULE_DIR / "fit.py")])
                    self.assertEqual(command[2], "--config")
                    self.assertEqual(kwargs["env"], child_env)
                    self.assertIsNot(kwargs["env"], child_env)
                    self.assertEqual(dict(os.environ), before_env)
                    self.assertEqual((sys.stdout, sys.stderr), before_streams)
                    config = fit.TrainingConfig(**json.loads(Path(command[3]).read_text()))
                    self.assertEqual(config, fit.training_config_from_env(child_env))
                    kwargs["stdout"].write("child diagnostic\n")
                    return subprocess.CompletedProcess(command, status)

                with patch.object(train, "LOG_PATH", root / "train.log"), patch.object(train.subprocess, "run", side_effect=child):
                    if status:
                        with self.assertRaisesRegex(SystemExit, "status 7.*train.log"):
                            train.run_trainer(child_env)
                    else:
                        train.run_trainer(child_env)
                self.assertEqual(dict(os.environ), before_env)
                self.assertEqual((sys.stdout, sys.stderr), before_streams)
                self.assertIn("child diagnostic", (root / "train.log").read_text())

    def test_config_cli_does_not_require_environment_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = fit.training_config_from_env(trainer_env())
            path.write_text(json.dumps(dataclasses.asdict(config)))
            with patch.object(fit, "run_training", return_value={}) as run, patch.object(fit, "training_config_from_env", side_effect=AssertionError("explicit config")), patch("builtins.print"):
                self.assertEqual(fit.trainer_main(["--config", str(path)]), 0)
                run.assert_called_once_with(config)

    def test_feature_schema_rejects_inconsistent_dense_dimensions(self):
        schema = json.loads((fit.NNUE_DIR / "feature-schema.json").read_text())
        for key, value in (("dense_feature_names", []), ("dense_feature_scales", [1]), ("dense_feature_scales", [float("nan")] * 8)):
            with self.subTest(key=key, value=value), patch.object(fit, "read_json", return_value={**schema, key: value}):
                with self.assertRaises(ValueError):
                    fit.current_feature_schema()

    def test_saved_implementation_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "implementation.json"
            original_manifest = {"reference_commit": "a", "sources": {"fit.py": "b"}}
            preserve_manifest(path, original_manifest)
            original = path.read_bytes()
            preserve_manifest(path, original_manifest)
            with self.assertRaises(RuntimeError):
                preserve_manifest(path, {"reference_commit": "changed"})
            self.assertEqual(path.read_bytes(), original)

    def test_manifest_publication_is_atomic_and_concurrent_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "implementation.json"
            manifest = {"sources": {"fit.py": "b"}}
            with patch("run_identity.os.link", side_effect=OSError("injected interrupted publication")):
                with self.assertRaises(OSError):
                    preserve_manifest(path, manifest)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(lambda _: preserve_manifest(path, manifest), range(16)))
            self.assertEqual(json.loads(path.read_text()), manifest)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_progressed_run_without_provenance_is_not_blessed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "implementation.json"
            with self.assertRaisesRegex(RuntimeError, "restore its original implementation manifest"):
                preserve_manifest(path, {"sources": {"fit.py": "current"}}, require_existing=True)
            self.assertFalse(path.exists())

    def test_provenance_tracks_build_inputs_and_ignores_generated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = ("nnue/fit.py", "nnue/src/core.rs", "nnue/feature-schema.json", "engine/Cargo.toml", "engine/Cargo.lock", "engine/src/net.mlp", "rust-toolchain.toml", ".cargo/config.toml")
            for name in inputs + ("nnue/output/model.json", "nnue/__pycache__/cache.pyc"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            original = implementation_sources(root)
            self.assertEqual(set(original), set(inputs))
            (root / "nnue/output/model.json").write_text("changed output")
            self.assertEqual(implementation_sources(root), original)
            (root / "engine/Cargo.toml").write_text("changed dependency")
            self.assertNotEqual(implementation_sources(root), original)


class TrainingResume(unittest.TestCase):
    def resolve(self, root, saved=None, manifest=None):
        state = root / "state.json"
        if saved is not None:
            state.write_text(json.dumps(saved))
        if manifest is not None:
            (root / "implementation-manifest.json").write_text(json.dumps(manifest))
        resolved = subprocess.CompletedProcess([], 0, "a" * 40 + "\n", "")
        with patch.object(train, "STATE_PATH", state), patch.object(train, "RUN_ROOT", root), patch.object(train, "REFERENCE_REF", ""), patch.object(train, "REFERENCE_COMMIT", ""), patch.object(train, "REFERENCE_BIN", Path()), patch.object(train, "resolve_reference_ref", side_effect=AssertionError("must not look up latest")), patch.object(train.subprocess, "run", return_value=resolved) as command, patch.object(train, "configure_training_dirs"):
            train.configure_release_ref()
            self.assertIn("a" * 40 + "^{commit}", command.call_args.args[0])
            self.assertEqual(train.REFERENCE_REF, "v2.5")

    def test_saved_reference_does_not_resolve_latest_release(self):
        with tempfile.TemporaryDirectory() as directory:
            self.resolve(Path(directory), {"reference_ref": "v2.5", "reference_commit": "a" * 40})

    def test_legacy_signature_still_supplies_immutable_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.resolve(Path(directory), {"reference_ref": "v2.5", "run_signature": {"reference_commit": "a" * 40}})

    def test_manifest_pins_interrupted_first_cycle_and_setup_failure(self):
        for saved in (None, {"last_error": "setup interrupted"}):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as directory:
                self.resolve(Path(directory), saved, {"reference_ref": "v2.5", "reference_commit": "a" * 40})

    def test_missing_mutable_or_conflicting_commit_refuses_resume(self):
        for commit in (None, "v2.5", "HEAD", "b" * 40):
            with self.subTest(commit=commit), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    self.resolve(Path(directory), {"reference_ref": "v2.5", "reference_commit": commit}, {"reference_ref": "v2.5", "reference_commit": "a" * 40})

    def test_first_cycle_configuration_is_validated_on_resume(self):
        state = train.RunState(cycle=0, run_signature={"recipe": "saved"})
        with patch.object(train.RunState, "load", return_value=state), patch.object(train, "current_run_signature", return_value={"recipe": "changed"}):
            with self.assertRaisesRegex(SystemExit, "recipe"):
                train.load_state()


if __name__ == "__main__":
    unittest.main()
