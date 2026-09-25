"""Check that versioned NNQ models can initialize the fine-tuning trainer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nnue import fit as fit_module
from nnue.fit import (
    JaxMlpModel,
    ensure_jax_loaded,
    load_dense_normalization,
    read_quantized_runtime_model,
    write_quantized_runtime_model,
)


ROOT = Path(__file__).resolve().parent.parent


class InitialModelTest(unittest.TestCase):
    def test_general_model_roundtrip_through_trainer_weights(self):
        ensure_jax_loaded()
        if not fit_module.JAX_AVAILABLE:
            self.skipTest("JAX is not installed in this Python environment")
        original = ROOT / "data/models/v27/general.nnq"
        state = read_quantized_runtime_model(original)
        normalization = load_dense_normalization(state)
        model = JaxMlpModel(
            state["architecture"],
            state["input_count_sparse"],
            state["input_count_dense"],
            seed=1,
            weight_decay=0.0007,
            ema_decay=0.9999,
        )
        model.load_initial_state(state, state["feature_set"], normalization)
        reconstructed = model.export_state(state["feature_set"], normalization)
        reconstructed["runtime_activation_scales"] = state["runtime_activation_scales"]
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "roundtrip.nnq"
            write_quantized_runtime_model(reconstructed, candidate)
            self.assertEqual(original.read_bytes(), candidate.read_bytes())


if __name__ == "__main__":
    unittest.main()
