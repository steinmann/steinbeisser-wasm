"""Release-asset integrity checks for the two v2.7 networks."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import unittest

from nnue.model_assets import encoded_model


ROOT = Path(__file__).resolve().parent.parent


class ModelAssetTests(unittest.TestCase):
    def test_versioned_models_match_manifest_and_embedded_assets(self) -> None:
        manifest = json.loads((ROOT / "data/models/v27/manifest.json").read_text())
        for name, asset in (
            ("general", "net.mlp"),
            ("classical", "net-classical.mlp"),
        ):
            with self.subTest(name=name):
                model = ROOT / f"data/models/v27/{name}.nnq"
                payload = model.read_bytes()
                text = (ROOT / "engine/src" / asset).read_text(encoding="ascii")
                self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest[name]["sha256"])
                self.assertEqual(text, encoded_model(model))
                self.assertEqual(base64.a85decode(text.strip()), payload)


if __name__ == "__main__":
    unittest.main()
