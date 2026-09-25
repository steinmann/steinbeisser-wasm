"""Export the two versioned NNQ networks to engine-embedded ASCII85 assets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def encoded_model(model: Path) -> str:
    payload = model.read_bytes()
    if not payload.startswith(b"NNQ1"):
        raise ValueError(f"not an NNQ model: {model}")
    return base64.a85encode(payload).decode("ascii") + "\n"


def export_model(model: Path, asset: Path, *, check: bool) -> str:
    expected = encoded_model(model)
    if check:
        if asset.read_text(encoding="ascii") != expected:
            raise ValueError(f"embedded asset does not match {model}: {asset}")
    else:
        asset.write_text(expected, encoding="ascii")
    decoded = base64.a85decode(asset.read_text(encoding="ascii").strip())
    if decoded != model.read_bytes():
        raise ValueError(f"ASCII85 roundtrip failed for {asset}")
    return hashlib.sha256(decoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    manifest = json.loads((root / "data/models/v27/manifest.json").read_text())
    for label, model, asset in (
        ("general", root / "data/models/v27/general.nnq", root / "engine/src/net.mlp"),
        ("classical", root / "data/models/v27/classical.nnq", root / "engine/src/net-classical.mlp"),
    ):
        expected_digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if expected_digest != manifest[label]["sha256"]:
            raise ValueError(f"{label} model differs from the release manifest")
        digest = export_model(model, asset, check=args.check)
        print(f"{label} model_sha256={digest} asset={asset}")


if __name__ == "__main__":
    main()
