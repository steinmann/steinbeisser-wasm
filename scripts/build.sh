#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${ROOT_DIR}/engine"
WEB_DIR="${ROOT_DIR}/web"
DIST_DIR="${ROOT_DIR}/dist"
PKG_DIR="${DIST_DIR}/pkg"
WASM_NAME="steinbeisser"
cd "${ROOT_DIR}"
BINDGEN_VERSION="$(python3 "${ROOT_DIR}/scripts/build_manifest.py" --bindgen-version)"
if [[ "$(wasm-bindgen --version)" != "wasm-bindgen ${BINDGEN_VERSION}" ]]; then
  echo "Install wasm-bindgen-cli ${BINDGEN_VERSION} to match engine/Cargo.lock" >&2
  exit 1
fi

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  TARGET_DIR="${CARGO_TARGET_DIR}"
else
  TARGET_DIR="${ROOT_DIR}/.build/cargo-target"
fi

if [[ "${TARGET_DIR}" != /* ]]; then
  TARGET_DIR="${ROOT_DIR}/${TARGET_DIR}"
fi

cargo build \
  --manifest-path "${ENGINE_DIR}/Cargo.toml" \
  --target-dir "${TARGET_DIR}" \
  --release \
  --locked \
  --target wasm32-unknown-unknown

rm -rf "${DIST_DIR}"
mkdir -p "${PKG_DIR}"

wasm-bindgen \
  --target web \
  --out-dir "${PKG_DIR}" \
  "${TARGET_DIR}/wasm32-unknown-unknown/release/${WASM_NAME}.wasm"

cp -R "${WEB_DIR}/." "${DIST_DIR}/"
BUILD_ID="$(python3 "${ROOT_DIR}/scripts/build_manifest.py" "${DIST_DIR}/build-manifest.json")"
python3 - <<'PY' "${DIST_DIR}/index.html" "${BUILD_ID}"
from pathlib import Path
import sys

index_path = Path(sys.argv[1])
build_id = sys.argv[2]
index_path.write_text(index_path.read_text().replace("__BUILD_ID__", build_id))
PY
touch "${DIST_DIR}/.nojekyll"

echo "Built ${DIST_DIR}"
