#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${ROOT_DIR}/engine"
DIST_DIR="${ROOT_DIR}/dist"
PKG_DIR="${DIST_DIR}/pkg"
WASM_NAME="steinbeisser_wasm_engine"
BUILD_ID="$(date -u +%Y%m%d%H%M%S)"

cargo build \
  --manifest-path "${ENGINE_DIR}/Cargo.toml" \
  --release \
  --target wasm32-unknown-unknown

rm -rf "${DIST_DIR}"
mkdir -p "${PKG_DIR}"

wasm-bindgen \
  --target web \
  --out-dir "${PKG_DIR}" \
  "${ENGINE_DIR}/target/wasm32-unknown-unknown/release/${WASM_NAME}.wasm"

cp "${ROOT_DIR}"/web/* "${DIST_DIR}/"
python3 - <<'PY' "${DIST_DIR}/index.html" "${BUILD_ID}"
from pathlib import Path
import sys

index_path = Path(sys.argv[1])
build_id = sys.argv[2]
index_path.write_text(index_path.read_text().replace("__BUILD_ID__", build_id))
PY
touch "${DIST_DIR}/.nojekyll"
