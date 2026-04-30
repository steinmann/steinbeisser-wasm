#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
ENGINE_DIR="${ROOT_DIR}/engine"
WEB_DIR="${ROOT_DIR}/web"
DIST_DIR="${ROOT_DIR}/dist"
PKG_DIR="${DIST_DIR}/pkg"
WASM_NAME="steinbeisser"
BUILD_ID="$(date -u +%Y%m%d%H%M%S)"

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  TARGET_DIR="${CARGO_TARGET_DIR}"
else
  TARGET_DIR="${WORKSPACE_DIR}/.build/steinbeisser-wasm"
fi

if [[ "${TARGET_DIR}" != /* ]]; then
  TARGET_DIR="${ROOT_DIR}/${TARGET_DIR}"
fi

cargo build \
  --manifest-path "${ENGINE_DIR}/Cargo.toml" \
  --target-dir "${TARGET_DIR}" \
  --release \
  --target wasm32-unknown-unknown

rm -rf "${DIST_DIR}"
mkdir -p "${PKG_DIR}"

wasm-bindgen \
  --target web \
  --out-dir "${PKG_DIR}" \
  "${TARGET_DIR}/wasm32-unknown-unknown/release/${WASM_NAME}.wasm"

cp -R "${WEB_DIR}/." "${DIST_DIR}/"
python3 - <<'PY' "${DIST_DIR}/index.html" "${BUILD_ID}"
from pathlib import Path
import sys

index_path = Path(sys.argv[1])
build_id = sys.argv[2]
index_path.write_text(index_path.read_text().replace("__BUILD_ID__", build_id))
PY
touch "${DIST_DIR}/.nojekyll"

echo "Built ${DIST_DIR}"
