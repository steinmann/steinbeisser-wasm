#!/usr/bin/env python3
"""Content identity for the deployed engine, model and frontend (including dirty edits)."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest(output_dir):
    files = [p for directory in ('engine/src', 'web', 'scripts')
             for p in (ROOT / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT / p for p in (
        'engine/Cargo.toml', 'engine/Cargo.lock', 'rust-toolchain.toml', '.cargo/config.toml',
    )]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    artifacts = {str(p.relative_to(output_dir)): digest(p)
                 for p in sorted((output_dir / 'pkg').rglob('*')) if p.is_file()}
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    identity = hashlib.sha256(json.dumps({'sources': sources, 'artifacts': artifacts},
                                       sort_keys=True).encode()).hexdigest()
    return {'version': 1, 'build_id': identity[:20], 'source_sha256': sources,
            'artifact_sha256': artifacts, 'commit': commit,
            'model_sha256': digest(ROOT / 'engine/src/net.mlp'),
            'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
            'wasm_bindgen': subprocess.check_output(['wasm-bindgen', '--version'], text=True).strip()}

def bindgen_version():
    lock = tomllib.loads((ROOT / 'engine/Cargo.lock').read_text())
    return next(p['version'] for p in lock['package'] if p['name'] == 'wasm-bindgen')

if __name__ == '__main__':
    if sys.argv[1:] == ['--bindgen-version']:
        print(bindgen_version())
    else:
        output = Path(sys.argv[1])
        result = manifest(output.parent)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(result['build_id'])
