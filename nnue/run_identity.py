"""Immutable, content-addressed implementation provenance for training resumes."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def implementation_sources(repo: Path) -> dict[str, str]:
    # Enumerate implementation roots, not run outputs/caches that a user may
    # place beneath nnue/. Build configuration and the evaluator are inputs too.
    paths = set((repo / "nnue").glob("*.py"))
    paths.update((repo / "nnue/src").rglob("*.rs"))
    paths.update((repo / "engine/src").rglob("*.rs"))
    paths.update((repo / "engine/src").rglob("*.mlp"))
    paths.update(repo / name for name in (
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml",
        "nnue/Cargo.toml", "nnue/Cargo.lock", "nnue/feature-schema.json",
        "engine/Cargo.toml", "engine/Cargo.lock", "engine/build.rs",
    ))
    return {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    }


def implementation_manifest(repo: Path, reference_ref: str, reference_commit: str) -> dict:
    dependencies = {}
    for package in ('numpy', 'jax', 'jaxlib', 'torch', 'tqdm'):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    toolchains = {
        command: subprocess.run(
            [command, '--version'], cwd=repo, check=True, capture_output=True,
            text=True, timeout=30,
        ).stdout.strip()
        for command in ('rustc', 'cargo')
    }
    return {'reference_ref': reference_ref, 'reference_commit': reference_commit,
            'python': sys.version, 'toolchains': toolchains,
            'dependencies': dependencies, 'sources': implementation_sources(repo)}


def preserve_manifest(path: Path, manifest: dict, *, require_existing: bool = False) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + '\n'

    def check_existing():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Training implementation changed since this run started; use a new run directory. Saved provenance has not been overwritten.')

    if path.exists():
        check_existing()
        return
    if require_existing:
        raise RuntimeError('Saved training state has no implementation provenance; restore its original implementation manifest or use a fresh run directory. Existing artifacts have not been changed.')
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fully synced temporary file is linked into place atomically and
    # exclusively: a crash cannot publish half a manifest, and competing starts
    # cannot overwrite one another. Leftover temporary files are not authority.
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            check_existing()
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
