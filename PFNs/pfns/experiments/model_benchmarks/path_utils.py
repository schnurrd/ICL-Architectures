from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path) -> Path:
    """Walk upward from `start` until the ICL-Architectures repo root is found."""
    start_path = Path(start).resolve()
    candidates = (start_path, *start_path.parents)
    for path in candidates:
        if (path / ".git").exists() and (path / "PFNs").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {start_path}.")


def build_repo_output_root(
    start: str | Path,
    *parts: str,
) -> Path:
    return find_repo_root(start) / "exp_outputs" / Path(*parts)
