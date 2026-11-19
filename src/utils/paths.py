from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def resolve_project_root(project_root: Optional[str] = None) -> Path:
    """Resolve project root safely.

    Default root is inferred from this file location:
    src/utils/paths.py -> project root at parents[2].
    """
    if project_root:
        return Path(project_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_paths(project_root: Optional[str] = None) -> Dict[str, Path]:
    """Return commonly used canonical paths."""
    root = resolve_project_root(project_root)
    return {
        "project_root": root,
        "data_raw": root / "data" / "raw",
        "data_metadata": root / "data" / "metadata",
        "data_sample": root / "data" / "sample",
        "outputs": root / "outputs",
        "runs": root / "runs",
    }


def get_data_raw_dir(project_root: Optional[str] = None) -> Path:
    return resolve_paths(project_root)["data_raw"]


def get_data_metadata_dir(project_root: Optional[str] = None) -> Path:
    return resolve_paths(project_root)["data_metadata"]


def get_data_sample_dir(project_root: Optional[str] = None) -> Path:
    return resolve_paths(project_root)["data_sample"]


def get_outputs_dir(project_root: Optional[str] = None) -> Path:
    return resolve_paths(project_root)["outputs"]


def get_runs_dir(project_root: Optional[str] = None) -> Path:
    return resolve_paths(project_root)["runs"]


def get_phase1_dataset_contract_dir(project_root: Optional[str] = None) -> Path:
    return get_outputs_dir(project_root) / "phase1_dataset_contract"


def get_phase1_nnunet_output_dir(project_root: Optional[str] = None) -> Path:
    return get_outputs_dir(project_root) / "phase1_nnunet_dataset"


def check_dir(path: Path, label: str) -> tuple[bool, str]:
    """Non-mutating directory existence check with readable message."""
    if path.exists() and path.is_dir():
        return True, f"{label}: OK ({path})"
    return False, f"{label}: MISSING ({path})"
