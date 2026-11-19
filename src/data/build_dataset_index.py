from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Set

from src.utils.paths import get_data_raw_dir, resolve_project_root

PATIENT_PATTERN = re.compile(r"^PatientID_\d+$")
TIMEPOINT_PATTERN = re.compile(r"^Timepoint_\d+$")
NIFTI_PATTERN = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
PATIENT_EXTRACT_PATTERN = re.compile(r"patientid[_\-]?(\d{1,6})", re.IGNORECASE)
TIMEPOINT_EXTRACT_PATTERN = re.compile(r"timepoint[_\-]?(\d{1,2})", re.IGNORECASE)
MODALITY_TOKEN_PATTERN = re.compile(r"brain_([a-z0-9]+)", re.IGNORECASE)

EXPECTED_MODALITIES = {"t1c", "t1n", "t2w", "t2f"}
MASK_HINTS = ("mask", "tumor", "seg", "label")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Phase 1B raw canonical index from data/raw only."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional raw data directory override. Default: <project_root>/data/raw",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview canonical index summary without writing CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/phase1_dataset_contract/raw_canonical_index.csv",
        help="Raw canonical index CSV output path. Ignored when --dry-run is set.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=8,
        help="Number of rows to show in dry-run preview.",
    )
    return parser


def _resolve_output_path(project_root: Path, output_path: str) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _normalize_patient_id(value: str) -> str:
    match = PATIENT_EXTRACT_PATTERN.search(str(value).strip())
    if not match:
        return str(value).strip()
    return f"PatientID_{match.group(1).zfill(4)}"


def _normalize_timepoint_id(value: str) -> str:
    match = TIMEPOINT_EXTRACT_PATTERN.search(str(value).strip())
    if not match:
        return str(value).strip()
    return f"Timepoint_{int(match.group(1))}"


def _guess_modalities(file_names: List[str]) -> List[str]:
    modalities: Set[str] = set()
    for file_name in file_names:
        for token in MODALITY_TOKEN_PATTERN.findall(file_name):
            modalities.add(token.lower())
    return sorted(modalities)


def _guess_mask_presence(file_names: List[str]) -> bool:
    lowered = [name.lower() for name in file_names]
    return any(any(hint in name for hint in MASK_HINTS) for name in lowered)


def _has_unexpected_extension(file_names: List[str]) -> bool:
    return any(not NIFTI_PATTERN.search(name) for name in file_names)


def _has_unexpected_modality(modalities: List[str]) -> bool:
    return any(modality not in EXPECTED_MODALITIES for modality in modalities)


def _collect_global_structure_anomalies(raw_dir: Path) -> dict:
    top_entries = list(raw_dir.iterdir())
    top_level_unexpected_dirs = [
        p.name for p in top_entries if p.is_dir() and not PATIENT_PATTERN.match(p.name)
    ]
    top_level_files = [p.name for p in top_entries if p.is_file()]
    return {
        "top_level_unexpected_dirs": sorted(top_level_unexpected_dirs),
        "top_level_files": sorted(top_level_files),
    }


def build_index_rows(raw_dir: Path) -> List[Dict[str, object]]:
    anomalies = _collect_global_structure_anomalies(raw_dir)
    root_has_structure_issue = bool(anomalies["top_level_unexpected_dirs"] or anomalies["top_level_files"])

    rows: List[Dict[str, object]] = []
    patient_dirs = sorted(
        [p for p in raw_dir.iterdir() if p.is_dir() and PATIENT_PATTERN.match(p.name)],
        key=lambda p: p.name,
    )

    for patient_dir in patient_dirs:
        patient_entries = sorted(patient_dir.iterdir(), key=lambda p: p.name.lower())
        patient_non_timepoint_entries = [
            p for p in patient_entries if not (p.is_dir() and TIMEPOINT_PATTERN.match(p.name))
        ]
        patient_has_structure_issue = bool(patient_non_timepoint_entries)

        timepoint_dirs = [
            p for p in patient_entries if p.is_dir() and TIMEPOINT_PATTERN.match(p.name)
        ]
        for timepoint_dir in sorted(timepoint_dirs, key=lambda p: p.name):
            tp_entries = sorted(timepoint_dir.iterdir(), key=lambda p: p.name.lower())
            files = [p for p in tp_entries if p.is_file()]
            subdirs = [p for p in tp_entries if p.is_dir()]
            file_names = [p.name for p in files]

            nii_count = sum(1 for name in file_names if NIFTI_PATTERN.search(name))
            modalities = _guess_modalities(file_names)
            has_mask_guess = _guess_mask_presence(file_names)

            quality_flag_missing_file = (len(file_names) != 5) or (nii_count != 5)
            quality_flag_missing_mask = not has_mask_guess

            quality_flag_unexpected_structure = (
                root_has_structure_issue
                or patient_has_structure_issue
                or bool(subdirs)
                or _has_unexpected_extension(file_names)
                or _has_unexpected_modality(modalities)
            )

            quality_flag_ready_for_nnunet_candidate = not (
                quality_flag_missing_file
                or quality_flag_missing_mask
                or quality_flag_unexpected_structure
            )

            rows.append(
                {
                    "raw_patient_id": patient_dir.name,
                    "raw_timepoint_id": timepoint_dir.name,
                    "normalized_patient_id": _normalize_patient_id(patient_dir.name),
                    "normalized_timepoint_id": _normalize_timepoint_id(timepoint_dir.name),
                    "file_count": len(file_names),
                    "nii_count": nii_count,
                    "guessed_modalities": ",".join(modalities),
                    "has_mask_guess": has_mask_guess,
                    "quality_flag_missing_file": quality_flag_missing_file,
                    "quality_flag_missing_mask": quality_flag_missing_mask,
                    "quality_flag_unexpected_structure": quality_flag_unexpected_structure,
                    "quality_flag_ready_for_nnunet_candidate": quality_flag_ready_for_nnunet_candidate,
                }
            )

    return rows


def _write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "raw_patient_id",
        "raw_timepoint_id",
        "normalized_patient_id",
        "normalized_timepoint_id",
        "file_count",
        "nii_count",
        "guessed_modalities",
        "has_mask_guess",
        "quality_flag_missing_file",
        "quality_flag_missing_mask",
        "quality_flag_unexpected_structure",
        "quality_flag_ready_for_nnunet_candidate",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    raw_dir = (
        Path(args.raw_dir).expanduser().resolve()
        if args.raw_dir
        else get_data_raw_dir(str(project_root))
    )

    print("=== Phase 1B Raw Canonical Index Builder ===")
    print(f"Project root: {project_root}")
    print(f"Raw dir: {raw_dir}")
    print(f"Dry run: {args.dry_run}")

    if not raw_dir.exists() or not raw_dir.is_dir():
        print("[ERROR] Raw directory is missing or invalid.")
        return 1

    rows = build_index_rows(raw_dir)
    patient_count = len({row["normalized_patient_id"] for row in rows})
    ready_count = sum(1 for row in rows if row["quality_flag_ready_for_nnunet_candidate"])
    print(f"Indexed rows (patient-timepoint): {len(rows)}")
    print(f"Unique normalized patients: {patient_count}")
    print(f"Ready-for-nnU-Net candidate rows: {ready_count}")

    if args.dry_run:
        preview_count = max(1, int(args.max_preview))
        print("")
        print("Dry-run preview rows:")
        for row in rows[:preview_count]:
            print(row)
        print("No file was written.")
        return 0

    output_csv = _resolve_output_path(project_root, args.output_csv)
    _write_csv(rows, output_csv)
    print(f"Raw canonical index CSV written: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
