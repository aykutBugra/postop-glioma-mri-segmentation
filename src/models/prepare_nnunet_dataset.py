from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from src.utils.paths import (
    get_data_raw_dir,
    get_phase1_nnunet_output_dir,
    resolve_project_root,
)

NIFTI_PATTERN = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
PATIENT_PATTERN = re.compile(r"patientid[_\-]?(\d{1,6})", re.IGNORECASE)
TIMEPOINT_PATTERN = re.compile(r"timepoint[_\-]?(\d{1,3})", re.IGNORECASE)

MODALITY_ORDER: List[Tuple[str, str]] = [
    ("t1c", "0000"),
    ("t1n", "0001"),
    ("t2w", "0002"),
    ("t2f", "0003"),
]
MASK_HINTS = ("mask", "tumor", "seg", "label")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 1D nnU-Net dataset from canonical index (conversion only, no training)."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    parser.add_argument(
        "--canonical-index-csv",
        default="outputs/phase1_dataset_contract/canonical_dataset_index.csv",
        help="Canonical dataset index CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory. Default: <project_root>/outputs/phase1_nnunet_dataset",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=901,
        help="nnU-Net dataset numeric id.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Phase1GliomaBaseline",
        help="nnU-Net dataset name suffix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write dataset files; print summary only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of ready rows to convert (for lightweight test).",
    )
    return parser


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _safe_dataset_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(name).strip())
    return cleaned or "Phase1Dataset"


def _load_canonical_rows(canonical_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with canonical_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "raw_patient_id",
            "raw_timepoint_id",
            "normalized_patient_id",
            "normalized_timepoint_id",
            "quality_flag_ready_for_nnunet_candidate",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in canonical CSV: {missing}")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _build_case_id(patient_id: str, timepoint_id: str) -> str:
    p_match = PATIENT_PATTERN.search(patient_id)
    t_match = TIMEPOINT_PATTERN.search(timepoint_id)
    if not p_match or not t_match:
        safe_p = re.sub(r"[^A-Za-z0-9]", "", patient_id)[:16] or "UNKP"
        safe_t = re.sub(r"[^A-Za-z0-9]", "", timepoint_id)[:16] or "UNKT"
        return f"PID{safe_p}_TP{safe_t}"
    return f"PID{p_match.group(1).zfill(4)}_TP{t_match.group(1).zfill(2)}"


def _collect_nifti_files(timepoint_dir: Path) -> List[Path]:
    if not timepoint_dir.exists() or not timepoint_dir.is_dir():
        return []
    return sorted(
        [p for p in timepoint_dir.iterdir() if p.is_file() and NIFTI_PATTERN.search(p.name)],
        key=lambda p: p.name.lower(),
    )


def _choose_modality_file(files: List[Path], modality: str) -> Optional[Path]:
    token = modality.lower()
    hits = [p for p in files if token in p.name.lower()]
    if not hits:
        return None
    return hits[0]


def _choose_mask_file(files: List[Path]) -> Optional[Path]:
    for p in files:
        name = p.name.lower()
        if any(hint in name for hint in MASK_HINTS):
            return p
    return None


def _ensure_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _write_case_map(case_rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "case_id",
        "raw_patient_id",
        "raw_timepoint_id",
        "normalized_patient_id",
        "normalized_timepoint_id",
        "image_0000",
        "image_0001",
        "image_0002",
        "image_0003",
        "label",
        "quality_flag_metadata_match_ambiguous",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)


def _read_label_values(label_path: Path) -> List[int]:
    arr = np.asanyarray(nib.load(str(label_path)).dataobj)
    return [int(v) for v in np.unique(arr)]


def _build_label_dict(label_values: List[int]) -> Dict[str, int]:
    unique_values = sorted(set(int(v) for v in label_values))
    if not unique_values:
        raise ValueError("No label values were found in converted masks.")
    if unique_values[0] != 0:
        raise ValueError(f"Expected background label 0, got values: {unique_values}")

    labels: Dict[str, int] = {"background": 0}
    for value in unique_values[1:]:
        labels[f"label_{value}"] = int(value)
    return labels


def _build_dataset_json(dataset_name: str, num_training: int, label_values: List[int]) -> Dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "channel_names": {
            "0": "T1c",
            "1": "T1n",
            "2": "T2w",
            "3": "T2f",
        },
        "labels": _build_label_dict(label_values),
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }


def _write_json(data: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _cleanup_existing_nifti(dataset_dir: Path) -> Dict[str, int]:
    removed = {"images_removed": 0, "labels_removed": 0}
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    for file_path in images_dir.glob("*.nii.gz"):
        file_path.unlink()
        removed["images_removed"] += 1
    for file_path in labels_dir.glob("*.nii.gz"):
        file_path.unlink()
        removed["labels_removed"] += 1
    return removed


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    canonical_csv = _resolve_path(project_root, args.canonical_index_csv)
    raw_dir = get_data_raw_dir(str(project_root))

    if args.output_dir:
        output_root = _resolve_path(project_root, args.output_dir)
    else:
        output_root = get_phase1_nnunet_output_dir(str(project_root))

    safe_name = _safe_dataset_name(args.dataset_name)
    dataset_dir = output_root / f"Dataset{int(args.dataset_id):03d}_{safe_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    print("=== Phase 1D nnU-Net Dataset Preparation ===")
    print(f"Project root: {project_root}")
    print(f"Canonical index: {canonical_csv}")
    print(f"Raw dir: {raw_dir}")
    print(f"Output root: {output_root}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Limit: {args.limit}")

    if not canonical_csv.exists() or not canonical_csv.is_file():
        print("[ERROR] canonical index CSV not found.")
        return 1
    if not raw_dir.exists() or not raw_dir.is_dir():
        print("[ERROR] data/raw directory not found.")
        return 1

    try:
        all_rows = _load_canonical_rows(canonical_csv)
    except Exception as exc:
        print(f"[ERROR] Failed to read canonical index: {exc}")
        return 1

    ready_rows = [r for r in all_rows if _to_bool(r.get("quality_flag_ready_for_nnunet_candidate", ""))]
    ready_rows = sorted(
        ready_rows,
        key=lambda r: (
            r.get("normalized_patient_id", ""),
            r.get("normalized_timepoint_id", ""),
            r.get("raw_patient_id", ""),
            r.get("raw_timepoint_id", ""),
        ),
    )
    non_ready_rows = [r for r in all_rows if not _to_bool(r.get("quality_flag_ready_for_nnunet_candidate", ""))]
    selected_rows = ready_rows[: args.limit] if args.limit and args.limit > 0 else ready_rows
    run_mode = "limited" if args.limit and args.limit > 0 else "full"

    case_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, str]] = []
    case_ids: set[str] = set()
    link_mode_counts = {"hardlink": 0, "copy": 0, "exists": 0}
    cleanup_summary = {"images_removed": 0, "labels_removed": 0}
    global_label_values: set[int] = set()

    if not args.dry_run and run_mode == "full":
        cleanup_summary = _cleanup_existing_nifti(dataset_dir)
        print(
            "Cleanup before full conversion: "
            f"images_removed={cleanup_summary['images_removed']}, "
            f"labels_removed={cleanup_summary['labels_removed']}"
        )

    for row in selected_rows:
        raw_patient_id = row.get("raw_patient_id", "").strip()
        raw_timepoint_id = row.get("raw_timepoint_id", "").strip()
        norm_patient_id = row.get("normalized_patient_id", "").strip()
        norm_timepoint_id = row.get("normalized_timepoint_id", "").strip()

        tp_dir = raw_dir / raw_patient_id / raw_timepoint_id
        files = _collect_nifti_files(tp_dir)
        modality_files: Dict[str, Path] = {}
        for modality, _channel in MODALITY_ORDER:
            src = _choose_modality_file(files, modality)
            if src is None:
                skipped_rows.append(
                    {
                        "raw_patient_id": raw_patient_id,
                        "raw_timepoint_id": raw_timepoint_id,
                        "reason": f"missing_modality_{modality}",
                    }
                )
                modality_files = {}
                break
            modality_files[modality] = src
        if not modality_files:
            continue

        mask_src = _choose_mask_file(files)
        if mask_src is None:
            skipped_rows.append(
                {
                    "raw_patient_id": raw_patient_id,
                    "raw_timepoint_id": raw_timepoint_id,
                    "reason": "missing_mask",
                }
            )
            continue

        case_id = _build_case_id(norm_patient_id or raw_patient_id, norm_timepoint_id or raw_timepoint_id)
        if case_id in case_ids:
            print(f"[ERROR] Duplicate case_id generated: {case_id}")
            return 1
        case_ids.add(case_id)

        image_targets: Dict[str, str] = {}
        label_target_rel = (dataset_dir / "labelsTr" / f"{case_id}.nii.gz").relative_to(output_root).as_posix()

        for modality, channel in MODALITY_ORDER:
            rel = (dataset_dir / "imagesTr" / f"{case_id}_{channel}.nii.gz").relative_to(output_root).as_posix()
            image_targets[channel] = rel

        if not args.dry_run:
            for modality, channel in MODALITY_ORDER:
                dst = output_root / image_targets[channel]
                mode = _ensure_link_or_copy(modality_files[modality], dst)
                link_mode_counts[mode] += 1
            label_dst = output_root / label_target_rel
            mode = _ensure_link_or_copy(mask_src, label_dst)
            link_mode_counts[mode] += 1
            label_values = _read_label_values(label_dst)
        else:
            label_values = _read_label_values(mask_src)
        global_label_values.update(label_values)

        case_rows.append(
            {
                "case_id": case_id,
                "raw_patient_id": raw_patient_id,
                "raw_timepoint_id": raw_timepoint_id,
                "normalized_patient_id": norm_patient_id,
                "normalized_timepoint_id": norm_timepoint_id,
                "image_0000": image_targets["0000"],
                "image_0001": image_targets["0001"],
                "image_0002": image_targets["0002"],
                "image_0003": image_targets["0003"],
                "label": label_target_rel,
                "quality_flag_metadata_match_ambiguous": row.get(
                    "quality_flag_metadata_match_ambiguous", ""
                ),
            }
        )

    dataset_json = _build_dataset_json(args.dataset_name, len(case_rows), sorted(global_label_values))
    summary = {
        "project_root": str(project_root),
        "canonical_index_csv": str(canonical_csv),
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "dataset_id": int(args.dataset_id),
        "dataset_name": args.dataset_name,
        "run_mode": run_mode,
        "dry_run": bool(args.dry_run),
        "limit": int(args.limit) if args.limit else None,
        "raw_rows_total": len(all_rows),
        "expected_ready_count": len(ready_rows),
        "excluded_non_ready_count": len(non_ready_rows),
        "ready_rows_total": len(ready_rows),
        "non_ready_rows_total": len(non_ready_rows),
        "selected_rows_for_run": len(selected_rows),
        "converted_case_count": len(case_rows),
        "skipped_case_count": len(skipped_rows),
        "skipped_examples": skipped_rows[:20],
        "known_non_ready_examples": [
            {
                "raw_patient_id": r.get("raw_patient_id", ""),
                "raw_timepoint_id": r.get("raw_timepoint_id", ""),
            }
            for r in non_ready_rows[:20]
        ],
        "global_label_values": sorted(global_label_values),
        "label_definition": dataset_json["labels"],
        "link_mode_counts": link_mode_counts,
        "cleanup_summary": cleanup_summary,
    }

    print(f"Raw rows total: {len(all_rows)}")
    print(f"Ready rows total: {len(ready_rows)}")
    print(f"Selected for this run: {len(selected_rows)}")
    print(f"Converted cases: {len(case_rows)}")
    print(f"Skipped cases: {len(skipped_rows)}")
    if len(case_rows) != len(selected_rows):
        print(
            "[WARN] Converted case count does not match selected row count. "
            "See skipped_examples in summary."
        )

    if args.dry_run:
        print("Dry-run mode: no files were written.")
        print(json.dumps(summary, indent=2))
        return 0

    case_map_csv = output_root / "case_id_map.csv"
    dataset_json_dataset_path = dataset_dir / "dataset.json"
    dataset_json_root_path = output_root / "dataset.json"
    conversion_summary_path = output_root / "conversion_summary.json"

    _write_case_map(case_rows, case_map_csv)
    _write_json(dataset_json, dataset_json_dataset_path)
    _write_json(dataset_json, dataset_json_root_path)
    _write_json(summary, conversion_summary_path)

    if run_mode == "full" and len(case_rows) != len(ready_rows):
        print(
            "[ERROR] Full conversion did not convert all ready rows. "
            f"ready={len(ready_rows)} converted={len(case_rows)}"
        )
        return 1

    print(f"Wrote: {case_map_csv}")
    print(f"Wrote: {dataset_json_dataset_path}")
    print(f"Wrote: {dataset_json_root_path}")
    print(f"Wrote: {conversion_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
