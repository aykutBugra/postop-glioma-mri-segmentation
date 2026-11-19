from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.paths import get_phase1_nnunet_output_dir, resolve_project_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic patient-level train/val/test splits for Phase 1E."
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
        "--case-id-map-csv",
        default="outputs/phase1_nnunet_dataset/case_id_map.csv",
        help="Case-id map CSV produced by prepare_nnunet_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Split output directory. Default: <project_root>/outputs/phase1_nnunet_dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    return parser


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _load_case_map(case_map_csv: Path) -> List[Dict[str, str]]:
    with case_map_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "case_id",
            "normalized_patient_id",
            "normalized_timepoint_id",
            "raw_patient_id",
            "raw_timepoint_id",
        }
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in case_id_map.csv: {missing}")
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in reader]


def _load_ready_keys(canonical_index_csv: Path) -> set[Tuple[str, str]]:
    with canonical_index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "normalized_patient_id",
            "normalized_timepoint_id",
            "quality_flag_ready_for_nnunet_candidate",
        }
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in canonical index CSV: {missing}")

        keys: set[Tuple[str, str]] = set()
        for row in reader:
            ready_text = str(row.get("quality_flag_ready_for_nnunet_candidate", "")).strip().lower()
            ready = ready_text in {"1", "true", "yes", "y", "t"}
            if ready:
                keys.add(
                    (
                        str(row.get("normalized_patient_id", "")).strip(),
                        str(row.get("normalized_timepoint_id", "")).strip(),
                    )
                )
        return keys


def _largest_remainder_counts(total: int, ratios: List[float]) -> List[int]:
    raw = [total * r for r in ratios]
    base = [math.floor(v) for v in raw]
    remainder = total - sum(base)
    fractions = sorted(
        [(idx, raw[idx] - base[idx]) for idx in range(len(ratios))],
        key=lambda x: (-x[1], x[0]),
    )
    counts = base[:]
    for idx, _frac in fractions[:remainder]:
        counts[idx] += 1
    return counts


def _build_patient_splits(
    patient_ids: List[str], seed: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Dict[str, set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(patient_ids)
    rng.shuffle(shuffled)

    train_count, val_count, test_count = _largest_remainder_counts(
        len(shuffled), [train_ratio, val_ratio, test_ratio]
    )
    train_patients = set(shuffled[:train_count])
    val_patients = set(shuffled[train_count : train_count + val_count])
    test_patients = set(shuffled[train_count + val_count : train_count + val_count + test_count])
    return {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }


def _validate_leakage(split_map: Dict[str, set[str]]) -> Tuple[bool, Dict[str, List[str]]]:
    intersections = {
        "train_val": sorted(split_map["train"].intersection(split_map["val"])),
        "train_test": sorted(split_map["train"].intersection(split_map["test"])),
        "val_test": sorted(split_map["val"].intersection(split_map["test"])),
    }
    has_leakage = any(intersections[key] for key in intersections)
    return (not has_leakage), intersections


def _write_manifest(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "split",
        "normalized_patient_id",
        "normalized_timepoint_id",
        "raw_patient_id",
        "raw_timepoint_id",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_case_id_list(case_ids: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(case_ids)) + "\n", encoding="utf-8")


def _write_splits_final_json(train_case_ids: List[str], val_case_ids: List[str], output_path: Path) -> None:
    # nnU-Net split contract: list of folds, each fold has train/val case id lists.
    payload = [
        {
            "train": sorted(train_case_ids),
            "val": sorted(val_case_ids),
        }
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        print(
            f"[ERROR] train/val/test ratios must sum to 1.0, got {ratio_sum:.10f}. "
            "Example: 0.70 0.15 0.15"
        )
        return 1

    project_root = resolve_project_root(args.project_root)
    canonical_index_csv = _resolve_path(project_root, args.canonical_index_csv)
    case_id_map_csv = _resolve_path(project_root, args.case_id_map_csv)
    output_dir = (
        _resolve_path(project_root, args.output_dir)
        if args.output_dir
        else get_phase1_nnunet_output_dir(str(project_root))
    )

    print("=== Phase 1E Patient-Level Split Builder ===")
    print(f"Project root: {project_root}")
    print(f"Canonical index CSV: {canonical_index_csv}")
    print(f"Case id map CSV: {case_id_map_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {args.seed}")
    print(
        f"Ratios train/val/test: {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}"
    )

    if not canonical_index_csv.exists():
        print("[ERROR] canonical index CSV not found.")
        return 1
    if not case_id_map_csv.exists():
        print("[ERROR] case_id_map.csv not found.")
        return 1

    try:
        case_rows = _load_case_map(case_id_map_csv)
        ready_keys = _load_ready_keys(canonical_index_csv)
    except Exception as exc:
        print(f"[ERROR] Failed to load inputs: {exc}")
        return 1

    if not case_rows:
        print("[ERROR] case_id_map.csv is empty.")
        return 1

    case_keys = {
        (row["normalized_patient_id"].strip(), row["normalized_timepoint_id"].strip())
        for row in case_rows
    }
    unexpected_case_keys = sorted(case_keys - ready_keys)
    if unexpected_case_keys:
        print(
            "[ERROR] Found case(s) in case_id_map.csv that are not ready candidates in canonical index."
        )
        print(f"        Example: {unexpected_case_keys[:3]}")
        return 1

    unique_patients = sorted({row["normalized_patient_id"].strip() for row in case_rows})
    split_patients = _build_patient_splits(
        unique_patients,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    leakage_pass, intersections = _validate_leakage(split_patients)
    if not leakage_pass:
        print("[ERROR] Patient leakage detected between split partitions.")
        print(json.dumps(intersections, indent=2))
        return 1

    manifest_rows: List[Dict[str, str]] = []
    for row in case_rows:
        patient_id = row["normalized_patient_id"].strip()
        split_name = None
        for candidate_split in ("train", "val", "test"):
            if patient_id in split_patients[candidate_split]:
                split_name = candidate_split
                break
        if split_name is None:
            print(f"[ERROR] Could not assign split for patient: {patient_id}")
            return 1

        manifest_rows.append(
            {
                "case_id": row["case_id"],
                "split": split_name,
                "normalized_patient_id": row["normalized_patient_id"],
                "normalized_timepoint_id": row["normalized_timepoint_id"],
                "raw_patient_id": row["raw_patient_id"],
                "raw_timepoint_id": row["raw_timepoint_id"],
            }
        )

    manifest_rows = sorted(manifest_rows, key=lambda r: (r["split"], r["case_id"]))
    split_manifest_csv = output_dir / "split_manifest_phase1.csv"
    split_summary_json = output_dir / "split_summary.json"
    train_case_ids_txt = output_dir / "train_case_ids.txt"
    val_case_ids_txt = output_dir / "val_case_ids.txt"
    held_out_test_case_ids_txt = output_dir / "held_out_test_case_ids.txt"
    splits_final_json = output_dir / "splits_final.json"

    _write_manifest(manifest_rows, split_manifest_csv)
    train_case_ids = sorted([row["case_id"] for row in manifest_rows if row["split"] == "train"])
    val_case_ids = sorted([row["case_id"] for row in manifest_rows if row["split"] == "val"])
    test_case_ids = sorted([row["case_id"] for row in manifest_rows if row["split"] == "test"])

    _write_case_id_list(train_case_ids, train_case_ids_txt)
    _write_case_id_list(val_case_ids, val_case_ids_txt)
    _write_case_id_list(test_case_ids, held_out_test_case_ids_txt)
    _write_splits_final_json(train_case_ids, val_case_ids, splits_final_json)

    def _split_counts(split_name: str) -> Dict[str, int]:
        split_cases = [r for r in manifest_rows if r["split"] == split_name]
        split_patients_local = {r["normalized_patient_id"] for r in split_cases}
        return {
            "unique_patients": len(split_patients_local),
            "timepoint_cases": len(split_cases),
        }

    summary = {
        "seed": args.seed,
        "ratios": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "manifest_rows_total": len(manifest_rows),
        "total": {
            "unique_patients": len(unique_patients),
            "timepoint_cases": len(manifest_rows),
        },
        "train": _split_counts("train"),
        "val": _split_counts("val"),
        "test": _split_counts("test"),
        "patient_intersections": intersections,
        "patient_intersection_counts": {
            key: len(value) for key, value in intersections.items()
        },
        "held_out_test": {
            "enabled": True,
            "test_case_ids_txt": str(held_out_test_case_ids_txt),
            "test_case_count": len(test_case_ids),
            "policy": "held-out test set is excluded from training and model selection",
        },
        "artifacts": {
            "split_manifest_phase1_csv": str(split_manifest_csv),
            "split_summary_json": str(split_summary_json),
            "train_case_ids_txt": str(train_case_ids_txt),
            "val_case_ids_txt": str(val_case_ids_txt),
            "held_out_test_case_ids_txt": str(held_out_test_case_ids_txt),
            "splits_final_json": str(splits_final_json),
        },
        "leakage_check": {
            "pass": leakage_pass,
            "reason": "none" if leakage_pass else "patient appears in multiple splits",
        },
    }
    split_summary_json.parent.mkdir(parents=True, exist_ok=True)
    split_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {split_manifest_csv}")
    print(f"Wrote: {split_summary_json}")
    print(f"Wrote: {train_case_ids_txt}")
    print(f"Wrote: {val_case_ids_txt}")
    print(f"Wrote: {held_out_test_case_ids_txt}")
    print(f"Wrote: {splits_final_json}")
    print("Leakage check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
