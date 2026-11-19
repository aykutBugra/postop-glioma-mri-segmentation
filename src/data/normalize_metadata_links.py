from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None
    PANDAS_IMPORT_ERROR = str(exc)
else:
    PANDAS_IMPORT_ERROR = None

from src.utils.paths import resolve_project_root

PATIENT_EXTRACT_PATTERN = re.compile(r"patientid[_\-]?(\d{1,6})", re.IGNORECASE)
TIMEPOINT_EXTRACT_PATTERN = re.compile(r"timepoint[_\-]?(\d{1,2})", re.IGNORECASE)
POST_TREATMENT_PATTERN = re.compile(
    r"post[\s_\-]*treatment[_\-]?(\d{1,2})", re.IGNORECASE
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize metadata links and build Phase 1B canonical dataset index."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    parser.add_argument(
        "--raw-index-csv",
        default="outputs/phase1_dataset_contract/raw_canonical_index.csv",
        help="Input raw canonical index CSV.",
    )
    parser.add_argument(
        "--clinical-xlsx",
        default="data/metadata/MU-Glioma-Post_ClinicalData-July2025.xlsx",
        help="Clinical metadata Excel path.",
    )
    parser.add_argument(
        "--segmentation-xlsx",
        default="data/metadata/MU-Glioma-Post_Segmentation_Volumes.xlsx",
        help="Segmentation metadata Excel path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1_dataset_contract",
        help="Output directory for match reports and enriched index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; print summaries only.",
    )
    return parser


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def normalize_patient_id(value: object) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None
    match = PATIENT_EXTRACT_PATTERN.search(text)
    if not match:
        return None
    return f"PatientID_{match.group(1).zfill(4)}"


def normalize_timepoint_id(value: object) -> Tuple[Optional[str], str, str]:
    text = str(value).strip()
    if not text:
        return None, "no_timepoint_token", "low"

    direct_match = TIMEPOINT_EXTRACT_PATTERN.search(text)
    if direct_match:
        return f"Timepoint_{int(direct_match.group(1))}", "timepoint_token", "high"

    post_treatment_match = POST_TREATMENT_PATTERN.search(text)
    if post_treatment_match:
        return (
            f"Timepoint_{int(post_treatment_match.group(1))}",
            "post_treatment_to_timepoint",
            "high",
        )

    return None, "no_timepoint_token", "low"


def _safe_first_patient_column(columns: List[object]) -> Optional[str]:
    for col in columns:
        col_norm = str(col).strip().lower()
        if "patient" in col_norm and "id" in col_norm:
            return str(col)
    return None


def _load_clinical_set(clinical_xlsx: Path) -> Tuple[set, str]:
    clinical_file = pd.ExcelFile(clinical_xlsx)
    target_sheet = "MU Glioma Post" if "MU Glioma Post" in clinical_file.sheet_names else clinical_file.sheet_names[0]
    df = pd.read_excel(clinical_xlsx, sheet_name=target_sheet)
    patient_col = "Patient_ID" if "Patient_ID" in df.columns else _safe_first_patient_column(df.columns.tolist())
    if patient_col is None:
        return set(), target_sheet
    values = df[patient_col].dropna().astype(str).tolist()
    normalized = {pid for pid in (normalize_patient_id(v) for v in values) if pid}
    return normalized, target_sheet


def _load_segmentation_tokens(segmentation_xlsx: Path) -> Dict[str, Dict[str, object]]:
    seg_book = pd.ExcelFile(segmentation_xlsx)
    token_map: Dict[str, Dict[str, object]] = {}
    for sheet_name in seg_book.sheet_names:
        df = pd.read_excel(segmentation_xlsx, sheet_name=sheet_name)
        patient_col = _safe_first_patient_column(df.columns.tolist())
        if not patient_col:
            continue
        values = df[patient_col].dropna().astype(str).str.strip().tolist()
        for raw_value in values:
            patient_id = normalize_patient_id(raw_value)
            timepoint_id, rule, confidence = normalize_timepoint_id(raw_value)
            key = raw_value
            if key not in token_map:
                token_map[key] = {
                    "raw_value": raw_value,
                    "normalized_patient_id": patient_id,
                    "normalized_timepoint_id": timepoint_id,
                    "timepoint_mapping_rule": rule,
                    "timepoint_mapping_confidence": confidence,
                    "source_sheets": set(),
                }
            token_map[key]["source_sheets"].add(sheet_name)
    return token_map


def _choose_mapping(values: List[Dict[str, object]]) -> Tuple[str, str]:
    if not values:
        return "no_segmentation_patient_match", "low"

    rules = {str(v["timepoint_mapping_rule"]) for v in values}
    confidences = {str(v["timepoint_mapping_confidence"]) for v in values}
    if "post_treatment_to_timepoint" in rules:
        rule = "post_treatment_to_timepoint"
    elif "timepoint_token" in rules:
        rule = "timepoint_token"
    else:
        rule = "no_timepoint_token"

    confidence = "high" if "high" in confidences else "low"
    return rule, confidence


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if pd is None:
        print("[ERROR] pandas is required for metadata normalization.")
        print(f"        Import error: {PANDAS_IMPORT_ERROR}")
        return 1

    project_root = resolve_project_root(args.project_root)
    raw_index_csv = _resolve_path(project_root, args.raw_index_csv)
    clinical_xlsx = _resolve_path(project_root, args.clinical_xlsx)
    segmentation_xlsx = _resolve_path(project_root, args.segmentation_xlsx)
    output_dir = _resolve_path(project_root, args.output_dir)

    print("=== Phase 1B Metadata Link Normalization ===")
    print(f"Project root: {project_root}")
    print(f"Raw index CSV: {raw_index_csv}")
    print(f"Clinical XLSX: {clinical_xlsx}")
    print(f"Segmentation XLSX: {segmentation_xlsx}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    if not raw_index_csv.exists():
        print("[ERROR] raw_canonical_index.csv not found.")
        return 1
    if not clinical_xlsx.exists():
        print("[ERROR] Clinical metadata Excel not found.")
        return 1
    if not segmentation_xlsx.exists():
        print("[ERROR] Segmentation metadata Excel not found.")
        return 1

    raw_df = pd.read_csv(raw_index_csv)
    required_cols = {
        "normalized_patient_id",
        "normalized_timepoint_id",
        "quality_flag_ready_for_nnunet_candidate",
    }
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns in raw index: {missing_cols}")
        return 1

    clinical_patient_set, clinical_sheet = _load_clinical_set(clinical_xlsx)
    seg_token_map = _load_segmentation_tokens(segmentation_xlsx)

    seg_values = list(seg_token_map.values())
    segmentation_patient_set = {
        str(v["normalized_patient_id"])
        for v in seg_values
        if v["normalized_patient_id"]
    }
    segmentation_timepoint_set = {
        f"{v['normalized_patient_id']}_{v['normalized_timepoint_id']}"
        for v in seg_values
        if v["normalized_patient_id"] and v["normalized_timepoint_id"]
    }

    seg_values_by_patient: Dict[str, List[Dict[str, object]]] = {}
    seg_values_by_patient_timepoint: Dict[str, List[Dict[str, object]]] = {}
    for item in seg_values:
        patient_id = item["normalized_patient_id"]
        timepoint_id = item["normalized_timepoint_id"]
        if patient_id:
            seg_values_by_patient.setdefault(str(patient_id), []).append(item)
            if timepoint_id:
                key = f"{patient_id}_{timepoint_id}"
                seg_values_by_patient_timepoint.setdefault(key, []).append(item)

    enriched_df = raw_df.copy()
    enriched_df["clinical_patient_match_found"] = enriched_df["normalized_patient_id"].isin(clinical_patient_set)
    enriched_df["segmentation_patient_match_found"] = enriched_df["normalized_patient_id"].isin(segmentation_patient_set)
    enriched_df["raw_patient_time_key"] = (
        enriched_df["normalized_patient_id"].astype(str)
        + "_"
        + enriched_df["normalized_timepoint_id"].astype(str)
    )
    enriched_df["segmentation_timepoint_match_found"] = enriched_df["raw_patient_time_key"].isin(segmentation_timepoint_set)

    mapping_rules: List[str] = []
    mapping_confidences: List[str] = []
    for _, row in enriched_df.iterrows():
        key = row["raw_patient_time_key"]
        patient = row["normalized_patient_id"]
        tp_matches = seg_values_by_patient_timepoint.get(key, [])
        if tp_matches:
            rule, conf = _choose_mapping(tp_matches)
        elif patient in seg_values_by_patient:
            rule, conf = _choose_mapping(seg_values_by_patient.get(str(patient), []))
            if rule in {"post_treatment_to_timepoint", "timepoint_token"}:
                rule = "no_timepoint_token"
                conf = "low"
        else:
            rule, conf = "no_segmentation_patient_match", "low"
        mapping_rules.append(rule)
        mapping_confidences.append(conf)

    enriched_df["timepoint_mapping_rule"] = mapping_rules
    enriched_df["timepoint_mapping_confidence"] = mapping_confidences

    enriched_df["quality_flag_clinical_patient_missing"] = ~enriched_df["clinical_patient_match_found"]
    enriched_df["quality_flag_segmentation_patient_missing"] = ~enriched_df["segmentation_patient_match_found"]
    enriched_df["quality_flag_segmentation_timepoint_missing"] = ~enriched_df["segmentation_timepoint_match_found"]
    enriched_df["quality_flag_metadata_match_ambiguous"] = (
        enriched_df["quality_flag_clinical_patient_missing"]
        | enriched_df["quality_flag_segmentation_patient_missing"]
        | enriched_df["quality_flag_segmentation_timepoint_missing"]
    )

    clinical_report = (
        enriched_df[["normalized_patient_id", "clinical_patient_match_found"]]
        .drop_duplicates()
        .sort_values("normalized_patient_id")
    )
    segmentation_patient_report = (
        enriched_df[["normalized_patient_id", "segmentation_patient_match_found"]]
        .drop_duplicates()
        .sort_values("normalized_patient_id")
    )
    segmentation_timepoint_report = (
        enriched_df[
            [
                "normalized_patient_id",
                "normalized_timepoint_id",
                "segmentation_timepoint_match_found",
                "timepoint_mapping_rule",
                "timepoint_mapping_confidence",
            ]
        ]
        .drop_duplicates()
        .sort_values(["normalized_patient_id", "normalized_timepoint_id"])
    )

    quality_flags_summary = {
        "row_count": int(len(enriched_df)),
        "quality_flag_clinical_patient_missing_true": int(enriched_df["quality_flag_clinical_patient_missing"].sum()),
        "quality_flag_segmentation_patient_missing_true": int(enriched_df["quality_flag_segmentation_patient_missing"].sum()),
        "quality_flag_segmentation_timepoint_missing_true": int(enriched_df["quality_flag_segmentation_timepoint_missing"].sum()),
        "quality_flag_metadata_match_ambiguous_true": int(enriched_df["quality_flag_metadata_match_ambiguous"].sum()),
        "quality_flag_ready_for_nnunet_candidate_true": int(enriched_df["quality_flag_ready_for_nnunet_candidate"].sum()),
    }

    dataset_contract_summary = {
        "clinical_sheet_used": clinical_sheet,
        "raw_rows": int(len(enriched_df)),
        "raw_unique_patients": int(enriched_df["normalized_patient_id"].nunique()),
        "clinical_normalized_patient_count": int(len(clinical_patient_set)),
        "segmentation_normalized_patient_count": int(len(segmentation_patient_set)),
        "segmentation_normalized_patient_timepoint_count": int(len(segmentation_timepoint_set)),
        "timepoint_mapping_rule_counts": {
            str(k): int(v)
            for k, v in enriched_df["timepoint_mapping_rule"].value_counts().to_dict().items()
        },
        "timepoint_mapping_confidence_counts": {
            str(k): int(v)
            for k, v in enriched_df["timepoint_mapping_confidence"].value_counts().to_dict().items()
        },
        "quality_flags_summary": quality_flags_summary,
    }

    if args.dry_run:
        print("Clinical patient matches:", int(clinical_report["clinical_patient_match_found"].sum()))
        print("Segmentation patient matches:", int(segmentation_patient_report["segmentation_patient_match_found"].sum()))
        print(
            "Segmentation timepoint matches:",
            int(segmentation_timepoint_report["segmentation_timepoint_match_found"].sum()),
        )
        print("No files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    clinical_report.to_csv(output_dir / "clinical_patient_match_report.csv", index=False)
    segmentation_patient_report.to_csv(
        output_dir / "segmentation_patient_match_report.csv",
        index=False,
    )
    segmentation_timepoint_report.to_csv(
        output_dir / "segmentation_timepoint_match_report.csv",
        index=False,
    )
    enriched_df.drop(columns=["raw_patient_time_key"]).to_csv(
        output_dir / "canonical_dataset_index.csv",
        index=False,
    )
    (output_dir / "quality_flags_summary.json").write_text(
        json.dumps(quality_flags_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dataset_contract_summary.json").write_text(
        json.dumps(dataset_contract_summary, indent=2),
        encoding="utf-8",
    )

    print("Wrote:", output_dir / "clinical_patient_match_report.csv")
    print("Wrote:", output_dir / "segmentation_patient_match_report.csv")
    print("Wrote:", output_dir / "segmentation_timepoint_match_report.csv")
    print("Wrote:", output_dir / "canonical_dataset_index.csv")
    print("Wrote:", output_dir / "quality_flags_summary.json")
    print("Wrote:", output_dir / "dataset_contract_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
