from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SEMANTIC_LABELS: Dict[str, str] = {
    "label_1": "NETC",
    "label_2": "SNFH",
    "label_3": "ET",
    "label_4": "RC",
}

SUBGROUP_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("lt_1ml", 0.0, 1.0),
    ("1_to_5ml", 1.0, 5.0),
    ("gt_5ml", 5.0, float("inf")),
)

RED_FLAG_THRESHOLDS: Dict[str, float] = {
    "mean_foreground_dice_min": 0.905,
    "label_2_mean_dice_min": 0.900,
    "label_3_mean_dice_min": 0.780,
    "mean_abs_wt_volume_error_ml_max": 2.35,
}

LOW_DICE_THRESHOLD = 0.2


@dataclass(frozen=True)
class ExperimentDecision:
    overall_pass: bool
    rollback_required: bool
    pain_point_success_count: int
    pain_point_successes: Dict[str, bool]
    red_flags: Dict[str, bool]
    label_1_volume_inflation_ml: float
    label_1_volume_inflation_delta_ml: Optional[float]
    label_4_mean_signed_volume_error_ml: float
    label_4_suppression_ml: float
    label_4_suppression_delta_ml: Optional[float]
    label_1_to_label_4_confusion_change: Optional[float]
    label_4_to_label_1_confusion_change: Optional[float]
    combined_confusion_change: Optional[float]


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def _load_metrics_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "case_id",
        "foreground_dice",
        "gt_foreground_volume_ml",
        "pred_foreground_volume_ml",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _bucket_mask(volume_series: pd.Series, lower: float, upper: float) -> pd.Series:
    volumes = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)
    if np.isinf(upper):
        return volumes > lower
    return (volumes >= lower) & (volumes < upper)


def _compute_label_summary(metrics_df: pd.DataFrame, label_name: str) -> Dict[str, float]:
    dice_col = f"{label_name}_dice"
    gt_present_col = f"{label_name}_gt_present"
    pred_present_col = f"{label_name}_pred_present"

    active_mask = _coerce_bool_series(metrics_df[gt_present_col]) | _coerce_bool_series(
        metrics_df[pred_present_col]
    )
    active_df = metrics_df.loc[active_mask]

    return {
        "mean_dice_active": _safe_mean(active_df[dice_col]),
        "median_dice_active": _safe_median(active_df[dice_col]),
    }


def build_subgroup_summary(
    metrics_df: pd.DataFrame,
    label_names: Sequence[str] = ("label_1", "label_4"),
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label_name in label_names:
        gt_volume_col = f"{label_name}_gt_volume_ml"
        pred_present_col = f"{label_name}_pred_present"
        dice_col = f"{label_name}_dice"
        abs_error_col = f"{label_name}_volume_abs_error_ml"

        pred_present = _coerce_bool_series(metrics_df[pred_present_col])
        for bucket_name, lower, upper in SUBGROUP_BUCKETS:
            bucket_df = metrics_df.loc[_bucket_mask(metrics_df[gt_volume_col], lower, upper)].copy()
            case_count = int(len(bucket_df))
            active_case_count = int(case_count)
            recall = (
                float(_coerce_bool_series(bucket_df[pred_present_col]).mean())
                if case_count
                else float("nan")
            )
            dice_values = pd.to_numeric(bucket_df[dice_col], errors="coerce")
            zero_dice_count = int((dice_values.fillna(0.0) <= 0).sum()) if case_count else 0
            rows.append(
                {
                    "label_name": label_name,
                    "semantic_name": SEMANTIC_LABELS.get(label_name, label_name),
                    "bucket_name": bucket_name,
                    "bucket_min_ml": lower,
                    "bucket_max_ml": None if np.isinf(upper) else upper,
                    "case_count": case_count,
                    "active_case_count": active_case_count,
                    "mean_dice_active": _safe_mean(bucket_df[dice_col]),
                    "median_dice_active": _safe_median(bucket_df[dice_col]),
                    "recall": recall,
                    "zero_dice_count": zero_dice_count,
                    "mean_abs_volume_error_ml": _safe_mean(bucket_df[abs_error_col]),
                    "pred_present_rate": float(pred_present.loc[bucket_df.index].mean())
                    if case_count
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_confusion_summary(
    metrics_df: pd.DataFrame,
    low_dice_threshold: float = LOW_DICE_THRESHOLD,
    label_1_name: str = "label_1",
    label_4_name: str = "label_4",
) -> pd.DataFrame:
    l1_gt = _coerce_bool_series(metrics_df[f"{label_1_name}_gt_present"])
    l1_pred = _coerce_bool_series(metrics_df[f"{label_1_name}_pred_present"])
    l4_gt = _coerce_bool_series(metrics_df[f"{label_4_name}_gt_present"])
    l4_pred = _coerce_bool_series(metrics_df[f"{label_4_name}_pred_present"])
    l1_dice = pd.to_numeric(metrics_df[f"{label_1_name}_dice"], errors="coerce").fillna(0.0)
    l4_dice = pd.to_numeric(metrics_df[f"{label_4_name}_dice"], errors="coerce").fillna(0.0)

    gt_label1_only = l1_gt & ~l4_gt
    gt_label4_only = l4_gt & ~l1_gt
    l1_low_dice_cross = l1_gt & (l1_dice < low_dice_threshold) & l4_pred
    l4_low_dice_cross = l4_gt & (l4_dice < low_dice_threshold) & l1_pred

    rows = [
        {
            "metric_name": "gt_label_1_only_case_count",
            "value": int(gt_label1_only.sum()),
        },
        {
            "metric_name": "gt_label_1_only_pred_label_4_present_count",
            "value": int((gt_label1_only & l4_pred).sum()),
        },
        {
            "metric_name": "gt_label_1_only_pred_label_4_present_rate",
            "value": float((gt_label1_only & l4_pred).sum() / gt_label1_only.sum())
            if int(gt_label1_only.sum()) > 0
            else float("nan"),
        },
        {
            "metric_name": "gt_label_4_only_case_count",
            "value": int(gt_label4_only.sum()),
        },
        {
            "metric_name": "gt_label_4_only_pred_label_1_present_count",
            "value": int((gt_label4_only & l1_pred).sum()),
        },
        {
            "metric_name": "gt_label_4_only_pred_label_1_present_rate",
            "value": float((gt_label4_only & l1_pred).sum() / gt_label4_only.sum())
            if int(gt_label4_only.sum()) > 0
            else float("nan"),
        },
        {
            "metric_name": "label_1_low_dice_cross_present_count",
            "value": int(l1_low_dice_cross.sum()),
        },
        {
            "metric_name": "label_4_low_dice_cross_present_count",
            "value": int(l4_low_dice_cross.sum()),
        },
        {
            "metric_name": "combined_confusion_rate",
            "value": float(
                (
                    float((gt_label1_only & l4_pred).sum() / gt_label1_only.sum())
                    if int(gt_label1_only.sum()) > 0
                    else 0.0
                )
                + (
                    float((gt_label4_only & l1_pred).sum() / gt_label4_only.sum())
                    if int(gt_label4_only.sum()) > 0
                    else 0.0
                )
            ),
        },
    ]
    return pd.DataFrame(rows)


def _confusion_lookup(confusion_df: pd.DataFrame) -> Dict[str, float]:
    return {
        row.metric_name: float(row.value)
        for row in confusion_df.itertuples(index=False)
        if pd.notna(row.value)
    }


def build_guardrail_summary(
    metrics_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> pd.DataFrame:
    label_2_summary = _compute_label_summary(metrics_df, "label_2")
    label_3_summary = _compute_label_summary(metrics_df, "label_3")
    confusion = _confusion_lookup(confusion_df)

    l1_small = subgroup_df[
        (subgroup_df["label_name"] == "label_1") & (subgroup_df["bucket_name"] == "lt_1ml")
    ].iloc[0]
    l4_small = subgroup_df[
        (subgroup_df["label_name"] == "label_4") & (subgroup_df["bucket_name"] == "lt_1ml")
    ].iloc[0]

    rows = [
        {
            "metric_name": "mean_foreground_dice",
            "current_value": _safe_mean(metrics_df["foreground_dice"]),
            "threshold_type": "min",
            "threshold_value": RED_FLAG_THRESHOLDS["mean_foreground_dice_min"],
            "pass": _safe_mean(metrics_df["foreground_dice"])
            >= RED_FLAG_THRESHOLDS["mean_foreground_dice_min"],
        },
        {
            "metric_name": "label_2_mean_dice_active",
            "current_value": label_2_summary["mean_dice_active"],
            "threshold_type": "min",
            "threshold_value": RED_FLAG_THRESHOLDS["label_2_mean_dice_min"],
            "pass": label_2_summary["mean_dice_active"]
            >= RED_FLAG_THRESHOLDS["label_2_mean_dice_min"],
        },
        {
            "metric_name": "label_3_mean_dice_active",
            "current_value": label_3_summary["mean_dice_active"],
            "threshold_type": "min",
            "threshold_value": RED_FLAG_THRESHOLDS["label_3_mean_dice_min"],
            "pass": label_3_summary["mean_dice_active"]
            >= RED_FLAG_THRESHOLDS["label_3_mean_dice_min"],
        },
        {
            "metric_name": "mean_abs_wt_volume_error_ml",
            "current_value": _safe_mean(
                (metrics_df["pred_foreground_volume_ml"] - metrics_df["gt_foreground_volume_ml"]).abs()
            ),
            "threshold_type": "max",
            "threshold_value": RED_FLAG_THRESHOLDS["mean_abs_wt_volume_error_ml_max"],
            "pass": _safe_mean(
                (metrics_df["pred_foreground_volume_ml"] - metrics_df["gt_foreground_volume_ml"]).abs()
            )
            <= RED_FLAG_THRESHOLDS["mean_abs_wt_volume_error_ml_max"],
        },
        {
            "metric_name": "label_1_lt_1ml_zero_dice_count",
            "current_value": float(l1_small["zero_dice_count"]),
            "threshold_type": "compare_to_baseline",
            "threshold_value": np.nan,
            "pass": True,
        },
        {
            "metric_name": "label_4_lt_1ml_zero_dice_count",
            "current_value": float(l4_small["zero_dice_count"]),
            "threshold_type": "compare_to_baseline",
            "threshold_value": np.nan,
            "pass": True,
        },
        {
            "metric_name": "combined_confusion_rate",
            "current_value": confusion.get("combined_confusion_rate", float("nan")),
            "threshold_type": "compare_to_baseline",
            "threshold_value": np.nan,
            "pass": True,
        },
    ]
    return pd.DataFrame(rows)


def build_baseline_comparison(
    current_subgroup_df: pd.DataFrame,
    current_confusion_df: pd.DataFrame,
    current_guardrail_df: pd.DataFrame,
    baseline_subgroup_df: pd.DataFrame,
    baseline_confusion_df: pd.DataFrame,
    baseline_guardrail_df: pd.DataFrame,
) -> pd.DataFrame:
    subgroup_keys = ["label_name", "bucket_name"]
    subgroup_compare = current_subgroup_df.merge(
        baseline_subgroup_df,
        on=subgroup_keys,
        suffixes=("_current", "_baseline"),
    )
    subgroup_rows: List[Dict[str, object]] = []
    for row in subgroup_compare.itertuples(index=False):
        subgroup_rows.extend(
            [
                {
                    "group": "subgroup",
                    "name": f"{row.label_name}:{row.bucket_name}:mean_dice_active",
                    "current_value": row.mean_dice_active_current,
                    "baseline_value": row.mean_dice_active_baseline,
                    "delta": row.mean_dice_active_current - row.mean_dice_active_baseline,
                },
                {
                    "group": "subgroup",
                    "name": f"{row.label_name}:{row.bucket_name}:recall",
                    "current_value": row.recall_current,
                    "baseline_value": row.recall_baseline,
                    "delta": row.recall_current - row.recall_baseline,
                },
                {
                    "group": "subgroup",
                    "name": f"{row.label_name}:{row.bucket_name}:zero_dice_count",
                    "current_value": row.zero_dice_count_current,
                    "baseline_value": row.zero_dice_count_baseline,
                    "delta": row.zero_dice_count_current - row.zero_dice_count_baseline,
                },
                {
                    "group": "subgroup",
                    "name": f"{row.label_name}:{row.bucket_name}:mean_abs_volume_error_ml",
                    "current_value": row.mean_abs_volume_error_ml_current,
                    "baseline_value": row.mean_abs_volume_error_ml_baseline,
                    "delta": row.mean_abs_volume_error_ml_current
                    - row.mean_abs_volume_error_ml_baseline,
                },
            ]
        )

    confusion_compare = current_confusion_df.merge(
        baseline_confusion_df,
        on="metric_name",
        suffixes=("_current", "_baseline"),
    )
    confusion_rows = [
        {
            "group": "confusion",
            "name": row.metric_name,
            "current_value": row.value_current,
            "baseline_value": row.value_baseline,
            "delta": row.value_current - row.value_baseline,
        }
        for row in confusion_compare.itertuples(index=False)
    ]

    guardrail_compare = current_guardrail_df.merge(
        baseline_guardrail_df[["metric_name", "current_value"]],
        on="metric_name",
        suffixes=("_current", "_baseline"),
    )
    guardrail_rows = [
        {
            "group": "guardrail",
            "name": row.metric_name,
            "current_value": row.current_value_current,
            "baseline_value": row.current_value_baseline,
            "delta": row.current_value_current - row.current_value_baseline,
        }
        for row in guardrail_compare.itertuples(index=False)
    ]

    return pd.DataFrame(subgroup_rows + confusion_rows + guardrail_rows)


def _metric_from_df(df: pd.DataFrame, name: str, column: str = "current_value") -> float:
    row = df.loc[df["metric_name"] == name]
    if len(row) == 0:
        return float("nan")
    return float(row.iloc[0][column])


def evaluate_phase2a_decision(
    current_metrics_df: pd.DataFrame,
    current_subgroup_df: pd.DataFrame,
    current_confusion_df: pd.DataFrame,
    current_guardrail_df: pd.DataFrame,
    baseline_metrics_df: Optional[pd.DataFrame] = None,
    baseline_subgroup_df: Optional[pd.DataFrame] = None,
    baseline_confusion_df: Optional[pd.DataFrame] = None,
) -> ExperimentDecision:
    current_confusion = _confusion_lookup(current_confusion_df)

    red_flags = {
        "mean_foreground_dice_below_floor": not bool(
            _metric_from_df(current_guardrail_df, "mean_foreground_dice")
            >= RED_FLAG_THRESHOLDS["mean_foreground_dice_min"]
        ),
        "label_2_mean_dice_below_floor": not bool(
            _metric_from_df(current_guardrail_df, "label_2_mean_dice_active")
            >= RED_FLAG_THRESHOLDS["label_2_mean_dice_min"]
        ),
        "label_3_mean_dice_below_floor": not bool(
            _metric_from_df(current_guardrail_df, "label_3_mean_dice_active")
            >= RED_FLAG_THRESHOLDS["label_3_mean_dice_min"]
        ),
        "wt_volume_error_above_ceiling": not bool(
            _metric_from_df(current_guardrail_df, "mean_abs_wt_volume_error_ml")
            <= RED_FLAG_THRESHOLDS["mean_abs_wt_volume_error_ml_max"]
        ),
    }

    label_1_signed_error = _safe_mean(current_metrics_df["label_1_volume_error_ml"])
    label_4_signed_error = _safe_mean(current_metrics_df["label_4_volume_error_ml"])
    label_4_suppression_ml = float(max(0.0, -label_4_signed_error))

    pain_point_successes = {
        "label_1_small_volume_improved": False,
        "label_4_small_volume_improved": False,
        "label_1_label_4_confusion_improved": False,
    }

    label_1_volume_inflation_delta_ml: Optional[float] = None
    label_4_suppression_delta_ml: Optional[float] = None
    label_1_to_label_4_confusion_change: Optional[float] = None
    label_4_to_label_1_confusion_change: Optional[float] = None
    combined_confusion_change: Optional[float] = None

    if baseline_metrics_df is not None and baseline_subgroup_df is not None and baseline_confusion_df is not None:
        baseline_label_1_signed_error = _safe_mean(baseline_metrics_df["label_1_volume_error_ml"])
        baseline_label_4_signed_error = _safe_mean(baseline_metrics_df["label_4_volume_error_ml"])
        baseline_l1_small = baseline_subgroup_df[
            (baseline_subgroup_df["label_name"] == "label_1")
            & (baseline_subgroup_df["bucket_name"] == "lt_1ml")
        ].iloc[0]
        baseline_l4_small = baseline_subgroup_df[
            (baseline_subgroup_df["label_name"] == "label_4")
            & (baseline_subgroup_df["bucket_name"] == "lt_1ml")
        ].iloc[0]
        current_l1_small = current_subgroup_df[
            (current_subgroup_df["label_name"] == "label_1")
            & (current_subgroup_df["bucket_name"] == "lt_1ml")
        ].iloc[0]
        current_l4_small = current_subgroup_df[
            (current_subgroup_df["label_name"] == "label_4")
            & (current_subgroup_df["bucket_name"] == "lt_1ml")
        ].iloc[0]

        baseline_confusion = _confusion_lookup(baseline_confusion_df)
        label_1_to_label_4_confusion_change = (
            current_confusion.get("gt_label_1_only_pred_label_4_present_rate", float("nan"))
            - baseline_confusion.get("gt_label_1_only_pred_label_4_present_rate", float("nan"))
        )
        label_4_to_label_1_confusion_change = (
            current_confusion.get("gt_label_4_only_pred_label_1_present_rate", float("nan"))
            - baseline_confusion.get("gt_label_4_only_pred_label_1_present_rate", float("nan"))
        )
        combined_confusion_change = (
            current_confusion.get("combined_confusion_rate", float("nan"))
            - baseline_confusion.get("combined_confusion_rate", float("nan"))
        )
        label_1_volume_inflation_delta_ml = label_1_signed_error - baseline_label_1_signed_error
        label_4_suppression_delta_ml = label_4_suppression_ml - max(
            0.0, -baseline_label_4_signed_error
        )

        pain_point_successes["label_1_small_volume_improved"] = bool(
            current_l1_small["mean_dice_active"] > baseline_l1_small["mean_dice_active"]
            or current_l1_small["zero_dice_count"] < baseline_l1_small["zero_dice_count"]
        )
        pain_point_successes["label_4_small_volume_improved"] = bool(
            current_l4_small["recall"] > baseline_l4_small["recall"]
            or current_l4_small["zero_dice_count"] < baseline_l4_small["zero_dice_count"]
        )
        pain_point_successes["label_1_label_4_confusion_improved"] = bool(
            current_confusion.get("combined_confusion_rate", float("nan"))
            < baseline_confusion.get("combined_confusion_rate", float("nan"))
            and current_confusion.get(
                "gt_label_1_only_pred_label_4_present_rate", float("nan")
            )
            <= baseline_confusion.get(
                "gt_label_1_only_pred_label_4_present_rate", float("nan")
            )
            and current_confusion.get(
                "gt_label_4_only_pred_label_1_present_rate", float("nan")
            )
            <= baseline_confusion.get(
                "gt_label_4_only_pred_label_1_present_rate", float("nan")
            )
        )

        red_flags["label_1_small_zero_dice_increase"] = bool(
            current_l1_small["zero_dice_count"] > baseline_l1_small["zero_dice_count"]
        )
        red_flags["label_4_small_zero_dice_increase"] = bool(
            current_l4_small["zero_dice_count"] > baseline_l4_small["zero_dice_count"]
        )
        red_flags["label_1_label_4_confusion_increase"] = bool(
            current_confusion.get("combined_confusion_rate", float("nan"))
            > baseline_confusion.get("combined_confusion_rate", float("nan"))
        )
    else:
        red_flags["label_1_small_zero_dice_increase"] = False
        red_flags["label_4_small_zero_dice_increase"] = False
        red_flags["label_1_label_4_confusion_increase"] = False

    pain_point_success_count = int(sum(1 for value in pain_point_successes.values() if value))
    rollback_required = any(red_flags.values())
    overall_pass = (not rollback_required) and pain_point_success_count >= 2

    return ExperimentDecision(
        overall_pass=overall_pass,
        rollback_required=rollback_required,
        pain_point_success_count=pain_point_success_count,
        pain_point_successes=pain_point_successes,
        red_flags=red_flags,
        label_1_volume_inflation_ml=float(max(0.0, label_1_signed_error)),
        label_1_volume_inflation_delta_ml=label_1_volume_inflation_delta_ml,
        label_4_mean_signed_volume_error_ml=label_4_signed_error,
        label_4_suppression_ml=label_4_suppression_ml,
        label_4_suppression_delta_ml=label_4_suppression_delta_ml,
        label_1_to_label_4_confusion_change=label_1_to_label_4_confusion_change,
        label_4_to_label_1_confusion_change=label_4_to_label_1_confusion_change,
        combined_confusion_change=combined_confusion_change,
    )


def run_phase2a_validation_analysis(
    current_metrics_csv: Path,
    output_dir: Path,
    baseline_metrics_csv: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_metrics_df = _load_metrics_df(Path(current_metrics_csv))
    current_subgroup_df = build_subgroup_summary(current_metrics_df)
    current_confusion_df = build_confusion_summary(current_metrics_df)
    current_guardrail_df = build_guardrail_summary(
        current_metrics_df, current_subgroup_df, current_confusion_df
    )

    baseline_metrics_df = None
    baseline_subgroup_df = None
    baseline_confusion_df = None
    baseline_guardrail_df = None
    baseline_comparison_df = None
    if baseline_metrics_csv is not None:
        baseline_metrics_df = _load_metrics_df(Path(baseline_metrics_csv))
        baseline_subgroup_df = build_subgroup_summary(baseline_metrics_df)
        baseline_confusion_df = build_confusion_summary(baseline_metrics_df)
        baseline_guardrail_df = build_guardrail_summary(
            baseline_metrics_df, baseline_subgroup_df, baseline_confusion_df
        )
        baseline_comparison_df = build_baseline_comparison(
            current_subgroup_df,
            current_confusion_df,
            current_guardrail_df,
            baseline_subgroup_df,
            baseline_confusion_df,
            baseline_guardrail_df,
        )

    decision = evaluate_phase2a_decision(
        current_metrics_df=current_metrics_df,
        current_subgroup_df=current_subgroup_df,
        current_confusion_df=current_confusion_df,
        current_guardrail_df=current_guardrail_df,
        baseline_metrics_df=baseline_metrics_df,
        baseline_subgroup_df=baseline_subgroup_df,
        baseline_confusion_df=baseline_confusion_df,
    )

    current_subgroup_path = output_dir / "subgroup_summary.csv"
    current_confusion_path = output_dir / "confusion_summary.csv"
    current_guardrail_path = output_dir / "guardrail_summary.csv"
    decision_path = output_dir / "decision_summary.json"
    baseline_comparison_path = output_dir / "baseline_comparison_summary.csv"

    current_subgroup_df.to_csv(current_subgroup_path, index=False)
    current_confusion_df.to_csv(current_confusion_path, index=False)
    current_guardrail_df.to_csv(current_guardrail_path, index=False)
    decision_path.write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
    if baseline_comparison_df is not None:
        baseline_comparison_df.to_csv(baseline_comparison_path, index=False)

    return {
        "subgroup_summary_csv": str(current_subgroup_path),
        "confusion_summary_csv": str(current_confusion_path),
        "guardrail_summary_csv": str(current_guardrail_path),
        "decision_summary_json": str(decision_path),
        "baseline_comparison_summary_csv": str(baseline_comparison_path)
        if baseline_comparison_df is not None
        else None,
        "decision": asdict(decision),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2A validation analysis with subgroup/confusion/guardrail outputs."
    )
    parser.add_argument("--current-metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics-csv", default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_phase2a_validation_analysis(
        current_metrics_csv=Path(args.current_metrics_csv),
        output_dir=Path(args.output_dir),
        baseline_metrics_csv=Path(args.baseline_metrics_csv)
        if args.baseline_metrics_csv
        else None,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
