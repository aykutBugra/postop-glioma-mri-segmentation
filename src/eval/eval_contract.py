from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class MetricSpec:
    name: str
    objective: str
    definition: str


METRIC_CONTRACT: List[MetricSpec] = [
    MetricSpec(
        name="WT Dice",
        objective="maximize",
        definition="Sorensen-Dice overlap on whole-tumor voxels between prediction and reference per case/timepoint.",
    ),
    MetricSpec(
        name="WT Volume Error",
        objective="minimize",
        definition="Absolute relative error of whole-tumor volume: |V_pred - V_ref| / max(V_ref, eps).",
    ),
    MetricSpec(
        name="ET Dice",
        objective="maximize",
        definition="Sorensen-Dice overlap on enhancing-tumor voxels between prediction and reference.",
    ),
    MetricSpec(
        name="RC Dice",
        objective="maximize",
        definition="Sorensen-Dice overlap on resection-cavity voxels between prediction and reference.",
    ),
    MetricSpec(
        name="NETC Present-Case Metric",
        objective="maximize",
        definition="NETC-region score computed only on cases where reference NETC is present (empty-reference cases excluded).",
    ),
    MetricSpec(
        name="Longitudinal Decision Metric",
        objective="maximize",
        definition="Agreement of patient-level longitudinal progression decisions across ordered timepoints.",
    ),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print Phase 1 evaluation metric contract (no metric computation)."
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for metric contract.",
    )
    return parser


def _print_text_contract() -> None:
    print("=== Evaluation Contract (Phase 1) ===")
    for idx, metric in enumerate(METRIC_CONTRACT, start=1):
        print(f"{idx}. {metric.name} | objective={metric.objective}")
        print(f"   definition: {metric.definition}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.format == "json":
        print(json.dumps([asdict(m) for m in METRIC_CONTRACT], indent=2))
    else:
        _print_text_contract()

    print("Note: This module defines contract only; no metric is computed in Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
