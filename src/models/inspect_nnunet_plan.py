from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.training_plan import _default_plans_json, _extract_planner_metadata, _resolve_path
from src.utils.paths import resolve_project_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect nnU-Net planner output and print patch/batch/spacing summary. "
            "If planner file is missing, returns pending placeholder summary."
        )
    )
    parser.add_argument("--project-root", default=None, help="Optional project root override.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=901,
        help="Dataset id for default planner path resolution.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Phase1GliomaBaseline",
        help="Dataset name for default planner path resolution.",
    )
    parser.add_argument(
        "--configuration-name",
        default="3d_fullres",
        help="Planner configuration to inspect.",
    )
    parser.add_argument(
        "--plans-json",
        default=None,
        help="Optional explicit nnUNetPlans.json path.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output file path.",
    )
    return parser


def _pending_payload(plans_json: Path, configuration_name: str) -> Dict[str, Any]:
    return {
        "status": "pending",
        "message": "Planner output not found yet. Run plan_and_preprocess first.",
        "plans_json": str(plans_json),
        "planner_metadata": {
            "configuration_name": configuration_name,
            "planned_patch_size": None,
            "planned_batch_size": None,
            "target_spacing": None,
        },
    }


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    plans_json = (
        _resolve_path(project_root, args.plans_json)
        if args.plans_json
        else _default_plans_json(project_root, args.dataset_id, args.dataset_name)
    )

    if not plans_json.exists():
        payload = _pending_payload(plans_json, args.configuration_name)
    else:
        try:
            plans = json.loads(plans_json.read_text(encoding="utf-8"))
            metadata = _extract_planner_metadata(plans, args.configuration_name)
            payload = {
                "status": "resolved",
                "message": "Planner metadata parsed.",
                "plans_json": str(plans_json),
                "planner_metadata": metadata,
            }
        except Exception as exc:
            payload = {
                "status": "pending",
                "message": f"Planner parse failed: {exc}",
                "plans_json": str(plans_json),
                "planner_metadata": {
                    "configuration_name": args.configuration_name,
                    "planned_patch_size": None,
                    "planned_batch_size": None,
                    "target_spacing": None,
                },
            }

    print(json.dumps(payload, indent=2))
    if args.json_out:
        out = _resolve_path(project_root, args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
