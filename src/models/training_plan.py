from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from src.utils.paths import get_phase1_nnunet_output_dir, resolve_project_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 1F training execution artifacts from nnU-Net planner output. "
            "This command does not start training."
        )
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training_plan.json/training_commands.txt.",
    )
    parser.add_argument(
        "--drive-project-root",
        default=None,
        help=(
            "Drive project root used as source/sync base in Colab. "
            "Defaults to --project-root."
        ),
    )
    parser.add_argument(
        "--colab-work-root",
        default="/content/phase1f_run",
        help="Colab working root. nnU-Net runtime folders are planned under this path.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=901,
        help="nnU-Net dataset id used in planning/training commands.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Phase1GliomaBaseline",
        help="nnU-Net dataset name suffix.",
    )
    parser.add_argument(
        "--configuration-name",
        default="3d_fullres",
        help="Planner/training configuration name.",
    )
    parser.add_argument(
        "--plans-json",
        default=None,
        help=(
            "Path to planner output JSON (typically nnUNetPlans.json). "
            "If omitted, resolved from nnUNet_preprocessed env var."
        ),
    )
    parser.add_argument(
        "--strict-planner",
        action="store_true",
        help="Fail if planner output JSON is missing or cannot be parsed.",
    )
    parser.add_argument(
        "--nnunet-raw",
        default=None,
        help="nnUNet_raw path for command plan. Defaults to env or <project_root>/outputs/nnunet_raw.",
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        default=None,
        help="nnUNet_preprocessed path for command plan. Defaults to env or <project_root>/outputs/nnunet_preprocessed.",
    )
    parser.add_argument(
        "--nnunet-results",
        default=None,
        help="nnUNet_results path for command plan. Defaults to env or <project_root>/outputs/nnunet_results.",
    )
    return parser


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _safe_dataset_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(name).strip())
    return cleaned or "Phase1Dataset"


def _default_plans_json(project_root: Path, dataset_id: int, dataset_name: str) -> Path:
    preprocessed_root = os.environ.get("nnUNet_preprocessed")
    if preprocessed_root:
        base = Path(preprocessed_root).expanduser().resolve()
    else:
        base = (project_root / "outputs" / "nnunet_preprocessed").resolve()
    folder = f"Dataset{dataset_id:03d}_{_safe_dataset_name(dataset_name)}"
    return base / folder / "nnUNetPlans.json"


def _as_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return None


def _extract_planner_metadata(plans: Dict[str, Any], configuration_name: str) -> Dict[str, Any]:
    configs = plans.get("configurations")
    if not isinstance(configs, dict):
        raise ValueError("Planner JSON does not contain 'configurations' dictionary.")
    config = configs.get(configuration_name)
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration '{configuration_name}' not found in planner JSON."
        )

    patch_size = _as_list(config.get("patch_size"))
    batch_size = config.get("batch_size")
    target_spacing = _as_list(config.get("spacing")) or _as_list(config.get("target_spacing"))
    if target_spacing is None:
        target_spacing = _as_list(plans.get("target_spacing"))

    if patch_size is None:
        raise ValueError(
            f"Planner metadata missing 'patch_size' for configuration '{configuration_name}'."
        )
    if batch_size is None:
        raise ValueError(
            f"Planner metadata missing 'batch_size' for configuration '{configuration_name}'."
        )
    if target_spacing is None:
        raise ValueError(
            f"Planner metadata missing 'target spacing' for configuration '{configuration_name}'."
        )

    return {
        "configuration_name": configuration_name,
        "planned_patch_size": patch_size,
        "planned_batch_size": batch_size,
        "target_spacing": target_spacing,
    }


def _build_command_blocks(dataset_id: int, configuration_name: str) -> Dict[str, Any]:
    plan_and_preprocess = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c {configuration_name}"
    fold0_smoke = f"nnUNetv2_train {dataset_id} {configuration_name} 0"
    fold0_full = f"nnUNetv2_train {dataset_id} {configuration_name} 0"
    resume_fold0 = f"nnUNetv2_train {dataset_id} {configuration_name} 0 --c"
    validation_rerun = f"nnUNetv2_train {dataset_id} {configuration_name} 0 --val"
    find_best = f"nnUNetv2_find_best_configuration {dataset_id}"
    optional_full_folds = [
        f"nnUNetv2_train {dataset_id} {configuration_name} 1",
        f"nnUNetv2_train {dataset_id} {configuration_name} 2",
        f"nnUNetv2_train {dataset_id} {configuration_name} 3",
        f"nnUNetv2_train {dataset_id} {configuration_name} 4",
    ]
    return {
        "plan_and_preprocess": plan_and_preprocess,
        "fold0_smoke": fold0_smoke,
        "fold0_full": fold0_full,
        "resume_fold0": resume_fold0,
        "validation_rerun": validation_rerun,
        "find_best_configuration": find_best,
        "optional_full_folds": optional_full_folds,
    }


def _flatten_commands(command_blocks: Dict[str, Any]) -> List[str]:
    fold_lines = command_blocks.get("optional_full_folds", [])
    if not isinstance(fold_lines, list):
        fold_lines = []
    return [
        "# Phase 1F note: run these only in Colab GPU, not local.",
        command_blocks["plan_and_preprocess"],
        "",
        "# Fold-0 smoke: start normally, stop after first healthy validation/checkpoint.",
        command_blocks["fold0_smoke"],
        "",
        "# Fold-0 full run:",
        command_blocks["fold0_full"],
        "",
        "# Resume fold-0 if interrupted:",
        command_blocks["resume_fold0"],
        "",
        "# Re-run validation on fold-0 checkpoints:",
        command_blocks["validation_rerun"],
        "",
        "# Optional additional folds after fold-0 is stable:",
        *fold_lines,
        "",
        "# Aggregate/best configuration selection:",
        command_blocks["find_best_configuration"],
    ]


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    output_dir = (
        _resolve_path(project_root, args.output_dir)
        if args.output_dir
        else get_phase1_nnunet_output_dir(str(project_root))
    )
    drive_project_root = (
        _resolve_path(project_root, args.drive_project_root)
        if args.drive_project_root
        else project_root
    )
    plans_json = (
        _resolve_path(project_root, args.plans_json)
        if args.plans_json
        else _default_plans_json(project_root, args.dataset_id, args.dataset_name)
    )
    nnunet_raw = (
        _resolve_path(project_root, args.nnunet_raw)
        if args.nnunet_raw
        else Path(os.environ.get("nnUNet_raw", str((project_root / "outputs" / "nnunet_raw").resolve())))
    )
    nnunet_preprocessed = (
        _resolve_path(project_root, args.nnunet_preprocessed)
        if args.nnunet_preprocessed
        else Path(
            os.environ.get(
                "nnUNet_preprocessed",
                str((project_root / "outputs" / "nnunet_preprocessed").resolve()),
            )
        )
    )
    nnunet_results = (
        _resolve_path(project_root, args.nnunet_results)
        if args.nnunet_results
        else Path(
            os.environ.get(
                "nnUNet_results",
                str((project_root / "outputs" / "nnunet_results").resolve()),
            )
        )
    )
    split_manifest_csv = output_dir / "split_manifest_phase1.csv"
    split_summary_json = output_dir / "split_summary.json"
    held_out_test_ids_txt = output_dir / "held_out_test_case_ids.txt"
    train_case_ids_txt = output_dir / "train_case_ids.txt"
    val_case_ids_txt = output_dir / "val_case_ids.txt"
    splits_final_json = output_dir / "splits_final.json"
    source_of_truth_dir = drive_project_root / "outputs" / "phase1_nnunet_dataset"

    dataset_folder = f"Dataset{args.dataset_id:03d}_{_safe_dataset_name(args.dataset_name)}"
    colab_work_root = PurePosixPath(args.colab_work_root)
    colab_paths = {
        "colab_work_root": str(colab_work_root),
        "nnUNet_raw": str(colab_work_root / "nnunet_raw"),
        "nnUNet_preprocessed": str(colab_work_root / "nnunet_preprocessed"),
        "nnUNet_results": str(colab_work_root / "nnunet_results"),
        "dataset_source_copy": str(colab_work_root / "outputs" / "phase1_nnunet_dataset"),
        "dataset_folder_name": dataset_folder,
    }

    print("=== Phase 1F Training Plan Artifact Builder ===")
    print(f"Project root: {project_root}")
    print(f"Drive project root: {drive_project_root}")
    print(f"Output dir: {output_dir}")
    print(f"Plans JSON: {plans_json}")
    print(f"Configuration: {args.configuration_name}")

    planner_status = "pending"
    planner_message = "Planner output not found yet. Fill after plan_and_preprocess in Colab."
    planner_metadata: Dict[str, Any] = {
        "configuration_name": args.configuration_name,
        "planned_patch_size": None,
        "planned_batch_size": None,
        "target_spacing": None,
    }
    if plans_json.exists():
        try:
            plans = json.loads(plans_json.read_text(encoding="utf-8"))
            planner_metadata = _extract_planner_metadata(plans, args.configuration_name)
            planner_status = "resolved"
            planner_message = "Planner metadata parsed from nnUNetPlans.json."
        except Exception as exc:
            if args.strict_planner:
                print(f"[ERROR] Failed to parse planner metadata: {exc}")
                return 1
            planner_status = "pending"
            planner_message = f"Planner parse failed, placeholder kept: {exc}"
    elif args.strict_planner:
        print("[ERROR] Planner output (nnUNetPlans.json) not found.")
        return 1

    command_blocks = _build_command_blocks(args.dataset_id, args.configuration_name)
    commands = _flatten_commands(command_blocks)
    checkpoint_policy = {
        "name": "gated_lexicographic_v1",
        "gate_rules": [
            "Reject checkpoint if any primary metric is missing or NaN.",
            "Reject checkpoint if RC Dice <= 0.01 or NETC present-case <= 0.01.",
            "Reject checkpoint if absolute WT Volume Error > 1.0.",
        ],
        "ranking_order": [
            "maximize mean(WT Dice, RC Dice, NETC present-case)",
            "minimize abs(WT Volume Error)",
            "maximize ET Dice",
            "earliest epoch as deterministic tie-break",
        ],
        "held_out_test_policy": "Held-out test remains closed during training and checkpoint selection.",
    }
    training_plan = {
        "phase": "Phase 1F",
        "training_started": False,
        "local_training_allowed": False,
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "source_of_truth_output_dir": str(source_of_truth_dir),
        "single_source_of_truth_policy": (
            "Use outputs/phase1_nnunet_dataset only. "
            "Drive is source/sync; runtime execution is on /content."
        ),
        "planner_output_json": str(plans_json),
        "planner_metadata_status": planner_status,
        "planner_metadata_message": planner_message,
        "planner_metadata": planner_metadata,
        "environment": {
            "nnUNet_raw": str(nnunet_raw),
            "nnUNet_preprocessed": str(nnunet_preprocessed),
            "nnUNet_results": str(nnunet_results),
        },
        "colab_staging": {
            "drive_project_root": str(drive_project_root),
            "drive_dataset_source": str(source_of_truth_dir),
            "colab_work_root": colab_paths["colab_work_root"],
            "colab_nnunet_raw": colab_paths["nnUNet_raw"],
            "colab_nnunet_preprocessed": colab_paths["nnUNet_preprocessed"],
            "colab_nnunet_results": colab_paths["nnUNet_results"],
            "colab_dataset_source_copy": colab_paths["dataset_source_copy"],
            "sync_policy": "Copy from Drive to /content before run; sync artifacts back to Drive after run.",
        },
        "split_sources": {
            "split_manifest_phase1_csv": str(split_manifest_csv),
            "split_summary_json": str(split_summary_json),
            "train_case_ids_txt": str(train_case_ids_txt),
            "val_case_ids_txt": str(val_case_ids_txt),
            "held_out_test_case_ids_txt": str(held_out_test_ids_txt),
            "splits_final_json": str(splits_final_json),
            "held_out_test_policy": "Held-out test set is never used for training/model selection in Phase 1F.",
        },
        "checkpoint_selection_policy": checkpoint_policy,
        "command_blocks": command_blocks,
        "notes": [
            "Baseline configuration is 3d_fullres unless updated by explicit plan.",
            "Patch size is planner-driven baseline; no manual patch override in Phase 1F.",
            "Larger patch experiments (e.g., 128^3) must be handled in a separate ablation phase.",
            "Do not run local training in Phase 1F if no GPU environment is available.",
            "Smoke run strategy: start fold-0 and stop manually after first healthy validation/checkpoint.",
        ],
        "prepared_commands": commands,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    training_plan_json = output_dir / "training_plan.json"
    training_commands_txt = output_dir / "training_commands.txt"
    training_checklist_md = output_dir / "training_checklist.md"
    training_plan_json.write_text(json.dumps(training_plan, indent=2), encoding="utf-8")
    training_commands_txt.write_text("\n".join(commands) + "\n", encoding="utf-8")
    training_checklist_md.write_text(
        "\n".join(
            [
                "# Phase 1F Training Checklist",
                "",
                "- [ ] Use only outputs/phase1_nnunet_dataset as source of truth.",
                "- [ ] Stage dataset from Drive to /content runtime before training.",
                "- [ ] Verify split_summary leakage_check is PASS.",
                "- [ ] Verify held_out_test_case_ids.txt is excluded from training/model selection.",
                "- [ ] Run nnUNetv2_plan_and_preprocess in Colab GPU environment.",
                "- [ ] Read planner metadata (patch/batch/spacing/config) and confirm baseline settings.",
                "- [ ] Run fold-0 smoke first; stop after first healthy validation/checkpoint.",
                "- [ ] Run fold-0 full training after smoke checks pass.",
                "- [ ] Select checkpoint with gated-lexicographic multi-metric policy (not single Dice).",
                "- [ ] Use validation metrics for checkpoint selection; keep held-out test closed.",
                "- [ ] Keep test set closed until final evaluation/reporting.",
                "- [ ] Do NOT run training in local non-GPU Phase 1F environment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {training_plan_json}")
    print(f"Wrote: {training_commands_txt}")
    print(f"Wrote: {training_checklist_md}")
    print(f"Planner metadata status: {planner_status}")
    print(f"Planner metadata message: {planner_message}")
    print("Planner metadata:")
    print(json.dumps(planner_metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
