from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.utils.paths import get_data_raw_dir, resolve_project_root

PATIENT_PATTERN = re.compile(r"^PatientID_\d+$")
TIMEPOINT_PATTERN = re.compile(r"^Timepoint_\d+$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect raw dataset structure in read-only Phase 1A mode."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional raw directory override. Default: <project_root>/data/raw",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for structure summary.",
    )
    parser.add_argument(
        "--md-out",
        default=None,
        help="Optional markdown output path for structure summary.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of sampled items to keep in summary lists.",
    )
    return parser


def _canonical_suffix(path: Path) -> str:
    suffix = "".join(path.suffixes[-2:]).lower()
    if suffix == ".nii.gz":
        return suffix
    return path.suffix.lower() or "<no_extension>"


def _append_limited(items: List[Any], value: Any, max_samples: int) -> None:
    if len(items) < max_samples:
        items.append(value)


def inspect_raw_structure(raw_dir: Path, max_samples: int = 10) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "ok",
        "raw_dir": str(raw_dir),
        "patient_count": 0,
        "top_level_file_count": 0,
        "top_level_unexpected_dir_count": 0,
        "timepoint_count": 0,
        "timepoint_count_distribution": {},
        "file_count_per_timepoint_distribution": {},
        "extension_distribution": {},
        "samples": {
            "patient_ids": [],
            "timepoint_paths": [],
            "file_paths": [],
            "top_level_unexpected_dirs": [],
            "top_level_files": [],
            "suspicious_entries": [],
        },
    }

    if not raw_dir.exists() or not raw_dir.is_dir():
        summary["status"] = "error"
        summary["error"] = "Raw directory is missing or invalid."
        return summary

    timepoint_count_dist: Counter[int] = Counter()
    file_count_dist: Counter[int] = Counter()
    extension_dist: Counter[str] = Counter()

    top_entries = sorted(raw_dir.iterdir(), key=lambda p: p.name.lower())
    patient_dirs = [p for p in top_entries if p.is_dir() and PATIENT_PATTERN.match(p.name)]
    unexpected_top_dirs = [p for p in top_entries if p.is_dir() and not PATIENT_PATTERN.match(p.name)]
    top_files = [p for p in top_entries if p.is_file()]

    summary["patient_count"] = len(patient_dirs)
    summary["top_level_unexpected_dir_count"] = len(unexpected_top_dirs)
    summary["top_level_file_count"] = len(top_files)
    for entry in unexpected_top_dirs:
        _append_limited(summary["samples"]["top_level_unexpected_dirs"], entry.name, max_samples)
    for entry in top_files:
        _append_limited(summary["samples"]["top_level_files"], entry.name, max_samples)

    total_timepoints = 0
    for patient_dir in patient_dirs:
        _append_limited(summary["samples"]["patient_ids"], patient_dir.name, max_samples)
        patient_children = sorted(patient_dir.iterdir(), key=lambda p: p.name.lower())
        timepoint_dirs = [
            p for p in patient_children if p.is_dir() and TIMEPOINT_PATTERN.match(p.name)
        ]
        patient_non_timepoint_entries = [
            p for p in patient_children if not (p.is_dir() and TIMEPOINT_PATTERN.match(p.name))
        ]

        if patient_non_timepoint_entries:
            _append_limited(
                summary["samples"]["suspicious_entries"],
                {
                    "type": "patient_non_timepoint_entry",
                    "path": str(patient_dir),
                    "entries": [p.name for p in patient_non_timepoint_entries[:5]],
                },
                max_samples,
            )

        timepoint_count_dist[len(timepoint_dirs)] += 1
        total_timepoints += len(timepoint_dirs)

        for timepoint_dir in timepoint_dirs:
            _append_limited(
                summary["samples"]["timepoint_paths"],
                str(timepoint_dir.relative_to(raw_dir)),
                max_samples,
            )
            entries = sorted(timepoint_dir.iterdir(), key=lambda p: p.name.lower())
            files = [p for p in entries if p.is_file()]
            subdirs = [p for p in entries if p.is_dir()]

            file_count_dist[len(files)] += 1
            if subdirs:
                _append_limited(
                    summary["samples"]["suspicious_entries"],
                    {
                        "type": "timepoint_contains_subdir",
                        "path": str(timepoint_dir),
                        "entries": [p.name for p in subdirs[:5]],
                    },
                    max_samples,
                )

            if len(files) == 0:
                _append_limited(
                    summary["samples"]["suspicious_entries"],
                    {
                        "type": "timepoint_without_files",
                        "path": str(timepoint_dir),
                    },
                    max_samples,
                )

            for file_path in files:
                extension_dist[_canonical_suffix(file_path)] += 1
                _append_limited(
                    summary["samples"]["file_paths"],
                    str(file_path.relative_to(raw_dir)),
                    max_samples,
                )

    summary["timepoint_count"] = total_timepoints
    summary["timepoint_count_distribution"] = dict(sorted(timepoint_count_dist.items()))
    summary["file_count_per_timepoint_distribution"] = dict(sorted(file_count_dist.items()))
    summary["extension_distribution"] = dict(
        sorted(extension_dist.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Raw Dataset Structure Summary")
    lines.append("")
    lines.append(f"- Status: `{summary.get('status', 'unknown')}`")
    lines.append(f"- Raw dir: `{summary.get('raw_dir', '')}`")

    if summary.get("status") != "ok":
        lines.append(f"- Error: {summary.get('error', 'unknown error')}")
        return "\n".join(lines) + "\n"

    lines.append(f"- Patient count: `{summary['patient_count']}`")
    lines.append(f"- Timepoint count: `{summary['timepoint_count']}`")
    lines.append(
        f"- Top-level unexpected dir count: `{summary['top_level_unexpected_dir_count']}`"
    )
    lines.append(f"- Top-level file count: `{summary['top_level_file_count']}`")
    lines.append("")
    lines.append("## Distributions")
    lines.append("")
    lines.append(f"- Timepoints per patient: `{summary['timepoint_count_distribution']}`")
    lines.append(
        "- Files per timepoint: "
        f"`{summary['file_count_per_timepoint_distribution']}`"
    )
    lines.append(f"- Extensions: `{summary['extension_distribution']}`")
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    lines.append(f"- Patients: `{summary['samples']['patient_ids']}`")
    lines.append(f"- Timepoints: `{summary['samples']['timepoint_paths']}`")
    lines.append(f"- Files: `{summary['samples']['file_paths']}`")
    lines.append(
        "- Suspicious entries: "
        f"`{summary['samples']['suspicious_entries']}`"
    )
    lines.append("")
    return "\n".join(lines)


def _resolve_output_path(project_root: Path, output_path: str) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    raw_dir = (
        Path(args.raw_dir).expanduser().resolve()
        if args.raw_dir
        else get_data_raw_dir(str(project_root))
    )
    max_samples = max(1, int(args.max_samples))

    summary = inspect_raw_structure(raw_dir=raw_dir, max_samples=max_samples)
    print("=== Raw Dataset Structure Inspection (Phase 1A) ===")
    print(f"Raw dir: {summary.get('raw_dir')}")
    print(f"Status: {summary.get('status')}")
    if summary.get("status") == "ok":
        print(
            f"Patients={summary['patient_count']} | "
            f"Timepoints={summary['timepoint_count']} | "
            f"Extensions={len(summary['extension_distribution'])}"
        )
        print(
            "Timepoints per patient distribution: "
            f"{summary['timepoint_count_distribution']}"
        )
        print(
            "Files per timepoint distribution: "
            f"{summary['file_count_per_timepoint_distribution']}"
        )
    else:
        print(f"Error: {summary.get('error')}")

    if args.json_out:
        json_out = _resolve_output_path(project_root, args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON report written: {json_out}")

    if args.md_out:
        md_out = _resolve_output_path(project_root, args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_markdown(summary), encoding="utf-8")
        print(f"Markdown report written: {md_out}")

    if summary.get("status") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
