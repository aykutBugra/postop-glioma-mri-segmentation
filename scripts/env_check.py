from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Allow running as "python scripts/env_check.py" from project root.
try:
    from src.utils.paths import resolve_paths, resolve_project_root
except ModuleNotFoundError:
    ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
    if str(ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(ROOT_FOR_IMPORT))
    from src.utils.paths import resolve_paths, resolve_project_root


REQUIRED_DIRS = [
    "data/raw",
    "data/metadata",
    "data/sample",
    "src",
    "scripts",
    "notebooks",
    "docs",
    "outputs",
]

EXCEL_SUFFIXES = {".xls", ".xlsx"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class CheckResult:
    name: str
    path: str
    ok: bool
    detail: str
    critical: bool = True


def _metadata_scan(metadata_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "metadata_dir": str(metadata_dir),
        "exists": metadata_dir.exists() and metadata_dir.is_dir(),
        "file_count": 0,
        "readme_count": 0,
        "excel_count": 0,
        "pdf_count": 0,
        "example_files": [],
    }
    if not metadata_dir.exists() or not metadata_dir.is_dir():
        return summary

    files = sorted([p for p in metadata_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    summary["file_count"] = len(files)
    summary["readme_count"] = sum(1 for f in files if f.name.lower().startswith("readme"))
    summary["excel_count"] = sum(1 for f in files if f.suffix.lower() in EXCEL_SUFFIXES)
    summary["pdf_count"] = sum(1 for f in files if f.suffix.lower() in PDF_SUFFIXES)
    summary["example_files"] = [f.name for f in files[:10]]
    return summary


def run_checks(project_root: Path) -> tuple[List[CheckResult], Dict[str, Any]]:
    checks: List[CheckResult] = []
    for rel in REQUIRED_DIRS:
        path = project_root / rel
        checks.append(
            CheckResult(
                name=f"dir:{rel}",
                path=str(path),
                ok=path.exists() and path.is_dir(),
                detail=f"Directory exists: {path}",
                critical=True,
            )
        )

    metadata_dir = project_root / "data" / "metadata"
    metadata = _metadata_scan(metadata_dir)
    checks.extend(
        [
            CheckResult(
                name="metadata:readme",
                path=str(metadata_dir),
                ok=metadata["readme_count"] > 0,
                detail="README file present in data/metadata",
                critical=True,
            ),
            CheckResult(
                name="metadata:excel",
                path=str(metadata_dir),
                ok=metadata["excel_count"] > 0,
                detail="Excel file (.xls/.xlsx) present in data/metadata",
                critical=True,
            ),
            CheckResult(
                name="metadata:pdf",
                path=str(metadata_dir),
                ok=metadata["pdf_count"] > 0,
                detail="PDF file present in data/metadata",
                critical=True,
            ),
        ]
    )
    return checks, metadata


def _build_summary(
    checks: Iterable[CheckResult],
    project_root: Path,
    metadata_scan: Dict[str, Any],
) -> Dict[str, Any]:
    checks_list = [asdict(c) for c in checks]
    total_checks = len(checks_list)
    failed_checks = [c for c in checks_list if not c["ok"]]
    failed_critical = [c for c in failed_checks if c["critical"]]
    status = "ok"
    if failed_critical:
        status = "error"
    elif failed_checks:
        status = "warning"

    return {
        "status": status,
        "project_root": str(project_root),
        "counts": {
            "total_checks": total_checks,
            "failed_checks": len(failed_checks),
            "failed_critical_checks": len(failed_critical),
        },
        "checks": checks_list,
        "metadata_scan": metadata_scan,
    }


def _print_report(summary: Dict[str, Any]) -> None:
    print("=== Phase 1A Environment Check (Read-Only) ===")
    print(f"Project root: {summary['project_root']}")
    print(f"Status: {summary['status'].upper()}")
    print("")

    for c in summary["checks"]:
        status = "OK" if c["ok"] else "MISSING"
        print(f"[{status}] {c['name']} -> {c['path']}")

    metadata = summary["metadata_scan"]
    print("")
    print(
        "Metadata files: "
        f"total={metadata['file_count']}, "
        f"readme={metadata['readme_count']}, "
        f"excel={metadata['excel_count']}, "
        f"pdf={metadata['pdf_count']}"
    )
    if metadata["example_files"]:
        print(f"Metadata examples: {', '.join(metadata['example_files'])}")
    counts = summary["counts"]
    print(
        "Summary: "
        f"total_checks={counts['total_checks']}, "
        f"failed={counts['failed_checks']}, "
        f"failed_critical={counts['failed_critical_checks']}"
    )
    print("Note: This script does not modify files or directories.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only environment checks for teknofest_proje_3 Phase 1A."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override. Default: auto-detected root.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for machine-readable summary.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    _ = resolve_paths(str(project_root))  # path resolution smoke check

    checks, metadata = run_checks(project_root)
    summary = _build_summary(checks, project_root, metadata)
    _print_report(summary)

    if args.json_out:
        json_out = Path(args.json_out)
        if not json_out.is_absolute():
            json_out = (project_root / json_out).resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON report written: {json_out}")

    has_critical_failures = any((not c.ok) and c.critical for c in checks)
    if has_critical_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
