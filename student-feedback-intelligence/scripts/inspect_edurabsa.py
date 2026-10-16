"""
Phase 1 — EduRABSA Dataset Inspector

Purpose:
    Recursively inspect the actual EduRABSA directory before any ML,
    preprocessing, or dataset-specific assumptions are introduced.

Usage from the project root:
    python scripts/inspect_edurabsa.py

The script:
    - recursively discovers files
    - reports file types and sizes
    - inspects supported tabular/text formats
    - reports columns/keys and record counts where possible
    - reports missing values and duplicate rows for tabular files
    - samples JSON/JSONL structures without modifying source data
    - writes a JSON inspection report to docs/edurabsa_inspection_report.json

It does NOT:
    - modify raw files
    - rename files
    - clean data
    - drop records
    - train models
    - invent labels or schema
"""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "edurabsa"
REPORT_DIR = PROJECT_ROOT / "docs"
REPORT_PATH = REPORT_DIR / "edurabsa_inspection_report.json"

SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_TABULAR = {".csv", ".tsv"}
SUPPORTED_JSON = {".json", ".jsonl", ".ndjson"}
SUPPORTED_EXCEL = {".xlsx", ".xls"}


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files() -> list[Path]:
    return sorted(
        p for p in DATA_ROOT.rglob("*")
        if p.is_file()
    )


def inspect_csv(path: Path, delimiter: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": "csv" if path.suffix.lower() == ".csv" else "tsv",
        "columns": [],
        "row_count": None,
        "missing_values": {},
        "duplicate_rows": None,
        "sample_rows": [],
        "error": None,
    }

    try:
        sep = delimiter
        if sep is None:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","

        # Prefer pandas when available because it provides robust CSV profiling.
        try:
            import pandas as pd

            df = pd.read_csv(path, sep=sep, low_memory=False)
            result["columns"] = [str(c) for c in df.columns]
            result["row_count"] = int(len(df))
            result["missing_values"] = {
                str(k): int(v)
                for k, v in df.isna().sum().items()
                if int(v) > 0
            }
            result["duplicate_rows"] = int(df.duplicated().sum())
            result["sample_rows"] = json.loads(
                df.head(5).to_json(orient="records", date_format="iso")
            )
            result["dtypes"] = {str(k): str(v) for k, v in df.dtypes.items()}
            return result
        except ImportError:
            pass

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            rows = list(reader)

        if rows:
            result["columns"] = rows[0]
            result["row_count"] = max(0, len(rows) - 1)
            result["sample_rows"] = [
                dict(zip(rows[0], row))
                for row in rows[1:6]
            ]
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def inspect_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": path.suffix.lower().lstrip("."),
        "structure": None,
        "record_count": None,
        "top_level_keys": [],
        "sample": None,
        "error": None,
    }

    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            records = []
            count = 0
            with path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    count += 1
                    if len(records) < 5:
                        records.append(obj)

            result["structure"] = "jsonl"
            result["record_count"] = count
            result["sample"] = records

            if records and isinstance(records[0], dict):
                result["top_level_keys"] = list(records[0].keys())

            return result

        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)

        result["sample"] = obj if not isinstance(obj, list) else obj[:5]

        if isinstance(obj, list):
            result["structure"] = "array"
            result["record_count"] = len(obj)
            if obj and isinstance(obj[0], dict):
                result["top_level_keys"] = list(obj[0].keys())
        elif isinstance(obj, dict):
            result["structure"] = "object"
            result["record_count"] = 1
            result["top_level_keys"] = list(obj.keys())
        else:
            result["structure"] = type(obj).__name__
            result["record_count"] = 1

        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def inspect_text(path: Path) -> dict[str, Any]:
    result = {
        "format": path.suffix.lower().lstrip("."),
        "line_count": None,
        "character_count": None,
        "sample_lines": [],
        "error": None,
    }

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()

        result["line_count"] = len(lines)
        result["character_count"] = sum(len(x) for x in lines)
        result["sample_lines"] = [x.rstrip("\n") for x in lines[:10]]
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def inspect_excel(path: Path) -> dict[str, Any]:
    result = {
        "format": path.suffix.lower().lstrip("."),
        "sheets": [],
        "error": None,
    }

    try:
        import pandas as pd

        book = pd.ExcelFile(path)
        result["sheets"] = list(book.sheet_names)

        sheet_info = {}
        for sheet in book.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            sheet_info[sheet] = {
                "row_count": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "missing_values": {
                    str(k): int(v)
                    for k, v in df.isna().sum().items()
                    if int(v) > 0
                },
                "duplicate_rows": int(df.duplicated().sum()),
                "sample_rows": json.loads(
                    df.head(5).to_json(orient="records", date_format="iso")
                ),
            }

        result["sheet_details"] = sheet_info
        return result

    except ImportError:
        result["error"] = (
            "pandas/openpyxl is not installed; install project requirements "
            "before inspecting Excel files."
        )
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> None:
    print("=" * 72)
    print("EduRABSA — Phase 1 Dataset Inspection")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print("\nERROR: EduRABSA directory does not exist.")
        print(f"Expected: {DATA_ROOT}")
        print("\nCreate/place the dataset there and run the script again.")
        return

    files = discover_files()

    if not files:
        print("\nNo files found.")
        print("Place the EduRABSA dataset inside data/raw/edurabsa/")
        return

    print(f"\nFiles discovered: {len(files)}")

    extension_counts = Counter(
        (p.suffix.lower() or "[no extension]") for p in files
    )

    print("\nFile types:")
    for ext, count in sorted(extension_counts.items()):
        print(f"  {ext:15} {count}")

    report = {
        "project": "AI-Based Student Feedback Sentiment Intelligence",
        "phase": "Phase 1 — Dataset Inspection",
        "dataset": "EduRABSA",
        "dataset_root": str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        "file_count": len(files),
        "file_types": dict(sorted(extension_counts.items())),
        "files": [],
    }

    for idx, path in enumerate(files, start=1):
        relative = path.relative_to(PROJECT_ROOT)
        print(f"\n[{idx}/{len(files)}] {relative}")
        print(f"  Size: {human_size(path.stat().st_size)}")

        item: dict[str, Any] = {
            "path": str(relative),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "size_human": human_size(path.stat().st_size),
            "sha256": file_sha256(path),
        }

        suffix = path.suffix.lower()

        if suffix in SUPPORTED_TABULAR:
            item["inspection"] = inspect_csv(path)
        elif suffix in SUPPORTED_JSON:
            item["inspection"] = inspect_json(path)
        elif suffix in SUPPORTED_TEXT:
            item["inspection"] = inspect_text(path)
        elif suffix in SUPPORTED_EXCEL:
            item["inspection"] = inspect_excel(path)
        else:
            item["inspection"] = {
                "status": "not_deep_inspected",
                "reason": "Unsupported/unclassified file format at this stage.",
            }

        report["files"].append(item)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Inspection complete.")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    print("\nIMPORTANT:")
    print("This inspection did not modify the raw dataset.")
    print("Send the terminal output/report summary back before we design the ML pipeline.")


if __name__ == "__main__":
    main()
