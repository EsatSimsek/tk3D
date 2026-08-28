from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.poomsae_scoring.run_history import (  # noqa: E402
    build_run_history,
    build_run_history_html,
    dump_report_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparable TK3D scoring run history report.")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--current-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    current_path = args.current_summary.resolve()
    current = _read_json(current_path)
    records = []
    for path in runs_root.glob("*/json/poomsae_scoring_summary.json"):
        resolved = path.resolve()
        if resolved == current_path:
            continue
        try:
            summary = _read_json(resolved)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        records.append(
            {
                "summary": summary,
                "summary_path": str(resolved),
                "modified_at_utc": datetime.fromtimestamp(
                    resolved.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    report = build_run_history(current, records)
    output_json = args.output_json.resolve()
    output_html = args.output_html.resolve()
    for target in (output_json, output_html):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(dump_report_json(report), encoding="utf-8", newline="")
    output_html.write_text(build_run_history_html(report), encoding="utf-8", newline="")
    print(output_json)
    print(output_html)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


if __name__ == "__main__":
    main()
