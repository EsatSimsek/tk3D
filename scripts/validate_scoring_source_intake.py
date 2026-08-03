from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poomsae_scoring.source_intake import inspect_source_intake, load_source_intake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a candidate scoring PDF without activating rules or numeric tolerances."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", default=str(ROOT))
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_source_intake(
        load_source_intake(args.manifest),
        workspace_root=args.workspace_root,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError(f"source intake report already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
        print(target.resolve())
    else:
        print(rendered)
    return 0 if report["status"] == "ready_for_manual_activation_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())

