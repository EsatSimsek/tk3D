from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.poomsae_scoring import (  # noqa: E402
    ScoringContractError,
    assess_accuracy_readiness,
    load_movement_timeline,
    load_poomsae_spec,
    load_rule_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a RulePack, PoomsaeSpec and exact pose timeline may enter Accuracy scoring."
    )
    parser.add_argument("--rule-pack", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--workspace-root", default=str(ROOT))
    parser.add_argument("--wholebody-diagnostics")
    parser.add_argument("--output", help="Write JSON to a new file; an existing file is never overwritten.")
    args = parser.parse_args()

    try:
        rule_pack = load_rule_pack(_resolve(args.rule_pack, ROOT))
        poomsae_spec = load_poomsae_spec(_resolve(args.poomsae_spec, ROOT))
        timeline = load_movement_timeline(_resolve(args.timeline, ROOT), poomsae_spec)
        diagnostics = (
            json.loads(_resolve(args.wholebody_diagnostics, ROOT).read_text(encoding="utf-8"))
            if args.wholebody_diagnostics
            else None
        )
        report = assess_accuracy_readiness(
            rule_pack,
            poomsae_spec,
            timeline,
            workspace_root=_resolve(args.workspace_root, ROOT),
            wholebody_diagnostics=diagnostics,
        )
    except ScoringContractError as exc:
        raise SystemExit(f"Scoring contract error: {exc}") from exc

    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
        return

    output_path = _resolve(args.output, ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", errors="strict", newline="") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise SystemExit(f"Output already exists; refusing to overwrite: {output_path}") from exc
    print(output_path)


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    main()
