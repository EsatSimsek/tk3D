from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.artifact_io import write_json_exclusive
from src.reproducibility import ReproducibilityStatus, check_reproducibility


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check TK3D clean-checkout and CURRENT_ACTIVE research readiness without loading models.",
        epilog=(
            "READY means Tier 1 and local CURRENT_ACTIVE assets are available; "
            "PARTIALLY_READY means Tier 1 is ready but one or more local research assets are unavailable."
        ),
    )
    parser.add_argument(
        "--model-config",
        default="config/model_config.yaml",
        help="Model/runtime config to validate (default: config/model_config.yaml).",
    )
    parser.add_argument(
        "--active-profile",
        default="config/scoring/profiles/poomsae1_trimmed.yaml",
        help="CURRENT_ACTIVE Poomsae profile YAML to validate.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Output root whose run location and write readiness are checked (default: outputs).",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable report.")
    parser.add_argument("--write-report", help="Optionally write the report once to this JSON path.")
    args = parser.parse_args()

    report = check_reproducibility(
        ROOT,
        model_config_path=args.model_config,
        active_profile_path=args.active_profile,
        output_root=args.output_root,
    )
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(report.status.value)
        print(f"Tier 1 clean/lightweight: {'READY' if report.tier1_ready else 'NOT_READY'}")
        print(f"CURRENT_ACTIVE local research: {'READY' if report.current_active_ready else 'NOT_READY'}")
        for check in report.checks:
            if check.status != "PASS":
                print(f"[{check.status}] {check.name}: {check.message}")
    if args.write_report:
        destination = Path(args.write_report)
        if not destination.is_absolute():
            destination = ROOT / destination
        write_json_exclusive(destination, payload, sort_keys=True)
        print(f"saved: {destination.resolve()}")
    return 1 if report.status is ReproducibilityStatus.NOT_READY else 0


if __name__ == "__main__":
    raise SystemExit(main())
