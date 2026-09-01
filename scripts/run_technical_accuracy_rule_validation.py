from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

import src.poomsae_scoring.technical_accuracy as technical_accuracy_module
import src.poomsae_scoring.technical_accuracy_metrics as technical_accuracy_metrics_module
import src.poomsae_scoring.technical_accuracy_validation as validation_module
from src.poomsae_scoring import (
    build_rule_accuracy_validation,
    load_movement_timeline,
    load_poomsae_spec,
    load_rule_accuracy_validation_profile,
    load_technical_accuracy_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run score-neutral synthetic validation for the Taegeuk 1 technical-accuracy rule system."
    )
    parser.add_argument("--validation-profile", type=Path, required=True)
    parser.add_argument("--technical-profile", type=Path, required=True)
    parser.add_argument("--poomsae-spec", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rule-inventory-csv", type=Path, required=True)
    parser.add_argument("--landmark-inventory-csv", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--scenario-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [
        args.output_json,
        args.rule_inventory_csv,
        args.landmark_inventory_csv,
        args.classification_csv,
        args.scenario_csv,
    ]
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("validation output paths must be unique")
    output_dirs = {path.parent for path in resolved_outputs}
    if len(output_dirs) != 1:
        raise ValueError("all validation outputs must share one unique run directory")
    output_dir = next(iter(output_dirs))
    if output_dir.exists():
        raise FileExistsError(f"validation run directory already exists; refusing overwrite: {output_dir}")

    profile = load_technical_accuracy_profile(args.technical_profile)
    spec = load_poomsae_spec(args.poomsae_spec)
    timeline = load_movement_timeline(args.timeline, spec)
    validation = load_rule_accuracy_validation_profile(args.validation_profile)
    report = build_rule_accuracy_validation(profile, spec, timeline, validation)
    report["input_files"] = {
        label: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for label, path in (
            ("validation_profile", args.validation_profile),
            ("technical_profile", args.technical_profile),
            ("poomsae_spec", args.poomsae_spec),
            ("movement_timeline", args.timeline),
        )
    }
    implementation_paths = {
        "validation_core": Path(validation_module.__file__).resolve(),
        "technical_accuracy_core": Path(technical_accuracy_module.__file__).resolve(),
        "technical_accuracy_metrics": Path(technical_accuracy_metrics_module.__file__).resolve(),
        "validation_cli": Path(__file__).resolve(),
    }
    report["implementation_files"] = {
        label: {"path": str(path), "sha256": _sha256(path)}
        for label, path in implementation_paths.items()
    }
    report["runtime"] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.partial-{uuid4().hex}")
    staging_dir.mkdir(exist_ok=False)
    staged_paths = {
        args.output_json: staging_dir / args.output_json.name,
        args.rule_inventory_csv: staging_dir / args.rule_inventory_csv.name,
        args.landmark_inventory_csv: staging_dir / args.landmark_inventory_csv.name,
        args.classification_csv: staging_dir / args.classification_csv.name,
        args.scenario_csv: staging_dir / args.scenario_csv.name,
    }
    try:
        _write_text_exclusive(
            staged_paths[args.output_json],
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        _write_csv(staged_paths[args.rule_inventory_csv], report["rule_validation_inventory"])
        _write_csv(staged_paths[args.landmark_inventory_csv], report["landmark_validation_inventory"])
        _write_csv(staged_paths[args.classification_csv], report["classification_cases"])
        _write_csv(staged_paths[args.scenario_csv], report["geometry_scenarios"])
        manifest = {
            "schema_version": 1,
            "validation_id": report["validation_id"],
            "status": report["status"],
            "run_id": output_dir.name,
            "artifacts": [
                _artifact_record(staged_paths[path], staging_dir) for path in outputs
            ],
        }
        _write_text_exclusive(
            staging_dir / "validation_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        staging_dir.rename(output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    manifest_path = output_dir / "validation_manifest.json"
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty validation table: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(output_dir).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
