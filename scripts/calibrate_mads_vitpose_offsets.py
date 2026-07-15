from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_mads_vitpose_adapter import MADS_TO_COCO_BODY
from src.exporter import export_session_json
from src.vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer, _refine_heatmap_peaks_udp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate robust MADS body-joint heatmap offsets without changing ViTPose confidence."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "mads_adapter" / "feature_cache",
    )
    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=["Kata:F3", "Taichi:S6"],
    )
    parser.add_argument(
        "--held-out-test-sequences",
        nargs="+",
        default=["Kata:F2"],
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "weights" / "vitpose_huge_wholebody_256x192.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-offset-heatmap-px", type=float, default=2.0)
    parser.add_argument("--min-validation-improvement-ratio", type=float, default=0.01)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=ROOT / "weights" / "vitpose_huge_wholebody_mads_offsets.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "mads_adapter" / "offset_calibration_report.json",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_offset_heatmap_px <= 0.0:
        raise SystemExit("batch-size and max-offset-heatmap-px must be positive")
    if not 0.0 <= args.min_validation_improvement_ratio < 1.0:
        raise SystemExit("min-validation-improvement-ratio must be in [0, 1)")
    runtime = ViTPosePlusWholeBodyInferencer(args.base_checkpoint.resolve(), args.device)
    train_caches, validation_caches = load_split_caches(
        args.cache_dir,
        validation_sequences=args.validation_sequences,
        held_out_test_sequences=args.held_out_test_sequences,
    )
    train_predicted, train_truth = decode_cache_coordinates(
        runtime, train_caches, batch_size=args.batch_size
    )
    validation_predicted, validation_truth = decode_cache_coordinates(
        runtime, validation_caches, batch_size=args.batch_size
    )
    body_offsets = robust_joint_offsets(
        train_predicted,
        train_truth,
        max_offset_heatmap_px=args.max_offset_heatmap_px,
    )
    train_metrics = coordinate_metrics(train_predicted, train_truth, body_offsets)
    validation_metrics = coordinate_metrics(validation_predicted, validation_truth, body_offsets)
    improvement = validation_metrics["improvement_ratio"]
    if improvement < args.min_validation_improvement_ratio:
        raise SystemExit(
            f"Offset calibration rejected: validation improvement {improvement:.4f} is below "
            f"{args.min_validation_improvement_ratio:.4f}"
        )

    import torch

    full_offsets = np.zeros((133, 2), dtype=np.float32)
    full_offsets[[pair[0] for pair in MADS_TO_COCO_BODY]] = body_offsets
    metadata = {
        "schema_version": 1,
        "adapter_type": "vitpose_mads_robust_heatmap_offsets",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "training_sequences": [item["metadata"]["sequence"] for item in train_caches],
        "validation_sequences": [item["metadata"]["sequence"] for item in validation_caches],
        "held_out_test_sequences": sorted(set(args.held_out_test_sequences)),
        "supervised_coco_joint_indices": [pair[0] for pair in MADS_TO_COCO_BODY],
        "max_offset_heatmap_px": args.max_offset_heatmap_px,
        "body_offsets_heatmap_xy": body_offsets.tolist(),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "production_approved": False,
        "approval_requirement": "Held-out 3D MADS benchmark must improve while retaining >=95% valid joints.",
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "heatmap_offsets_xy": torch.from_numpy(full_offsets),
            "metadata": metadata,
        },
        args.output_checkpoint,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_session_json(
        {**metadata, "output_checkpoint": str(args.output_checkpoint.resolve())},
        args.report,
    )
    print(f"offset adapter: {args.output_checkpoint.resolve()}")
    print(f"validation mean error: {validation_metrics['base_mean_px']:.4f} -> "
          f"{validation_metrics['calibrated_mean_px']:.4f}")
    print(f"validation improvement: {improvement:.2%}")
    print(f"report: {args.report.resolve()}")


def load_split_caches(
    cache_dir: Path,
    *,
    validation_sequences: list[str],
    held_out_test_sequences: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    validation_set = {value.lower() for value in validation_sequences}
    test_set = {value.lower() for value in held_out_test_sequences}
    if validation_set & test_set:
        raise ValueError("Validation and held-out test sequences must be disjoint")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    discovered: set[str] = set()
    for path in sorted(cache_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        label = str(payload["metadata"]["sequence"])
        normalized = label.lower()
        discovered.add(normalized)
        if normalized in test_set:
            raise ValueError(f"Held-out test sequence leaked into feature cache: {label}")
        (validation if normalized in validation_set else train).append(payload)
    missing_validation = sorted(validation_set - discovered)
    if missing_validation:
        raise ValueError(f"Validation caches missing: {missing_validation}")
    if not train or not validation:
        raise ValueError("Training and validation caches must both be non-empty")
    return train, validation


def decode_cache_coordinates(
    runtime: ViTPosePlusWholeBodyInferencer,
    caches: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    coco_indices = torch.as_tensor(
        [pair[0] for pair in MADS_TO_COCO_BODY], dtype=torch.long, device=runtime.device
    )
    predicted_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    runtime._model.head.eval()
    for payload in caches:
        features = payload["features"]
        targets = payload["targets"]
        for start in range(0, features.shape[0], batch_size):
            with torch.no_grad():
                heatmaps = (
                    runtime._model.head(features[start : start + batch_size].to(runtime.device).float())
                    .index_select(1, coco_indices)
                    .cpu()
                    .numpy()
                )
            target_batch = targets[start : start + batch_size].float().numpy()
            for predicted_heatmaps, target_heatmaps in zip(heatmaps, target_batch, strict=True):
                flat = predicted_heatmaps.reshape(predicted_heatmaps.shape[0], -1)
                indices = np.argmax(flat, axis=1)
                peaks = np.column_stack(
                    [indices % predicted_heatmaps.shape[2], indices // predicted_heatmaps.shape[2]]
                ).astype(float)
                predicted_rows.append(
                    _refine_heatmap_peaks_udp(predicted_heatmaps, peaks, kernel_size=11)
                )
                truth_rows.append(_heatmap_centers(target_heatmaps))
    return np.stack(predicted_rows), np.stack(truth_rows)


def robust_joint_offsets(
    predicted_xy: np.ndarray,
    truth_xy: np.ndarray,
    *,
    max_offset_heatmap_px: float,
) -> np.ndarray:
    predicted = np.asarray(predicted_xy, dtype=float)
    truth = np.asarray(truth_xy, dtype=float)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[-1] != 2:
        raise ValueError("predicted_xy and truth_xy must share shape [samples, joints, 2]")
    offsets = np.median(truth - predicted, axis=0)
    norms = np.linalg.norm(offsets, axis=1)
    scale = np.minimum(1.0, max_offset_heatmap_px / np.maximum(norms, 1e-12))
    return offsets * scale[:, None]


def coordinate_metrics(
    predicted_xy: np.ndarray,
    truth_xy: np.ndarray,
    offsets_xy: np.ndarray,
) -> dict[str, float]:
    base = np.linalg.norm(predicted_xy - truth_xy, axis=-1)
    calibrated = np.linalg.norm(predicted_xy + offsets_xy[None, ...] - truth_xy, axis=-1)
    base_mean = float(np.mean(base))
    calibrated_mean = float(np.mean(calibrated))
    return {
        "base_mean_px": base_mean,
        "base_median_px": float(np.median(base)),
        "calibrated_mean_px": calibrated_mean,
        "calibrated_median_px": float(np.median(calibrated)),
        "improvement_ratio": (base_mean - calibrated_mean) / base_mean if base_mean > 0.0 else 0.0,
    }


def _heatmap_centers(heatmaps: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmaps, dtype=float)
    height, width = values.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    denominator = np.sum(values, axis=(1, 2))
    if np.any(denominator <= 0.0):
        raise ValueError("Target heatmaps must contain positive mass")
    return np.column_stack(
        [
            np.sum(values * xx, axis=(1, 2)) / denominator,
            np.sum(values * yy, axis=(1, 2)) / denominator,
        ]
    )


if __name__ == "__main__":
    main()
