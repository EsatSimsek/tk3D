from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exporter import export_session_json
from src.mads_dataset import (
    MadsSequence,
    discover_mads_sequences,
    load_mads_camera_calibration,
    load_mads_ground_truth,
    resolve_mads_roots,
)
from src.vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer


MADS_TO_COCO_BODY: tuple[tuple[int, int], ...] = (
    (5, 8),
    (6, 11),
    (7, 9),
    (8, 12),
    (9, 10),
    (10, 13),
    (11, 2),
    (12, 5),
    (13, 3),
    (14, 6),
    (15, 4),
    (16, 7),
)


@dataclass(slots=True)
class CachedFeatures:
    features: Any
    targets: Any
    weights: Any
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-controlled ViTPose body heatmap adapter from MADS mocap projections."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--train-actions",
        nargs="+",
        default=["Kata", "Taichi"],
        help="MADS multiview actions used for adapter training.",
    )
    parser.add_argument(
        "--test-sequences",
        nargs="+",
        default=["Kata:F2"],
        help="Held-out sequences that must never enter train/validation caches.",
    )
    parser.add_argument(
        "--validation-sequences",
        nargs="+",
        default=["Kata:F3", "Taichi:S6"],
    )
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-scope", choices=["final_layer", "head"], default="final_layer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "weights" / "vitpose_huge_wholebody_256x192.pth",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "mads_adapter" / "feature_cache",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=ROOT / "weights" / "vitpose_huge_wholebody_mads_head.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "mads_adapter" / "training_report.json",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise SystemExit("frame-stride, epochs, batch-size, and patience must be positive")
    if args.learning_rate <= 0.0:
        raise SystemExit("learning-rate must be positive")
    if not args.base_checkpoint.exists():
        raise SystemExit(f"Base ViTPose checkpoint not found: {args.base_checkpoint}")

    roots = resolve_mads_roots(args.dataset_root)
    sequences = [item for item in discover_mads_sequences(roots) if item.modality == "multiview"]
    train_sequences, validation_sequences = split_sequences(
        sequences,
        actions=args.train_actions,
        test_labels=args.test_sequences,
        validation_labels=args.validation_sequences,
    )
    if not train_sequences or not validation_sequences:
        raise SystemExit("Both training and validation splits must contain at least one sequence")

    runtime = ViTPosePlusWholeBodyInferencer(
        checkpoint_path=args.base_checkpoint.resolve(),
        device=args.device,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    train_caches = [
        cache_sequence_features(
            runtime,
            sequence,
            args.cache_dir,
            args.frame_stride,
            rebuild=args.rebuild_cache,
        )
        for sequence in train_sequences
    ]
    validation_caches = [
        cache_sequence_features(
            runtime,
            sequence,
            args.cache_dir,
            args.frame_stride,
            rebuild=args.rebuild_cache,
        )
        for sequence in validation_sequences
    ]
    training = train_adapter_head(
        runtime,
        train_caches,
        validation_caches,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        train_scope=args.train_scope,
    )

    metadata = {
        "schema_version": 1,
        "adapter_type": "vitpose_mads_projected_mocap_heatmap_head",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "dataset_root": str(roots.dataset_root),
        "frame_stride": args.frame_stride,
        "train_scope": args.train_scope,
        "train_sequences": [sequence_label(item) for item in train_sequences],
        "validation_sequences": [sequence_label(item) for item in validation_sequences],
        "held_out_test_sequences": sorted(set(args.test_sequences)),
        "supervised_coco_joint_indices": [pair[0] for pair in MADS_TO_COCO_BODY],
        "production_approved": False,
        "approval_requirement": (
            "Held-out 3D benchmark must improve while retaining at least 95% valid joints."
        ),
        **training["metrics"],
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime._torch.save(
        {
            "head_state_dict": training["head_state_dict"],
            "metadata": metadata,
        },
        args.output_checkpoint,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_session_json(
        {
            **metadata,
            "output_checkpoint": str(args.output_checkpoint.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "epoch_history": training["history"],
        },
        args.report,
    )
    print(f"adapter: {args.output_checkpoint.resolve()}")
    print(f"report: {args.report.resolve()}")
    print(f"baseline validation loss: {metadata['baseline_validation_loss']:.8f}")
    print(f"best validation loss: {metadata['best_validation_loss']:.8f}")


def split_sequences(
    sequences: list[MadsSequence],
    *,
    actions: list[str],
    test_labels: list[str],
    validation_labels: list[str],
) -> tuple[list[MadsSequence], list[MadsSequence]]:
    action_set = {value.lower() for value in actions}
    test_set = {value.lower() for value in test_labels}
    validation_set = {value.lower() for value in validation_labels}
    if test_set & validation_set:
        raise ValueError("Test and validation sequence labels must be disjoint")
    eligible = [item for item in sequences if item.action.lower() in action_set]
    available = {sequence_label(item).lower() for item in eligible}
    missing = sorted((test_set | validation_set) - available)
    if missing:
        raise ValueError(f"Requested MADS sequence labels not found: {missing}")
    training = [
        item
        for item in eligible
        if sequence_label(item).lower() not in test_set | validation_set
    ]
    validation = [item for item in eligible if sequence_label(item).lower() in validation_set]
    return training, validation


def sequence_label(sequence: MadsSequence) -> str:
    return f"{sequence.action}:{sequence.sequence}"


def cache_sequence_features(
    runtime: ViTPosePlusWholeBodyInferencer,
    sequence: MadsSequence,
    cache_dir: Path,
    frame_stride: int,
    *,
    rebuild: bool,
) -> CachedFeatures:
    import torch

    cache_path = cache_dir / f"{sequence.action}_{sequence.sequence}_stride{frame_stride}.pt"
    if cache_path.exists() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return CachedFeatures(
            payload["features"], payload["targets"], payload["weights"], payload["metadata"]
        )

    gt = load_mads_ground_truth(sequence.ground_truth_path, expected_joint_count=15)
    action_root = sequence.ground_truth_path.parent
    features: list[Any] = []
    targets: list[Any] = []
    weights: list[Any] = []
    sample_rows: list[dict[str, Any]] = []
    runtime._model.eval()
    for camera_number, video_path in sorted(sequence.videos.items()):
        camera_id = f"C{camera_number}"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open MADS video: {video_path}")
        image_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        calibration = load_mads_camera_calibration(
            action_root / f"Calib_Cam{camera_number}.mat",
            camera_id,
            image_size,
        )
        frame_idx = 0
        kept = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx >= gt.shape[0]:
                break
            if frame_idx % frame_stride == 0 and np.all(np.isfinite(gt[frame_idx])):
                projected, _ = cv2.projectPoints(
                    gt[frame_idx],
                    calibration.rotation_vector,
                    calibration.translation_vector,
                    calibration.intrinsic_matrix,
                    calibration.distortion_coefficients,
                )
                projected = projected.reshape(-1, 2)
                bbox = _bbox_from_projected_pose(projected)
                tensor, crop = runtime._preprocess(frame, bbox_xyxy=bbox)
                with torch.no_grad():
                    source = torch.full(
                        (tensor.shape[0],), 5, dtype=torch.long, device=tensor.device
                    )
                    feature = runtime._model.backbone(tensor, source)
                target, target_weights = heatmap_targets(
                    projected,
                    crop,
                    heatmap_size=(48, 64),
                    sigma=2.0,
                )
                if np.count_nonzero(target_weights) >= 10:
                    features.append(feature[0].detach().cpu().to(torch.float16))
                    targets.append(torch.from_numpy(target).to(torch.float16))
                    weights.append(torch.from_numpy(target_weights).to(torch.float16))
                    sample_rows.append(
                        {
                            "camera_id": camera_id,
                            "frame_idx": frame_idx,
                            "valid_target_count": int(np.count_nonzero(target_weights)),
                        }
                    )
                    kept += 1
            frame_idx += 1
        capture.release()
        print(f"cached {sequence_label(sequence)} {camera_id}: {kept} samples", flush=True)

    if not features:
        raise ValueError(f"No trainable samples generated for {sequence_label(sequence)}")
    payload = {
        "features": torch.stack(features),
        "targets": torch.stack(targets),
        "weights": torch.stack(weights),
        "metadata": {
            "sequence": sequence_label(sequence),
            "frame_stride": frame_stride,
            "sample_count": len(features),
            "samples": sample_rows,
        },
    }
    torch.save(payload, cache_path)
    return CachedFeatures(
        payload["features"], payload["targets"], payload["weights"], payload["metadata"]
    )


def heatmap_targets(
    projected_xy: np.ndarray,
    crop: tuple[float, float, float, float],
    *,
    heatmap_size: tuple[int, int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = heatmap_size
    targets = np.zeros((len(MADS_TO_COCO_BODY), height, width), dtype=np.float32)
    weights = np.zeros(len(MADS_TO_COCO_BODY), dtype=np.float32)
    x1, y1, x2, y2 = crop
    if x2 <= x1 or y2 <= y1 or sigma <= 0.0:
        raise ValueError("Invalid crop or heatmap sigma")
    yy, xx = np.mgrid[:height, :width]
    points = np.asarray(projected_xy, dtype=float)
    for target_idx, (_, mads_idx) in enumerate(MADS_TO_COCO_BODY):
        point = points[mads_idx]
        heatmap_x = (point[0] - x1) * (width - 1.0) / (x2 - x1)
        heatmap_y = (point[1] - y1) * (height - 1.0) / (y2 - y1)
        if not np.isfinite(heatmap_x) or not np.isfinite(heatmap_y):
            continue
        if not (-3.0 * sigma <= heatmap_x <= width - 1.0 + 3.0 * sigma):
            continue
        if not (-3.0 * sigma <= heatmap_y <= height - 1.0 + 3.0 * sigma):
            continue
        targets[target_idx] = np.exp(
            -((xx - heatmap_x) ** 2 + (yy - heatmap_y) ** 2) / (2.0 * sigma**2)
        )
        weights[target_idx] = 1.0
    return targets, weights


def train_adapter_head(
    runtime: ViTPosePlusWholeBodyInferencer,
    train_caches: list[CachedFeatures],
    validation_caches: list[CachedFeatures],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    train_scope: str,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_data = _combine_caches(train_caches)
    validation_data = _combine_caches(validation_caches)
    train_loader = DataLoader(
        TensorDataset(train_data.features, train_data.targets, train_data.weights),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_data.features, validation_data.targets, validation_data.weights),
        batch_size=batch_size,
        shuffle=False,
    )
    head = runtime._model.head
    for parameter in head.parameters():
        parameter.requires_grad = train_scope == "head"
    if train_scope == "final_layer":
        for parameter in head.final_layer.parameters():
            parameter.requires_grad = True
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    coco_indices = torch.as_tensor(
        [pair[0] for pair in MADS_TO_COCO_BODY], dtype=torch.long, device=runtime.device
    )
    baseline_loss = _evaluate_head(head, validation_loader, coco_indices, runtime.device)
    best_loss = baseline_loss
    best_state = copy.deepcopy(head.state_dict())
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        if train_scope == "head":
            head.train()
        else:
            # Keep frozen BatchNorm running statistics identical to the base
            # model while adapting only the final supervised output rows.
            head.eval()
            head.final_layer.train()
        total_loss = 0.0
        total_batches = 0
        for features, targets, weights in train_loader:
            features = features.to(runtime.device, dtype=torch.float32)
            targets = targets.to(runtime.device, dtype=torch.float32)
            weights = weights.to(runtime.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predicted = head(features).index_select(1, coco_indices)
            loss = _weighted_heatmap_loss(predicted, targets, weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        validation_loss = _evaluate_head(head, validation_loader, coco_indices, runtime.device)
        train_loss = total_loss / max(total_batches, 1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        print(
            f"epoch {epoch:02d}: train={train_loss:.8f} validation={validation_loss:.8f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    head.load_state_dict(best_state, strict=True)
    return {
        "head_state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "history": history,
        "metrics": {
            "train_sample_count": int(train_data.features.shape[0]),
            "validation_sample_count": int(validation_data.features.shape[0]),
            "baseline_validation_loss": baseline_loss,
            "best_validation_loss": best_loss,
            "completed_epoch_count": len(history),
        },
    }


def _combine_caches(caches: list[CachedFeatures]) -> CachedFeatures:
    import torch

    if not caches:
        raise ValueError("At least one feature cache is required")
    return CachedFeatures(
        torch.cat([item.features for item in caches]),
        torch.cat([item.targets for item in caches]),
        torch.cat([item.weights for item in caches]),
        {"sequences": [item.metadata["sequence"] for item in caches]},
    )


def _evaluate_head(head: Any, loader: Any, coco_indices: Any, device: str) -> float:
    import torch

    head.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, targets, weights in loader:
            predicted = head(features.to(device, dtype=torch.float32)).index_select(1, coco_indices)
            loss = _weighted_heatmap_loss(
                predicted,
                targets.to(device, dtype=torch.float32),
                weights.to(device, dtype=torch.float32),
            )
            total += float(loss.cpu())
            count += 1
    return total / max(count, 1)


def _weighted_heatmap_loss(predicted: Any, targets: Any, weights: Any) -> Any:
    weighted = (predicted - targets) ** 2 * weights[..., None, None]
    denominator = weights.sum().clamp_min(1.0) * predicted.shape[-1] * predicted.shape[-2]
    return weighted.sum() / denominator


def _bbox_from_projected_pose(projected_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(projected_xy, dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(finite) < 10:
        raise ValueError("Projected MADS pose has too few finite joints")
    mins = np.min(points[finite], axis=0)
    maxs = np.max(points[finite], axis=0)
    # Match the live pose tracker's motion margin before `_preprocess` adds
    # aspect-ratio padding. Training and inference crops must use one scale.
    span = np.maximum(maxs - mins, 24.0) * 1.35
    center = (mins + maxs) / 2.0
    return np.asarray(
        [
            center[0] - span[0] / 2.0,
            center[1] - span[1] / 2.0,
            center[0] + span[0] / 2.0,
            center[1] + span[1] / 2.0,
        ],
        dtype=float,
    )


if __name__ == "__main__":
    main()
