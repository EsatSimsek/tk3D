from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, PersonPose2D, PersonPose3D
from .model_runtime import ModelRuntimeError


@dataclass(slots=True)
class Pose3DConfig:
    model_name: str
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    enabled: bool = True


class RTMW3DEstimator:
    """RTMW3D-x adapter for auxiliary single-view 3D estimates."""

    def __init__(self, config: Pose3DConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._model: Any | None = None
        if config.enabled and not dry_run:
            self._model = self._build_model()

    def predict(self, pose2d: PersonPose2D, frame: np.ndarray | None = None) -> PersonPose3D:
        if self.dry_run or not self.config.enabled:
            keypoints = np.full((COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
            scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
            return PersonPose3D(
                camera_id=pose2d.camera_id,
                frame_idx=pose2d.frame_idx,
                keypoints_camera=keypoints,
                scores=scores,
            )
        if self._model is None:
            raise RuntimeError("RTMW3DEstimator model is not initialized")
        if frame is None:
            raise ValueError("RTMW3DEstimator live inference requires the source frame.")
        result = self._model(frame)
        keypoints, scores = _extract_mmpose_3d(result)
        return PersonPose3D(
            camera_id=pose2d.camera_id,
            frame_idx=pose2d.frame_idx,
            keypoints_camera=keypoints,
            scores=scores,
        )

    def _build_model(self) -> Any:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"RTMW3D-x config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"RTMW3D-x checkpoint not found: {self.config.checkpoint_path}")
        try:
            from mmpose.apis import MMPoseInferencer
        except ModuleNotFoundError as exc:
            raise ModelRuntimeError(
                "MMPose is not installed. Install the RTMW3D-compatible MMPose environment before live inference."
            ) from exc
        return MMPoseInferencer(
            pose3d=str(self.config.config_path),
            pose3d_weights=str(self.config.checkpoint_path),
            device=self.config.device,
        )


def _extract_mmpose_3d(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, dict):
        result = next(result)
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    if not predictions:
        return (
            np.full((COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float),
            np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float),
        )
    best = max(predictions, key=_prediction_score)
    raw_keypoints = best.get("keypoints_3d", best.get("keypoints", []))
    keypoints = np.asarray(raw_keypoints, dtype=float)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        raise ModelRuntimeError(f"RTMW3D output has unsupported keypoint shape: {keypoints.shape}")
    raw_scores = best.get("keypoint_scores", best.get("keypoint_scores_3d", np.ones(keypoints.shape[0])))
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim > 1:
        scores = scores.reshape(-1)
    output_keypoints = np.full((COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
    output_scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
    count = min(COCO_WHOLEBODY_KEYPOINTS, keypoints.shape[0])
    output_keypoints[:count] = keypoints[:count, :3]
    output_scores[: min(count, scores.shape[0])] = scores[: min(count, scores.shape[0])]
    return output_keypoints, output_scores


def _prediction_score(item: dict[str, Any]) -> float:
    scores = np.asarray(item.get("keypoint_scores", item.get("keypoint_scores_3d", [0.0])), dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))
