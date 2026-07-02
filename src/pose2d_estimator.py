from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, PersonPose2D, empty_pose_2d
from .model_runtime import ModelRuntimeError


@dataclass(slots=True)
class Pose2DConfig:
    model_name: str
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    score_threshold: float = 0.30


class RTMW2DEstimator:
    """RTMW-x-l adapter. The MMPose initialization is intentionally isolated here."""

    def __init__(self, config: Pose2DConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._model: Any | None = None
        if not dry_run:
            self._model = self._build_model()

    def predict(self, frame: np.ndarray, camera_id: str, frame_idx: int) -> PersonPose2D:
        if self.dry_run:
            return empty_pose_2d(camera_id, frame_idx)
        if self._model is None:
            raise RuntimeError("RTMW2DEstimator model is not initialized")

        result = self._model(frame)
        keypoints_xy, scores = _extract_mmpose_wholebody(result)
        return pose2d_from_arrays(
            camera_id=camera_id,
            frame_idx=frame_idx,
            keypoints_xy=keypoints_xy,
            scores=scores,
            score_threshold=self.config.score_threshold,
        )

    def _build_model(self) -> Any:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"RTMW-x-l config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"RTMW-x-l checkpoint not found: {self.config.checkpoint_path}")
        try:
            from mmpose.apis import MMPoseInferencer
        except ModuleNotFoundError as exc:
            raise ModelRuntimeError(
                "MMPose is not installed. Install the RTMW-compatible MMPose environment before live inference."
            ) from exc
        return MMPoseInferencer(
            pose2d=str(self.config.config_path),
            pose2d_weights=str(self.config.checkpoint_path),
            device=self.config.device,
        )


def pose2d_from_arrays(
    camera_id: str,
    frame_idx: int,
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> PersonPose2D:
    if keypoints_xy.shape != (COCO_WHOLEBODY_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shape {(COCO_WHOLEBODY_KEYPOINTS, 2)}, got {keypoints_xy.shape}")
    valid_mask = np.asarray(scores) >= score_threshold
    return PersonPose2D(
        camera_id=camera_id,
        frame_idx=frame_idx,
        keypoints_xy=keypoints_xy,
        scores=scores,
        valid_mask=valid_mask,
    )


def _extract_mmpose_wholebody(result: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, dict):
        result = next(result)
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    if not predictions:
        return (
            np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float),
            np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float),
        )
    best = max(predictions, key=lambda item: float(np.nanmean(item.get("keypoint_scores", [0.0]))))
    keypoints = np.asarray(best["keypoints"], dtype=float)
    scores = np.asarray(best.get("keypoint_scores", np.ones(keypoints.shape[0])), dtype=float)
    if keypoints.shape[0] < COCO_WHOLEBODY_KEYPOINTS:
        padded_xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        padded_scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        padded_xy[: keypoints.shape[0]] = keypoints[:, :2]
        padded_scores[: scores.shape[0]] = scores
        return padded_xy, padded_scores
    return keypoints[:COCO_WHOLEBODY_KEYPOINTS, :2], scores[:COCO_WHOLEBODY_KEYPOINTS]
