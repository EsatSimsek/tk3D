from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, PersonPose2D, empty_pose_2d
from .mmpose_compat import install_mmpose_runtime_compat
from .model_runtime import ModelRuntimeError

@dataclass(slots=True)
class Pose2DConfig:
    model_name: str
    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    score_threshold: float = 0.30
    input_size: tuple[int, int] = (256, 192)

class RTMW2DEstimator:
    """RTMW adapter. The MMPose initialization is intentionally isolated here."""

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

    def predict_batch(self, frames: list[np.ndarray], camera_id: str, frame_indices: list[int]) -> list[PersonPose2D]:
        if len(frames) != len(frame_indices):
            raise ValueError("frames and frame_indices must have the same length")
        if self.dry_run:
            return [empty_pose_2d(camera_id, frame_idx) for frame_idx in frame_indices]
        if self._model is None:
            raise RuntimeError("RTMW2DEstimator model is not initialized")
        try:
            results = self._model(frames)
            if not isinstance(results, list):
                results = list(results)
        except Exception as exc:
            warnings.warn(f"RTMW batch inference failed; falling back to single-frame inference: {exc}", RuntimeWarning)
            return [self.predict(frame, camera_id, frame_idx) for frame, frame_idx in zip(frames, frame_indices)]
        if len(results) != len(frame_indices):
            raise ModelRuntimeError(f"RTMW returned {len(results)} results for {len(frame_indices)} frames")
        poses = []
        for result, frame_idx in zip(results, frame_indices, strict=True):
            keypoints_xy, scores = _extract_mmpose_wholebody(result)
            poses.append(
                pose2d_from_arrays(
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    keypoints_xy=keypoints_xy,
                    scores=scores,
                    score_threshold=self.config.score_threshold,
                )
            )
        return poses

    def predict_many(self, frames: list[np.ndarray], camera_ids: list[str], frame_indices: list[int]) -> list[PersonPose2D]:
        if len(frames) != len(camera_ids) or len(frames) != len(frame_indices):
            raise ValueError("frames, camera_ids, and frame_indices must have the same length")
        if self.dry_run:
            return [empty_pose_2d(camera_id, frame_idx) for camera_id, frame_idx in zip(camera_ids, frame_indices)]
        if self._model is None:
            raise RuntimeError("RTMW2DEstimator model is not initialized")
        try:
            results = self._model(frames)
            if not isinstance(results, list):
                results = list(results)
        except Exception as exc:
            warnings.warn(f"RTMW multi-camera batch failed; falling back to single-frame inference: {exc}", RuntimeWarning)
            return [self.predict(frame, camera_id, frame_idx) for frame, camera_id, frame_idx in zip(frames, camera_ids, frame_indices)]
        if len(results) != len(frame_indices):
            raise ModelRuntimeError(f"RTMW returned {len(results)} results for {len(frame_indices)} frames")
        poses = []
        for result, camera_id, frame_idx in zip(results, camera_ids, frame_indices, strict=True):
            keypoints_xy, scores = _extract_mmpose_wholebody(result)
            poses.append(
                pose2d_from_arrays(
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    keypoints_xy=keypoints_xy,
                    scores=scores,
                    score_threshold=self.config.score_threshold,
                )
            )
        return poses

    def _build_model(self) -> Any:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"RTMW config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"RTMW checkpoint not found: {self.config.checkpoint_path}")
        install_mmpose_runtime_compat()
        try:
            from mmpose.apis.inferencers import Pose2DInferencer
        except ModuleNotFoundError as exc:
            raise ModelRuntimeError(
                "MMPose is not installed. Install the RTMW-compatible MMPose environment before live inference."
            ) from exc
        return Pose2DInferencer(
            model=str(self.config.config_path),
            weights=str(self.config.checkpoint_path),
            device=self.config.device,
            det_model="whole_image",
        )

class ViTPose2DEstimator:
    """ViTPose-Huge whole-body adapter. The MMPose initialization is intentionally isolated here."""

    def __init__(self, config: Pose2DConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._model: Any | None = None
        self._person_bbox_by_camera: dict[str, np.ndarray] = {}
        self._missed_bbox_updates_by_camera: dict[str, int] = {}
        if not dry_run:
            self._model = self._build_model()

    def predict(self, frame: np.ndarray, camera_id: str, frame_idx: int) -> PersonPose2D:
        if self.dry_run:
            return empty_pose_2d(camera_id, frame_idx)
        if self._model is None:
            raise RuntimeError("ViTPose2DEstimator model is not initialized")

        bbox = self._person_bbox_by_camera.get(camera_id)
        if hasattr(self._model, "predict_arrays"):
            keypoints_xy, scores = self._model.predict_arrays(frame, bbox_xyxy=bbox)
        else:
            result = self._model(frame)
            keypoints_xy, scores = _extract_mmpose_wholebody(result, allow_padding=False)
        pose = pose2d_from_arrays(
            camera_id=camera_id,
            frame_idx=frame_idx,
            keypoints_xy=keypoints_xy,
            scores=scores,
            score_threshold=self.config.score_threshold,
        )
        tracked_bbox = _bbox_from_pose(pose, frame.shape[1], frame.shape[0])
        if tracked_bbox is not None:
            previous = self._person_bbox_by_camera.get(camera_id)
            self._person_bbox_by_camera[camera_id] = tracked_bbox if previous is None else 0.75 * previous + 0.25 * tracked_bbox
            self._missed_bbox_updates_by_camera[camera_id] = 0
        else:
            missed = self._missed_bbox_updates_by_camera.get(camera_id, 0) + 1
            self._missed_bbox_updates_by_camera[camera_id] = missed
            if missed >= 3:
                self._person_bbox_by_camera.pop(camera_id, None)
        return pose

    def predict_batch(self, frames: list[np.ndarray], camera_id: str, frame_indices: list[int]) -> list[PersonPose2D]:
        if len(frames) != len(frame_indices):
            raise ValueError("frames and frame_indices must have the same length")
        if self.dry_run:
            return [empty_pose_2d(camera_id, frame_idx) for frame_idx in frame_indices]
        if self._model is None:
            raise RuntimeError("ViTPose2DEstimator model is not initialized")
        return [
            self.predict(frame, camera_id, frame_idx)
            for frame, frame_idx in zip(frames, frame_indices, strict=True)
        ]

    def predict_many(self, frames: list[np.ndarray], camera_ids: list[str], frame_indices: list[int]) -> list[PersonPose2D]:
        if len(frames) != len(camera_ids) or len(frames) != len(frame_indices):
            raise ValueError("frames, camera_ids, and frame_indices must have the same length")
        return [
            self.predict(frame, camera_id, frame_idx)
            for frame, camera_id, frame_idx in zip(frames, camera_ids, frame_indices, strict=True)
        ]

    def _build_model(self) -> Any:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"ViTPose config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {self.config.checkpoint_path}")
        install_mmpose_runtime_compat()
        from .vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer

        return ViTPosePlusWholeBodyInferencer(
            checkpoint_path=self.config.checkpoint_path,
            device=self.config.device,
            input_height=int(self.config.input_size[0]),
            input_width=int(self.config.input_size[1]),
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

def _prediction_score(item: dict[str, Any]) -> float:
    scores = np.asarray(item.get("keypoint_scores", [0.0]), dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))

def _extract_mmpose_wholebody(result: Any, allow_padding: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
    best = max(predictions, key=_prediction_score)
    keypoints = np.asarray(best["keypoints"], dtype=float)
    scores = np.asarray(best.get("keypoint_scores", np.ones(keypoints.shape[0])), dtype=float)
    if keypoints.shape[0] < COCO_WHOLEBODY_KEYPOINTS:
        if not allow_padding:
            raise ModelRuntimeError(
                f"Expected ViTPose whole-body output with {COCO_WHOLEBODY_KEYPOINTS} keypoints, "
                f"got {keypoints.shape[0]}. Check that the configured model is a whole-body "
                "ViTPose-Huge checkpoint, not a COCO body-only checkpoint."
            )
        padded_xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        padded_scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        padded_xy[: keypoints.shape[0]] = keypoints[:, :2]
        padded_scores[: scores.shape[0]] = scores
        return padded_xy, padded_scores
    return keypoints[:COCO_WHOLEBODY_KEYPOINTS, :2], scores[:COCO_WHOLEBODY_KEYPOINTS]


def _bbox_from_pose(pose: PersonPose2D, image_width: int, image_height: int) -> np.ndarray | None:
    body_count = min(17, pose.keypoints_xy.shape[0])
    valid = pose.valid_mask[:body_count] & np.all(np.isfinite(pose.keypoints_xy[:body_count]), axis=1)
    if np.count_nonzero(valid) < 5:
        return None
    points = pose.keypoints_xy[:body_count][valid]
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 32.0)
    center = (mins + maxs) / 2.0
    expanded = span * 1.35
    bbox = np.asarray(
        [center[0] - expanded[0] / 2.0, center[1] - expanded[1] / 2.0,
         center[0] + expanded[0] / 2.0, center[1] + expanded[1] / 2.0],
        dtype=float,
    )
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, max(float(image_width - 1), 0.0))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, max(float(image_height - 1), 0.0))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None
