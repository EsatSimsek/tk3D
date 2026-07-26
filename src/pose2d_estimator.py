from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, PersonPose2D, empty_pose_2d
from .mmpose_compat import install_mmpose_runtime_compat
from .model_runtime import ModelRuntimeError
from .person_tracking import PersonDetectorConfig, RFDETRPersonTracker
from .pose_temporal import TemporalPose2DConfig, TemporalPose2DFilter


@dataclass(slots=True)
class Pose2DConfig:
    model_name: str
    config_path: Path
    checkpoint_path: Path
    adapter_checkpoint_path: Path | None = None
    allow_unapproved_adapter: bool = False
    device: str = "cuda:0"
    score_threshold: float = 0.30
    input_size: tuple[int, int] = (256, 192)
    flip_test: bool = True
    temporal_filter_enabled: bool = True
    temporal_stabilize_left_right: bool = True
    person_detector: PersonDetectorConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device cannot be empty")
        if not np.isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and between 0 and 1")
        if not isinstance(self.input_size, (list, tuple)) or len(self.input_size) != 2:
            raise ValueError("input_size must contain height and width")
        normalized_size = tuple(int(value) for value in self.input_size)
        if any(value <= 0 or value % 16 for value in normalized_size):
            raise ValueError("input_size values must be positive multiples of 16")
        self.input_size = normalized_size


@dataclass(slots=True)
class GuidedPose2DResult:
    guided_pose: PersonPose2D
    unconstrained_pose: PersonPose2D


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
        self._bbox_frame_idx_by_camera: dict[str, int] = {}
        self._missed_bbox_updates_by_camera: dict[str, int] = {}
        self._previous_frame_by_camera: dict[str, np.ndarray] = {}
        self._person_track_id_by_camera: dict[str, int] = {}
        self._person_tracker: RFDETRPersonTracker | None = None
        self._temporal_filter = TemporalPose2DFilter(
            TemporalPose2DConfig(
                enabled=self.config.temporal_filter_enabled,
                stabilize_left_right=self.config.temporal_stabilize_left_right,
            )
        )
        if not dry_run:
            self._model = self._build_model()
            if self.config.person_detector is not None and self.config.person_detector.enabled:
                self._person_tracker = RFDETRPersonTracker(self.config.person_detector, device=self.config.device)

    def predict(self, frame: np.ndarray, camera_id: str, frame_idx: int) -> PersonPose2D:
        if self.dry_run:
            return empty_pose_2d(camera_id, frame_idx)
        if self._model is None:
            raise RuntimeError("ViTPose2DEstimator model is not initialized")

        bbox = self._person_bbox_by_camera.get(camera_id)
        person_id = 0
        if self._person_tracker is not None:
            tracked_person = self._person_tracker.track(frame, camera_id, frame_idx)
            if tracked_person is not None:
                previous_track_id = self._person_track_id_by_camera.get(camera_id)
                if previous_track_id != tracked_person.track_id:
                    self._temporal_filter.reset(camera_id)
                self._person_track_id_by_camera[camera_id] = tracked_person.track_id
                person_id = tracked_person.track_id
                bbox = tracked_person.bbox_xyxy.copy()
                self._person_bbox_by_camera[camera_id] = bbox.copy()
                self._bbox_frame_idx_by_camera[camera_id] = frame_idx
                self._missed_bbox_updates_by_camera[camera_id] = 0
            else:
                previous_track_id = self._person_track_id_by_camera.pop(camera_id, None)
                if previous_track_id is not None:
                    self._temporal_filter.reset(camera_id)
                self._person_bbox_by_camera.pop(camera_id, None)
                self._bbox_frame_idx_by_camera.pop(camera_id, None)
                bbox = None
        else:
            previous_frame = self._previous_frame_by_camera.get(camera_id)
            motion_bbox = _motion_person_bbox(previous_frame, frame)
            self._previous_frame_by_camera[camera_id] = frame.copy()
            if _motion_requires_reacquisition(bbox, motion_bbox):
                # A static human-shaped object can fool the initial foreground
                # heuristic.  Motion provides an independent identity signal and
                # lets the tracker switch to the active athlete before inference.
                bbox = np.asarray(motion_bbox, dtype=float).copy()
                self._person_bbox_by_camera[camera_id] = bbox.copy()
                self._bbox_frame_idx_by_camera[camera_id] = frame_idx
                self._missed_bbox_updates_by_camera[camera_id] = 0
                self._temporal_filter.reset(camera_id)
        if hasattr(self._model, "predict_arrays"):
            keypoints_xy, scores = self._model.predict_arrays(frame, bbox_xyxy=bbox)
        else:
            result = self._model(frame)
            keypoints_xy, scores = _extract_mmpose_wholebody(result, allow_padding=False)
        raw_pose = pose2d_from_arrays(
            camera_id=camera_id,
            frame_idx=frame_idx,
            keypoints_xy=keypoints_xy,
            scores=scores,
            score_threshold=self.config.score_threshold,
            person_id=person_id,
        )
        # Keep crop tracking tied to the current observation.  Feeding the
        # smoothed pose back into the crop creates a lag-amplifying loop.
        pose = self._temporal_filter.filter(raw_pose)
        if self._person_tracker is not None:
            return pose

        pose_bbox = _bbox_from_pose(raw_pose, frame.shape[1], frame.shape[0])
        visual_bbox = _initial_visual_bbox(frame)
        tracked_bbox = _combine_bbox_candidates(
            pose_bbox,
            visual_bbox,
            self._person_bbox_by_camera.get(camera_id),
        )
        if tracked_bbox is not None:
            previous = self._person_bbox_by_camera.get(camera_id)
            previous_frame_idx = self._bbox_frame_idx_by_camera.get(camera_id)
            frame_delta = 1 if previous_frame_idx is None else max(frame_idx - previous_frame_idx, 1)
            self._person_bbox_by_camera[camera_id] = _update_tracked_bbox(
                previous,
                tracked_bbox,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                frame_delta=frame_delta,
            )
            self._bbox_frame_idx_by_camera[camera_id] = frame_idx
            self._missed_bbox_updates_by_camera[camera_id] = 0
        else:
            missed = self._missed_bbox_updates_by_camera.get(camera_id, 0) + 1
            self._missed_bbox_updates_by_camera[camera_id] = missed
            if missed >= 3:
                self._person_bbox_by_camera.pop(camera_id, None)
                self._bbox_frame_idx_by_camera.pop(camera_id, None)
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

    def predict_guided(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_idx: int,
        baseline_pose: PersonPose2D,
        priors_xy: np.ndarray,
        prior_valid: np.ndarray,
        search_radius_px: float,
    ) -> GuidedPose2DResult:
        """Run a fresh heatmap decode near independent multi-view priors."""
        if self.dry_run:
            empty = empty_pose_2d(camera_id, frame_idx)
            return GuidedPose2DResult(empty, empty)
        if self._model is None:
            raise RuntimeError("ViTPose2DEstimator model is not initialized")
        if not hasattr(self._model, "predict_arrays_guided"):
            raise ModelRuntimeError("Configured ViTPose runtime does not support guided heatmap decoding")
        prior_points = np.asarray(priors_xy, dtype=float)
        valid = np.asarray(prior_valid, dtype=bool)
        if prior_points.shape != (COCO_WHOLEBODY_KEYPOINTS, 2):
            raise ValueError(
                f"priors_xy must have shape {(COCO_WHOLEBODY_KEYPOINTS, 2)}, got {prior_points.shape}"
            )
        if valid.shape != (COCO_WHOLEBODY_KEYPOINTS,):
            raise ValueError(
                f"prior_valid must have shape {(COCO_WHOLEBODY_KEYPOINTS,)}, got {valid.shape}"
            )
        bbox = _bbox_from_pose_and_priors(
            baseline_pose,
            prior_points,
            valid,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
        )
        guided_xy, guided_scores, unconstrained_xy, unconstrained_scores = (
            self._model.predict_arrays_guided(
                frame,
                priors_xy=prior_points,
                prior_valid=valid,
                bbox_xyxy=bbox,
                search_radius_px=search_radius_px,
            )
        )
        guided_pose = pose2d_from_arrays(
            camera_id,
            frame_idx,
            guided_xy,
            guided_scores,
            self.config.score_threshold,
            person_id=baseline_pose.person_id,
        )
        unconstrained_pose = pose2d_from_arrays(
            camera_id,
            frame_idx,
            unconstrained_xy,
            unconstrained_scores,
            self.config.score_threshold,
            person_id=baseline_pose.person_id,
        )
        return GuidedPose2DResult(guided_pose, unconstrained_pose)

    def _build_model(self) -> Any:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"ViTPose config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {self.config.checkpoint_path}")
        install_mmpose_runtime_compat()
        from .vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer

        return ViTPosePlusWholeBodyInferencer(
            checkpoint_path=self.config.checkpoint_path,
            adapter_checkpoint_path=self.config.adapter_checkpoint_path,
            allow_unapproved_adapter=self.config.allow_unapproved_adapter,
            device=self.config.device,
            input_height=int(self.config.input_size[0]),
            input_width=int(self.config.input_size[1]),
            flip_test=self.config.flip_test,
        )

def pose2d_from_arrays(
    camera_id: str,
    frame_idx: int,
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    person_id: int = 0,
) -> PersonPose2D:
    points = np.asarray(keypoints_xy, dtype=float)
    confidence = np.asarray(scores, dtype=float)
    if points.shape != (COCO_WHOLEBODY_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shape {(COCO_WHOLEBODY_KEYPOINTS, 2)}, got {points.shape}")
    if confidence.shape != (COCO_WHOLEBODY_KEYPOINTS,):
        raise ValueError(f"Expected scores shape {(COCO_WHOLEBODY_KEYPOINTS,)}, got {confidence.shape}")
    if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be finite and between 0 and 1")
    finite_points = np.all(np.isfinite(points), axis=1)
    finite_scores = np.isfinite(confidence)
    confidence = np.where(finite_scores, np.clip(confidence, 0.0, 1.0), 0.0)
    points = np.where(finite_points[:, None], points, np.nan)
    valid_mask = finite_points & finite_scores & (confidence >= score_threshold)
    return PersonPose2D(
        camera_id=camera_id,
        frame_idx=frame_idx,
        keypoints_xy=points,
        scores=confidence,
        valid_mask=valid_mask,
        person_id=int(person_id),
    )


def _prediction_score(item: dict[str, Any]) -> float:
    scores = np.asarray(item.get("keypoint_scores", [0.0]), dtype=float).reshape(-1)
    scores = scores[: min(17, scores.size)]
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))

def _extract_mmpose_wholebody(result: Any, allow_padding: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(result, dict):
        try:
            result = next(result)
        except (StopIteration, TypeError) as exc:
            raise ModelRuntimeError("MMPose returned no readable inference result") from exc
    if not isinstance(result, dict):
        raise ModelRuntimeError(f"MMPose result must be a mapping, got {type(result).__name__}")
    predictions = result.get("predictions", [])
    if not isinstance(predictions, (list, tuple)):
        raise ModelRuntimeError("MMPose predictions must be a sequence of mappings")
    predictions = list(predictions)
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    if not predictions:
        return (
            np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float),
            np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float),
        )
    if not isinstance(predictions, list) or not all(isinstance(item, dict) for item in predictions):
        raise ModelRuntimeError("MMPose predictions must be a list of mappings")
    best = max(predictions, key=_prediction_score)
    if "keypoints" not in best:
        raise ModelRuntimeError("MMPose prediction is missing keypoints")
    keypoints = np.asarray(best["keypoints"], dtype=float)
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ModelRuntimeError(f"MMPose keypoints have unsupported shape: {keypoints.shape}")
    scores = np.asarray(best.get("keypoint_scores", np.ones(keypoints.shape[0])), dtype=float).reshape(-1)
    output_count = min(keypoints.shape[0], scores.shape[0])
    if output_count < COCO_WHOLEBODY_KEYPOINTS:
        if not allow_padding:
            raise ModelRuntimeError(
                f"Expected ViTPose whole-body output with {COCO_WHOLEBODY_KEYPOINTS} keypoints, "
                f"got {output_count}. Check that the configured model is a whole-body "
                "ViTPose-Huge checkpoint, not a COCO body-only checkpoint."
            )
        padded_xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        padded_scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        padded_xy[:output_count] = keypoints[:output_count, :2]
        padded_scores[:output_count] = scores[:output_count]
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
    # Preserve extra motion margin before the model's aspect-ratio padding;
    # fast martial-arts limbs otherwise leave the tracked crop.
    expanded = span * 1.35
    bbox = np.asarray(
        [center[0] - expanded[0] / 2.0, center[1] - expanded[1] / 2.0,
         center[0] + expanded[0] / 2.0, center[1] + expanded[1] / 2.0],
        dtype=float,
    )
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, max(float(image_width - 1), 0.0))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, max(float(image_height - 1), 0.0))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _bbox_from_pose_and_priors(
    pose: PersonPose2D,
    priors_xy: np.ndarray,
    prior_valid: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    body_count = min(17, pose.keypoints_xy.shape[0], priors_xy.shape[0])
    observed_valid = (
        pose.valid_mask[:body_count]
        & np.all(np.isfinite(pose.keypoints_xy[:body_count]), axis=1)
    )
    guided_valid = (
        np.asarray(prior_valid[:body_count], dtype=bool)
        & np.all(np.isfinite(priors_xy[:body_count]), axis=1)
    )
    point_groups = []
    if np.any(observed_valid):
        point_groups.append(pose.keypoints_xy[:body_count][observed_valid])
    if np.any(guided_valid):
        point_groups.append(np.asarray(priors_xy[:body_count], dtype=float)[guided_valid])
    if not point_groups:
        return _bbox_from_pose(pose, image_width, image_height)
    points = np.concatenate(point_groups, axis=0)
    if points.shape[0] < 5:
        return _bbox_from_pose(pose, image_width, image_height)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 32.0)
    center = (mins + maxs) / 2.0
    expanded = span * 1.35
    bbox = np.r_[
        center - expanded / 2.0,
        center + expanded / 2.0,
    ]
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, max(float(image_width - 1), 0.0))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, max(float(image_height - 1), 0.0))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _update_tracked_bbox(
    previous: np.ndarray | None,
    candidate: np.ndarray,
    image_width: int,
    image_height: int,
    frame_delta: int = 1,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float).reshape(4)
    if previous is None:
        return candidate.copy()
    previous = np.asarray(previous, dtype=float).reshape(4)
    previous_center = (previous[:2] + previous[2:]) / 2.0
    candidate_center = (candidate[:2] + candidate[2:]) / 2.0
    previous_size = np.maximum(previous[2:] - previous[:2], 32.0)
    candidate_size = np.maximum(candidate[2:] - candidate[:2], 32.0)

    dt = max(int(frame_delta), 1)
    center_delta = candidate_center - previous_center
    maximum_center_step = 0.30 * float(np.linalg.norm(previous_size)) * dt
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance > maximum_center_step > 0.0:
        center_delta *= maximum_center_step / center_distance
    center_alpha = 1.0 - (1.0 - 0.35) ** dt
    center = previous_center + center_alpha * center_delta

    ratio_steps = min(dt, 3)
    size_ratio = np.clip(candidate_size / previous_size, 0.75 ** ratio_steps, 1.35 ** ratio_steps)
    bounded_size = previous_size * size_ratio
    growing = bounded_size >= previous_size
    size_alpha = np.where(
        growing,
        1.0 - (1.0 - 0.25) ** dt,
        1.0 - (1.0 - 0.08) ** dt,
    )
    size = previous_size + size_alpha * (bounded_size - previous_size)
    updated = np.r_[center - size / 2.0, center + size / 2.0]
    updated[[0, 2]] = np.clip(updated[[0, 2]], 0.0, max(float(image_width - 1), 0.0))
    updated[[1, 3]] = np.clip(updated[[1, 3]], 0.0, max(float(image_height - 1), 0.0))
    return updated


def _combine_bbox_candidates(
    pose_bbox: np.ndarray | None,
    visual_bbox: np.ndarray | None,
    previous: np.ndarray | None,
) -> np.ndarray | None:
    if pose_bbox is None:
        return None if visual_bbox is None else np.asarray(visual_bbox, dtype=float).copy()
    if visual_bbox is None:
        return np.asarray(pose_bbox, dtype=float).copy()
    pose_bbox = np.asarray(pose_bbox, dtype=float).reshape(4)
    visual_bbox = np.asarray(visual_bbox, dtype=float).reshape(4)
    pose_center = (pose_bbox[:2] + pose_bbox[2:]) / 2.0
    visual_center = (visual_bbox[:2] + visual_bbox[2:]) / 2.0
    reference = pose_bbox if previous is None else np.asarray(previous, dtype=float).reshape(4)
    reference_size = np.maximum(reference[2:] - reference[:2], 32.0)
    if np.linalg.norm(pose_center - visual_center) > 0.55 * np.linalg.norm(reference_size):
        return pose_bbox
    pose_size = pose_bbox[2:] - pose_bbox[:2]
    visual_size = visual_bbox[2:] - visual_bbox[:2]
    center = 0.70 * pose_center + 0.30 * visual_center
    size = np.maximum(pose_size, 0.80 * visual_size)
    return np.r_[center - size / 2.0, center + size / 2.0]


def _initial_visual_bbox(frame: np.ndarray) -> np.ndarray | None:
    """Independent visual anchor used to keep pose-derived crops from drifting."""
    from .vitpose_plus_runtime import _initial_person_bbox

    return _initial_person_bbox(frame)


def _motion_person_bbox(previous_frame: np.ndarray | None, frame: np.ndarray) -> np.ndarray | None:
    """Return a generously padded box around the largest moving foreground region."""
    if previous_frame is None or previous_frame.shape != frame.shape:
        return None
    height, width = frame.shape[:2]
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.GaussianBlur(previous_gray, (5, 5), 0)
    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
    difference = cv2.absdiff(previous_gray, current_gray)
    _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
    mask[: int(0.04 * height)] = 0
    mask[int(0.97 * height) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = 0.0007 * float(width * height)
    candidates: list[tuple[float, np.ndarray]] = []
    for component in stats[1:]:
        x, y, box_width, box_height, area = component.astype(float)
        if area < minimum_area or box_height < 0.04 * height:
            continue
        if box_width > 0.75 * width or box_height > 0.95 * height:
            continue
        score = area * (1.0 + box_height / max(height, 1))
        candidates.append((score, np.asarray([x, y, x + box_width, y + box_height], dtype=float)))
    if not candidates:
        return None

    component_bbox = max(candidates, key=lambda item: item[0])[1]
    center = (component_bbox[:2] + component_bbox[2:]) / 2.0
    component_size = component_bbox[2:] - component_bbox[:2]
    # Frame differencing may show only a swinging arm or a moving torso.  A
    # large human-sized pad prevents that partial region from becoming a crop.
    center[1] = max(center[1], 0.50 * height)
    size = np.asarray(
        [max(1.45 * component_size[0], 0.28 * width), max(1.35 * component_size[1], 0.90 * height)],
        dtype=float,
    )
    bbox = np.r_[center - size / 2.0, center + size / 2.0]
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, max(float(width - 1), 0.0))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, max(float(height - 1), 0.0))
    return bbox


def _motion_requires_reacquisition(
    tracked_bbox: np.ndarray | None,
    motion_bbox: np.ndarray | None,
) -> bool:
    if motion_bbox is None:
        return False
    if tracked_bbox is None:
        return True
    tracked = np.asarray(tracked_bbox, dtype=float).reshape(4)
    motion = np.asarray(motion_bbox, dtype=float).reshape(4)
    tracked_center = (tracked[:2] + tracked[2:]) / 2.0
    motion_center = (motion[:2] + motion[2:]) / 2.0
    tracked_size = np.maximum(tracked[2:] - tracked[:2], 32.0)
    normalized_offset = np.linalg.norm((motion_center - tracked_center) / tracked_size)
    return bool(normalized_offset > 0.72)
