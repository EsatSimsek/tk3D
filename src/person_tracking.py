from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PersonDetectorConfig:
    enabled: bool = False
    backend: str = "rfdetr"
    model_variant: str = "small"
    threshold: float = 0.25
    target_confidence_threshold: float = 0.65
    person_class_id: int = 1
    bbox_padding: float = 0.18
    bbox_stationary_alpha: float = 0.35
    bbox_motion_scale_ratio: float = 0.12
    track_activation_threshold: float = 0.25
    minimum_matching_threshold: float = 0.80
    lost_track_buffer: int = 30
    reacquire_after_frames: int = 12
    frame_rate: int = 30
    optimize_for_inference: bool = False

    def __post_init__(self) -> None:
        if self.backend != "rfdetr":
            raise ValueError("Only person detector backend=rfdetr is supported")
        if self.model_variant not in {"nano", "small", "medium", "large"}:
            raise ValueError("RF-DETR model_variant must be nano, small, medium, or large")
        for name, value in (
            ("threshold", self.threshold),
            ("target_confidence_threshold", self.target_confidence_threshold),
            ("track_activation_threshold", self.track_activation_threshold),
            ("minimum_matching_threshold", self.minimum_matching_threshold),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.person_class_id < 0:
            raise ValueError("person_class_id must be non-negative")
        if not np.isfinite(self.bbox_padding) or not 0.0 <= self.bbox_padding <= 1.0:
            raise ValueError("bbox_padding must be finite and between 0 and 1")
        if not np.isfinite(self.bbox_stationary_alpha) or not 0.0 < self.bbox_stationary_alpha <= 1.0:
            raise ValueError("bbox_stationary_alpha must be finite and between 0 and 1")
        if not np.isfinite(self.bbox_motion_scale_ratio) or self.bbox_motion_scale_ratio <= 0.0:
            raise ValueError("bbox_motion_scale_ratio must be finite and positive")
        if self.lost_track_buffer < 1 or self.reacquire_after_frames < 1 or self.frame_rate < 1:
            raise ValueError("tracking frame counts must be positive")


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    bbox_xyxy: np.ndarray
    confidence: float
    track_id: int
    detected_this_frame: bool


class RFDETRPersonTracker:
    """RF-DETR person detector with one persistent athlete identity per camera."""

    def __init__(self, config: PersonDetectorConfig, device: str = "cuda:0") -> None:
        if not config.enabled:
            raise ValueError("RFDETRPersonTracker requires an enabled config")
        try:
            import supervision as sv
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR person tracking requires rfdetr and supervision; install requirements-pose.txt"
            ) from exc

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        detector_device = device if device != "cuda:0" else "cuda"
        self.config = config
        self._sv = sv
        self._model = model_classes[config.model_variant](device=detector_device)
        if config.optimize_for_inference:
            self._model.optimize_for_inference()
        self._trackers: dict[str, Any] = {}
        self._active_track_ids: dict[str, int] = {}
        self._last_results: dict[str, TrackedPerson] = {}
        self._missed_frames: dict[str, int] = {}

    def track(self, frame: np.ndarray, camera_id: str, frame_idx: int) -> TrackedPerson | None:
        del frame_idx  # ByteTrack advances once for each supplied video sample.
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError(f"Expected a non-empty BGR frame with shape HxWx3, got {getattr(frame, 'shape', None)}")
        detections = self._model.predict(frame, threshold=self.config.threshold)
        if isinstance(detections, list):
            if len(detections) != 1:
                raise RuntimeError(f"RF-DETR returned {len(detections)} outputs for one frame")
            detections = detections[0]
        if getattr(detections, "class_id", None) is None:
            raise RuntimeError("RF-DETR returned detections without class IDs")
        person_detections = detections[detections.class_id == self.config.person_class_id]
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            tracker = self._sv.ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                frame_rate=self.config.frame_rate,
                minimum_consecutive_frames=1,
            )
            self._trackers[camera_id] = tracker
        tracked = tracker.update_with_detections(person_detections)
        return self._select_tracked_person(tracked, frame.shape[1], frame.shape[0], camera_id)

    def _select_tracked_person(
        self,
        tracked: Any,
        image_width: int,
        image_height: int,
        camera_id: str,
    ) -> TrackedPerson | None:
        boxes = np.asarray(tracked.xyxy, dtype=float).reshape(-1, 4)
        raw_confidences = getattr(tracked, "confidence", None)
        raw_track_ids = getattr(tracked, "tracker_id", None)
        if raw_confidences is None or raw_track_ids is None:
            if boxes.shape[0] == 0:
                confidences = np.empty((0,), dtype=float)
                track_ids = np.empty((0,), dtype=int)
            else:
                raise RuntimeError("ByteTrack returned tracked boxes without confidence or track IDs")
        else:
            confidences = np.asarray(raw_confidences, dtype=float).reshape(-1)
            track_ids = np.asarray(raw_track_ids, dtype=int).reshape(-1)
        if not (boxes.shape[0] == confidences.shape[0] == track_ids.shape[0]):
            raise RuntimeError("ByteTrack returned inconsistent box, confidence, and track ID counts")
        usable = (
            np.all(np.isfinite(boxes), axis=1)
            & np.isfinite(confidences)
            & (track_ids >= 0)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes = boxes[usable]
        confidences = confidences[usable]
        track_ids = track_ids[usable]
        active_track_id = self._active_track_ids.get(camera_id)

        selected_index: int | None = None
        if active_track_id is not None:
            matches = np.flatnonzero(track_ids == active_track_id)
            if matches.size:
                selected_index = int(matches[0])

        if selected_index is None and active_track_id is None:
            selected_index = _best_initial_track_index(
                boxes,
                confidences,
                minimum_confidence=self.config.target_confidence_threshold,
            )
        elif selected_index is None:
            missed = self._missed_frames.get(camera_id, 0) + 1
            self._missed_frames[camera_id] = missed
            if missed >= self.config.reacquire_after_frames:
                selected_index = _best_initial_track_index(
                    boxes,
                    confidences,
                    minimum_confidence=self.config.target_confidence_threshold,
                )

        if selected_index is None:
            last = self._last_results.get(camera_id)
            missed = self._missed_frames.get(camera_id, 0)
            if last is None or missed > self.config.lost_track_buffer:
                return None
            return TrackedPerson(last.bbox_xyxy.copy(), last.confidence, last.track_id, False)

        track_id = int(track_ids[selected_index])
        self._active_track_ids[camera_id] = track_id
        self._missed_frames[camera_id] = 0
        bbox = _pad_bbox(
            boxes[selected_index],
            image_width=image_width,
            image_height=image_height,
            padding=self.config.bbox_padding,
        )
        previous = self._last_results.get(camera_id)
        if previous is not None and previous.track_id == track_id:
            bbox = _stabilize_bbox(
                bbox,
                previous.bbox_xyxy,
                stationary_alpha=self.config.bbox_stationary_alpha,
                motion_scale_ratio=self.config.bbox_motion_scale_ratio,
            )
        result = TrackedPerson(
            bbox_xyxy=bbox,
            confidence=float(confidences[selected_index]),
            track_id=track_id,
            detected_this_frame=True,
        )
        self._last_results[camera_id] = result
        return result


def person_detector_config_from_mapping(
    raw: dict[str, Any] | None,
    *,
    frame_rate: float,
) -> PersonDetectorConfig:
    if not np.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("frame_rate must be finite and positive")
    values = raw or {}
    return PersonDetectorConfig(
        enabled=bool(values.get("enabled", False)),
        backend=str(values.get("backend", "rfdetr")),
        model_variant=str(values.get("model_variant", "small")),
        threshold=float(values.get("threshold", 0.25)),
        target_confidence_threshold=float(values.get("target_confidence_threshold", 0.65)),
        person_class_id=int(values.get("person_class_id", 1)),
        bbox_padding=float(values.get("bbox_padding", 0.18)),
        bbox_stationary_alpha=float(values.get("bbox_stationary_alpha", 0.35)),
        bbox_motion_scale_ratio=float(values.get("bbox_motion_scale_ratio", 0.12)),
        track_activation_threshold=float(values.get("track_activation_threshold", 0.25)),
        minimum_matching_threshold=float(values.get("minimum_matching_threshold", 0.80)),
        lost_track_buffer=int(values.get("lost_track_buffer", 30)),
        reacquire_after_frames=int(values.get("reacquire_after_frames", 12)),
        frame_rate=max(int(round(frame_rate)), 1),
        optimize_for_inference=bool(values.get("optimize_for_inference", False)),
    )


def _best_initial_track_index(
    boxes: np.ndarray,
    confidences: np.ndarray,
    minimum_confidence: float,
) -> int | None:
    boxes = np.asarray(boxes, dtype=float)
    confidences = np.asarray(confidences, dtype=float).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"boxes must have shape [N, 4], got {boxes.shape}")
    if confidences.shape != (boxes.shape[0],):
        raise ValueError("confidences must contain one value per box")
    if boxes.size == 0 or confidences.size == 0:
        return None
    valid_boxes = (
        np.all(np.isfinite(boxes), axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    eligible = np.flatnonzero(
        valid_boxes & np.isfinite(confidences) & (confidences >= minimum_confidence)
    )
    if eligible.size == 0:
        return None
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0)
    areas = sizes[:, 0] * sizes[:, 1]
    maximum_area = max(float(np.max(areas[eligible])), 1.0)
    scores = confidences[eligible] + 0.08 * np.sqrt(areas[eligible] / maximum_area)
    return int(eligible[int(np.argmax(scores))])


def _pad_bbox(
    bbox: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    padding: float,
) -> np.ndarray:
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(padding) or not 0.0 <= padding <= 1.0:
        raise ValueError("padding must be finite and between 0 and 1")
    values = np.asarray(bbox, dtype=float).reshape(4)
    if not np.all(np.isfinite(values)) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"bbox must be finite and non-empty, got {values.tolist()}")
    center = (values[:2] + values[2:]) / 2.0
    size = np.maximum(values[2:] - values[:2], 2.0) * (1.0 + 2.0 * padding)
    padded = np.r_[center - size / 2.0, center + size / 2.0]
    padded[[0, 2]] = np.clip(padded[[0, 2]], 0.0, max(float(image_width - 1), 0.0))
    padded[[1, 3]] = np.clip(padded[[1, 3]], 0.0, max(float(image_height - 1), 0.0))
    return padded


def _stabilize_bbox(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    stationary_alpha: float,
    motion_scale_ratio: float,
) -> np.ndarray:
    """Suppress detector-box jitter while following genuine athlete motion quickly."""
    current_values = np.asarray(current, dtype=float).reshape(4)
    previous_values = np.asarray(previous, dtype=float).reshape(4)
    if not np.all(np.isfinite(current_values)) or not np.all(np.isfinite(previous_values)):
        raise ValueError("current and previous boxes must be finite")
    if (
        current_values[2] <= current_values[0]
        or current_values[3] <= current_values[1]
        or previous_values[2] <= previous_values[0]
        or previous_values[3] <= previous_values[1]
    ):
        raise ValueError("current and previous boxes must be non-empty")
    if not np.isfinite(stationary_alpha) or not 0.0 < stationary_alpha <= 1.0:
        raise ValueError("stationary_alpha must be finite and between 0 and 1")
    if not np.isfinite(motion_scale_ratio) or motion_scale_ratio <= 0.0:
        raise ValueError("motion_scale_ratio must be finite and positive")
    current_center = (current_values[:2] + current_values[2:]) / 2.0
    previous_center = (previous_values[:2] + previous_values[2:]) / 2.0
    current_size = np.maximum(current_values[2:] - current_values[:2], 2.0)
    previous_size = np.maximum(previous_values[2:] - previous_values[:2], 2.0)
    previous_diagonal = max(float(np.linalg.norm(previous_size)), 2.0)
    center_motion = float(np.linalg.norm(current_center - previous_center)) / previous_diagonal
    scale_motion = float(np.max(np.abs(np.log(current_size / previous_size))))
    motion_ratio = center_motion + scale_motion
    motion_weight = min(motion_ratio / motion_scale_ratio, 1.0)
    alpha = stationary_alpha + (1.0 - stationary_alpha) * motion_weight
    center = previous_center + alpha * (current_center - previous_center)
    size = previous_size + alpha * (current_size - previous_size)
    return np.r_[center - size / 2.0, center + size / 2.0]
