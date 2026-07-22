from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, PersonPose2D


BODY_LEFT_RIGHT_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


@dataclass(frozen=True, slots=True)
class TemporalPose2DConfig:
    enabled: bool = True
    stationary_alpha: float = 0.22
    motion_alpha: float = 0.88
    motion_scale_ratio: float = 0.08
    max_jump_ratio: float = 0.55
    swap_cost_ratio: float = 0.65
    stabilize_left_right: bool = False
    velocity_alpha: float = 0.35
    minimum_scale_px: float = 48.0

    def __post_init__(self) -> None:
        if not 0.0 < self.stationary_alpha <= self.motion_alpha <= 1.0:
            raise ValueError("Temporal alpha values must satisfy 0 < stationary <= motion <= 1")
        if self.motion_scale_ratio <= 0.0 or self.max_jump_ratio <= 0.0:
            raise ValueError("Temporal motion and jump ratios must be positive")
        if not 0.0 < self.swap_cost_ratio < 1.0:
            raise ValueError("swap_cost_ratio must be between 0 and 1")
        if not 0.0 <= self.velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be between 0 and 1")
        if self.minimum_scale_px <= 0.0:
            raise ValueError("minimum_scale_px must be positive")


@dataclass(slots=True)
class _PoseState:
    frame_idx: int
    xy: np.ndarray
    velocity: np.ndarray
    scores: np.ndarray
    valid: np.ndarray


class TemporalPose2DFilter:
    """Confidence-aware causal pose filter with rear-view left/right stabilization.

    The filter deliberately follows large, confident motion quickly while applying
    stronger smoothing to the small frame-to-frame changes that appear as jitter.
    Frame index deltas are respected, so sparse diagnostic runs are not treated as
    consecutive high-frame-rate observations.
    """

    def __init__(self, config: TemporalPose2DConfig | None = None) -> None:
        self.config = config or TemporalPose2DConfig()
        self._states: dict[str, _PoseState] = {}

    def reset(self, camera_id: str | None = None) -> None:
        if camera_id is None:
            self._states.clear()
        else:
            self._states.pop(camera_id, None)

    def filter(self, pose: PersonPose2D) -> PersonPose2D:
        if not self.config.enabled:
            return pose
        state = self._states.get(pose.camera_id)
        raw_xy = np.asarray(pose.keypoints_xy, dtype=float).copy()
        raw_scores = np.asarray(pose.scores, dtype=float).copy()
        raw_valid = np.asarray(pose.valid_mask, dtype=bool).copy()
        raw_valid &= np.all(np.isfinite(raw_xy), axis=1)

        if state is None or pose.frame_idx <= state.frame_idx:
            return self._initialize(pose, raw_xy, raw_scores, raw_valid)

        dt = max(int(pose.frame_idx - state.frame_idx), 1)
        predicted = state.xy + state.velocity * dt
        scale = max(_body_scale_px(raw_xy, raw_valid), _body_scale_px(state.xy, state.valid), self.config.minimum_scale_px)
        if self.config.stabilize_left_right:
            _stabilize_body_left_right(
                raw_xy,
                raw_scores,
                raw_valid,
                predicted,
                state.valid,
                scale,
                self.config.swap_cost_ratio,
            )

        filtered_xy = raw_xy.copy()
        filtered_scores = raw_scores.copy()
        filtered_valid = raw_valid.copy()
        velocity = np.zeros((COCO_WHOLEBODY_KEYPOINTS, 2), dtype=float)

        for joint_idx in range(COCO_WHOLEBODY_KEYPOINTS):
            if not raw_valid[joint_idx]:
                filtered_xy[joint_idx] = predicted[joint_idx] if state.valid[joint_idx] else np.nan
                filtered_scores[joint_idx] = 0.0
                filtered_valid[joint_idx] = False
                velocity[joint_idx] = state.velocity[joint_idx] if state.valid[joint_idx] else 0.0
                continue
            if not state.valid[joint_idx] or not np.all(np.isfinite(predicted[joint_idx])):
                velocity[joint_idx] = 0.0
                continue

            residual = raw_xy[joint_idx] - predicted[joint_idx]
            # Alpha must respond to the complete displacement between samples.
            # Dividing this by ``dt`` made sparse runs (for example stride=20)
            # look stationary and left the rendered skeleton far behind the
            # athlete.  Only jump rejection needs a per-frame displacement.
            motion_ratio = float(np.linalg.norm(residual)) / max(scale, 1e-6)
            jump_ratio_per_frame = motion_ratio / dt
            confidence = float(np.clip(raw_scores[joint_idx], 0.0, 1.0))
            if jump_ratio_per_frame > self.config.max_jump_ratio and confidence < 0.75:
                filtered_xy[joint_idx] = predicted[joint_idx]
                filtered_scores[joint_idx] = 0.0
                filtered_valid[joint_idx] = False
                velocity[joint_idx] = state.velocity[joint_idx]
                continue

            motion_weight = min(motion_ratio / self.config.motion_scale_ratio, 1.0)
            alpha = self.config.stationary_alpha + (
                self.config.motion_alpha - self.config.stationary_alpha
            ) * motion_weight
            alpha *= 0.55 + 0.45 * confidence
            alpha = float(np.clip(alpha, self.config.stationary_alpha * 0.5, 1.0))
            filtered_xy[joint_idx] = predicted[joint_idx] + alpha * residual
            measured_velocity = (filtered_xy[joint_idx] - state.xy[joint_idx]) / dt
            velocity[joint_idx] = (
                (1.0 - self.config.velocity_alpha) * state.velocity[joint_idx]
                + self.config.velocity_alpha * measured_velocity
            )

        self._states[pose.camera_id] = _PoseState(
            frame_idx=pose.frame_idx,
            xy=filtered_xy.copy(),
            velocity=velocity,
            scores=filtered_scores.copy(),
            valid=filtered_valid.copy(),
        )
        return PersonPose2D(
            camera_id=pose.camera_id,
            frame_idx=pose.frame_idx,
            keypoints_xy=filtered_xy,
            scores=filtered_scores,
            valid_mask=filtered_valid,
            person_id=pose.person_id,
        )

    def _initialize(
        self,
        pose: PersonPose2D,
        xy: np.ndarray,
        scores: np.ndarray,
        valid: np.ndarray,
    ) -> PersonPose2D:
        self._states[pose.camera_id] = _PoseState(
            frame_idx=pose.frame_idx,
            xy=xy.copy(),
            velocity=np.zeros((COCO_WHOLEBODY_KEYPOINTS, 2), dtype=float),
            scores=scores.copy(),
            valid=valid.copy(),
        )
        return PersonPose2D(
            camera_id=pose.camera_id,
            frame_idx=pose.frame_idx,
            keypoints_xy=xy,
            scores=scores,
            valid_mask=valid,
            person_id=pose.person_id,
        )


def _body_scale_px(xy: np.ndarray, valid: np.ndarray) -> float:
    body_valid = np.asarray(valid[:17], dtype=bool) & np.all(np.isfinite(xy[:17]), axis=1)
    if np.count_nonzero(body_valid) < 2:
        return 0.0
    points = xy[:17][body_valid]
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def _stabilize_body_left_right(
    xy: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    predicted: np.ndarray,
    predicted_valid: np.ndarray,
    scale: float,
    swap_cost_ratio: float,
) -> None:
    for left, right in BODY_LEFT_RIGHT_PAIRS:
        if not (valid[left] and valid[right] and predicted_valid[left] and predicted_valid[right]):
            continue
        direct_cost = float(np.linalg.norm(xy[left] - predicted[left]) + np.linalg.norm(xy[right] - predicted[right]))
        swapped_cost = float(np.linalg.norm(xy[right] - predicted[left]) + np.linalg.norm(xy[left] - predicted[right]))
        if direct_cost < 0.12 * scale or swapped_cost >= swap_cost_ratio * direct_cost:
            continue
        xy[[left, right]] = xy[[right, left]]
        scores[[left, right]] = scores[[right, left]]
        valid[[left, right]] = valid[[right, left]]
