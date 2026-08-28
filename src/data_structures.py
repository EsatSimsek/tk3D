from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


COCO_WHOLEBODY_KEYPOINTS = 133
COCO_BODY_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_BODY_JOINTS: dict[str, int] = {name: idx for idx, name in enumerate(COCO_BODY_JOINT_NAMES)}
COCO_BODY_JOINT_INDICES: tuple[int, ...] = tuple(COCO_BODY_JOINTS.values())
COCO_FOOT_JOINTS: dict[str, int] = {
    "left_big_toe": 17,
    "left_small_toe": 18,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_small_toe": 21,
    "right_heel": 22,
}
COCO_FACE_INDICES: tuple[int, ...] = tuple(range(23, 91))
COCO_LEFT_HAND_OFFSET = 91
COCO_RIGHT_HAND_OFFSET = 112
COCO_HAND_LOCAL_JOINTS: dict[str, int] = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}
COCO_LEFT_HAND_INDICES: tuple[int, ...] = tuple(range(COCO_LEFT_HAND_OFFSET, COCO_RIGHT_HAND_OFFSET))
COCO_RIGHT_HAND_INDICES: tuple[int, ...] = tuple(range(COCO_RIGHT_HAND_OFFSET, COCO_WHOLEBODY_KEYPOINTS))


def coco_hand_joint(side: str, joint_name: str) -> int:
    if side not in {"left", "right"}:
        raise ValueError("hand side must be left or right")
    if joint_name not in COCO_HAND_LOCAL_JOINTS:
        raise ValueError(f"unknown hand joint: {joint_name}")
    offset = COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET
    return offset + COCO_HAND_LOCAL_JOINTS[joint_name]


@dataclass(slots=True)
class CameraView:
    camera_id: str
    video_path: Path
    calibration_video_path: Path | None = None
    frame_offset: int = 0
    time_offset_sec: float = 0.0


@dataclass(slots=True)
class Session:
    session_id: str
    task_name: str
    root_dir: Path
    cameras: list[CameraView]
    fps: float | None = None
    sync_method: str = "timestamp"


@dataclass(slots=True)
class PersonPose2D:
    camera_id: str
    frame_idx: int
    keypoints_xy: np.ndarray
    scores: np.ndarray
    valid_mask: np.ndarray
    person_id: int = 0

    def __post_init__(self) -> None:
        self.keypoints_xy = _as_shape(self.keypoints_xy, (COCO_WHOLEBODY_KEYPOINTS, 2), "keypoints_xy")
        self.scores = _as_shape(self.scores, (COCO_WHOLEBODY_KEYPOINTS,), "scores")
        self.valid_mask = _as_shape(self.valid_mask.astype(bool), (COCO_WHOLEBODY_KEYPOINTS,), "valid_mask")


@dataclass(slots=True)
class PersonPose3D:
    camera_id: str
    frame_idx: int
    keypoints_camera: np.ndarray
    scores: np.ndarray
    person_id: int = 0

    def __post_init__(self) -> None:
        self.keypoints_camera = _as_shape(
            self.keypoints_camera, (COCO_WHOLEBODY_KEYPOINTS, 3), "keypoints_camera"
        )
        self.scores = _as_shape(self.scores, (COCO_WHOLEBODY_KEYPOINTS,), "scores")


@dataclass(slots=True)
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]
    intrinsic_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    projection_matrix: np.ndarray
    reprojection_error_px: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            camera_id=data["camera_id"],
            image_size=tuple(data["image_size"]),
            intrinsic_matrix=np.asarray(data["intrinsic_matrix"], dtype=float),
            distortion_coefficients=np.asarray(data["distortion_coefficients"], dtype=float),
            rotation_vector=np.asarray(data["rotation_vector"], dtype=float),
            translation_vector=np.asarray(data["translation_vector"], dtype=float),
            projection_matrix=np.asarray(data["projection_matrix"], dtype=float),
            reprojection_error_px=data.get("reprojection_error_px"),
        )


@dataclass(slots=True)
class TriangulatedPose3D:
    frame_idx: int
    keypoints_3d_world: np.ndarray
    triangulation_score: np.ndarray
    reprojection_error: np.ndarray
    used_cameras: np.ndarray

    def __post_init__(self) -> None:
        self.keypoints_3d_world = _as_shape(
            self.keypoints_3d_world, (COCO_WHOLEBODY_KEYPOINTS, 3), "keypoints_3d_world"
        )
        self.triangulation_score = _as_shape(
            self.triangulation_score, (COCO_WHOLEBODY_KEYPOINTS,), "triangulation_score"
        )
        self.reprojection_error = _as_shape(
            self.reprojection_error, (COCO_WHOLEBODY_KEYPOINTS,), "reprojection_error"
        )
        self.used_cameras = _as_shape(self.used_cameras, (COCO_WHOLEBODY_KEYPOINTS,), "used_cameras")


@dataclass(slots=True)
class Validation:
    frame_valid_ratio: np.ndarray
    joint_valid_ratio: np.ndarray
    mean_reprojection_error_px: np.ndarray
    warnings: list[str] = field(default_factory=list)


def empty_pose_2d(camera_id: str, frame_idx: int) -> PersonPose2D:
    xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
    scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
    valid = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=bool)
    return PersonPose2D(camera_id=camera_id, frame_idx=frame_idx, keypoints_xy=xy, scores=scores, valid_mask=valid)


def _as_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value
