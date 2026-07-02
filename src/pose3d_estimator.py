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
        raise ModelRuntimeError(
            "RTMW3D-x live inference adapter is staged, but output parsing must be matched to the installed "
            "RTMW3D package version. Multi-view triangulation remains the final 3D source."
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
