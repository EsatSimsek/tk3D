from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS
from .model_runtime import ModelRuntimeError


class ViTPosePlusWholeBodyInferencer:
    """Minimal ViTPose+ WholeBody runtime for the official multi-head checkpoint."""

    wholebody_head_prefix = "associate_keypoint_heads.4."

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        adapter_checkpoint_path: Path | None = None,
        allow_unapproved_adapter: bool = False,
        repo_root: Path | None = None,
        input_height: int = 256,
        input_width: int = 192,
    ) -> None:
        if input_height <= 0 or input_width <= 0 or input_height % 16 or input_width % 16:
            raise ValueError("ViTPose input dimensions must be positive multiples of 16")
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.checkpoint_path = checkpoint_path
        self.adapter_checkpoint_path = adapter_checkpoint_path
        self.allow_unapproved_adapter = bool(allow_unapproved_adapter)
        self.heatmap_offsets_xy = np.zeros((COCO_WHOLEBODY_KEYPOINTS, 2), dtype=float)
        self.device = self._resolve_device(device)
        self.repo_root = repo_root or Path("external/vitpose")
        self._torch = self._import_torch()
        self._model = self._build_model()

    def __call__(self, frame: np.ndarray, bbox_xyxy: np.ndarray | None = None) -> dict[str, Any]:
        keypoints, scores = self.predict_arrays(frame, bbox_xyxy=bbox_xyxy)
        return {"predictions": [[{"keypoints": keypoints, "keypoint_scores": scores}]]}

    def predict_arrays(
        self,
        frame: np.ndarray,
        bbox_xyxy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor, crop = self._preprocess(frame, bbox_xyxy=bbox_xyxy)
        with self._torch.no_grad():
            heatmaps = self._model(tensor)[0].detach().cpu().numpy()
        return self._decode_heatmaps(heatmaps, crop)

    def _build_model(self) -> Any:
        torch = self._torch
        nn = torch.nn
        ViTMoE = _load_vit_moe_class(self.repo_root)
        input_height = self.input_height
        input_width = self.input_width

        class WholeBodyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = ViTMoE(
                    img_size=(input_height, input_width),
                    patch_size=16,
                    embed_dim=1280,
                    depth=32,
                    num_heads=16,
                    ratio=1,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_path_rate=0.55,
                    num_expert=6,
                    part_features=320,
                )
                self.head = _TopdownHeatmapSimpleHead(1280, COCO_WHOLEBODY_KEYPOINTS)

            def forward(self, x: Any) -> Any:
                source = torch.full((x.shape[0],), 5, dtype=torch.long, device=x.device)
                features = self.backbone(x, source)
                return self.head(features)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
        model = WholeBodyModel()
        model.backbone.load_state_dict(_strip_prefix(state_dict, "backbone."), strict=True)
        model.head.load_state_dict(_strip_prefix(state_dict, self.wholebody_head_prefix), strict=True)
        if self.adapter_checkpoint_path is not None:
            if not self.adapter_checkpoint_path.exists():
                raise FileNotFoundError(f"ViTPose adapter checkpoint not found: {self.adapter_checkpoint_path}")
            try:
                adapter = torch.load(self.adapter_checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                adapter = torch.load(self.adapter_checkpoint_path, map_location="cpu")
            if not isinstance(adapter, dict):
                raise ValueError("ViTPose adapter checkpoint must be a mapping")
            metadata = adapter.get("metadata", {})
            approved = isinstance(metadata, dict) and metadata.get("production_approved") is True
            if not approved and not self.allow_unapproved_adapter:
                raise ValueError(
                    "ViTPose adapter has not passed held-out 3D approval; "
                    "use allow_unapproved_adapter only for diagnostic benchmarking"
                )
            adapter_state = adapter.get("head_state_dict")
            if adapter_state is not None:
                model.head.load_state_dict(adapter_state, strict=True)
            offsets = adapter.get("heatmap_offsets_xy")
            if offsets is not None:
                values = np.asarray(offsets, dtype=float)
                if values.shape != (COCO_WHOLEBODY_KEYPOINTS, 2) or not np.all(np.isfinite(values)):
                    raise ValueError("ViTPose heatmap_offsets_xy must have shape [133, 2] and be finite")
                self.heatmap_offsets_xy = values
            if adapter_state is None and offsets is None:
                raise ValueError("ViTPose adapter contains neither head_state_dict nor heatmap_offsets_xy")
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(
        self,
        frame: np.ndarray,
        bbox_xyxy: np.ndarray | None = None,
    ) -> tuple[Any, tuple[float, float, float, float]]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected a BGR image with shape HxWx3, got {getattr(frame, 'shape', None)}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        initial_bbox = _initial_person_bbox(frame) if bbox_xyxy is None else bbox_xyxy
        crop = _aspect_correct_bbox(initial_bbox, frame.shape[1], frame.shape[0], self.input_width / self.input_height)
        x1, y1, x2, y2 = crop
        # The checkpoint is trained with UDP affine transforms: ROI endpoints
        # map to pixel centers 0 and size-1 instead of the outer image edge.
        scale_x = (self.input_width - 1.0) / max(x2 - x1, 1e-6)
        scale_y = (self.input_height - 1.0) / max(y2 - y1, 1e-6)
        affine = np.asarray([[scale_x, 0.0, -x1 * scale_x], [0.0, scale_y, -y1 * scale_y]], dtype=np.float32)
        warped = cv2.warpAffine(
            rgb,
            affine,
            (self.input_width, self.input_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        image = warped.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return self._torch.from_numpy(image).to(self.device), crop

    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        crop: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        heatmap_height, heatmap_width = heatmaps.shape[1:]
        flat = heatmaps.reshape(COCO_WHOLEBODY_KEYPOINTS, -1)
        indices = np.argmax(flat, axis=1)
        raw_scores = flat[np.arange(COCO_WHOLEBODY_KEYPOINTS), indices]
        peak_xy = np.column_stack(
            [
                (indices % heatmap_width).astype(float),
                (indices // heatmap_width).astype(float),
            ]
        )
        refined_xy = _refine_heatmap_peaks_udp(heatmaps, peak_xy, kernel_size=11)
        refined_xy += getattr(self, "heatmap_offsets_xy", np.zeros_like(refined_xy))
        x1, y1, x2, y2 = crop
        keypoints[:, 0] = x1 + refined_xy[:, 0] * (x2 - x1) / max(heatmap_width - 1.0, 1.0)
        keypoints[:, 1] = y1 + refined_xy[:, 1] * (y2 - y1) / max(heatmap_height - 1.0, 1.0)
        # The checkpoint is trained with Gaussian MSE heatmap targets. Applying a
        # sigmoid makes a zero response look like 0.5 confidence and validates
        # every joint; retain the calibrated heatmap peak instead.
        scores[:] = np.clip(raw_scores, 0.0, 1.0)
        return keypoints, scores

    def _resolve_device(self, requested: str) -> str:
        torch = self._import_torch()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModelRuntimeError("PyTorch is required for ViTPose+ inference.") from exc
        return torch


class _TopdownHeatmapSimpleHead:
    def __new__(cls, in_channels: int, out_channels: int) -> Any:
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.deconv_layers = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

            def forward(self, x: Any) -> Any:
                return self.final_layer(self.deconv_layers(x))

        return Head()


def _refine_heatmap_peaks_udp(
    heatmaps: np.ndarray,
    peak_xy: np.ndarray,
    kernel_size: int = 11,
) -> np.ndarray:
    """DARK/UDP sub-pixel refinement for Gaussian heatmap peaks."""
    values = np.asarray(heatmaps, dtype=np.float32)
    coords = np.asarray(peak_xy, dtype=float).copy()
    if values.ndim != 3 or coords.shape != (values.shape[0], 2):
        raise ValueError("heatmaps and peak_xy shapes are incompatible")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    height, width = values.shape[1:]
    for joint_idx, heatmap in enumerate(values):
        x = int(coords[joint_idx, 0])
        y = int(coords[joint_idx, 1])
        if not (1 <= x < width - 1 and 1 <= y < height - 1):
            continue
        blurred = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
        logged = np.log(np.clip(blurred, 1e-3, 50.0))
        dx = 0.5 * (logged[y, x + 1] - logged[y, x - 1])
        dy = 0.5 * (logged[y + 1, x] - logged[y - 1, x])
        dxx = logged[y, x + 1] - 2.0 * logged[y, x] + logged[y, x - 1]
        dyy = logged[y + 1, x] - 2.0 * logged[y, x] + logged[y - 1, x]
        dxy = 0.25 * (
            logged[y + 1, x + 1]
            - logged[y - 1, x + 1]
            - logged[y + 1, x - 1]
            + logged[y - 1, x - 1]
        )
        hessian = np.asarray([[dxx, dxy], [dxy, dyy]], dtype=float)
        derivative = np.asarray([dx, dy], dtype=float)
        if abs(np.linalg.det(hessian)) <= 1e-9:
            continue
        offset = -np.linalg.solve(hessian, derivative)
        if np.all(np.isfinite(offset)) and np.linalg.norm(offset) <= 2.0:
            coords[joint_idx] += offset
    return coords


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _aspect_correct_bbox(
    bbox_xyxy: np.ndarray | None,
    image_width: int,
    image_height: int,
    target_aspect: float,
    padding: float = 1.25,
) -> tuple[float, float, float, float]:
    if bbox_xyxy is None:
        # Poomsae/AIST recordings are staged around the image center. Use a
        # centered, aspect-correct initial crop; subsequent frames use the
        # confidence-derived tracked person box.
        crop_height = float(image_height)
        crop_width = min(float(image_width), crop_height * target_aspect)
        crop_height = crop_width / target_aspect
        center_x = image_width / 2.0
        center_y = image_height / 2.0
    else:
        bbox = np.asarray(bbox_xyxy, dtype=float).reshape(4)
        if not np.all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"Invalid person bbox: {bbox.tolist()}")
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float((bbox[1] + bbox[3]) / 2.0)
        crop_width = float(bbox[2] - bbox[0]) * padding
        crop_height = float(bbox[3] - bbox[1]) * padding
        if crop_width / max(crop_height, 1e-6) > target_aspect:
            crop_height = crop_width / target_aspect
        else:
            crop_width = crop_height * target_aspect
    crop_width = max(crop_width, 32.0)
    crop_height = max(crop_height, 32.0)
    return (
        center_x - crop_width / 2.0,
        center_y - crop_height / 2.0,
        center_x + crop_width / 2.0,
        center_y + crop_height / 2.0,
    )


def _initial_person_bbox(frame: np.ndarray) -> np.ndarray | None:
    """Find the central foreground person in staged, static-camera recordings.

    This is deliberately conservative: it rejects wall/curtain-sized regions
    and returns ``None`` when no plausible human-sized foreground component is
    found, allowing the aspect-correct center crop fallback.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < 175, 255, 0).astype(np.uint8)
    mask[: int(height * 0.07)] = 0
    mask[int(height * 0.96) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, np.ndarray]] = []
    image_area = float(width * height)
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index].astype(float)
        area_ratio = area / image_area
        height_ratio = box_height / max(height, 1)
        width_ratio = box_width / max(width, 1)
        if not (0.001 <= area_ratio <= 0.12 and 0.12 <= height_ratio <= 0.80 and 0.02 <= width_ratio <= 0.40):
            continue
        center_x, center_y = centroids[index]
        if not (0.12 * height <= center_y <= 0.90 * height):
            continue
        horizontal_distance = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        if horizontal_distance > 0.50:
            continue
        score = area * height_ratio / (1.0 + 6.0 * horizontal_distance)
        candidates.append((score, np.asarray([x, y, x + box_width, y + box_height], dtype=float)))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _load_vit_moe_class(repo_root: Path) -> Any:
    trusted_root = repo_root.resolve()
    source = (trusted_root / "mmpose" / "models" / "backbones" / "vit_moe.py").resolve()
    if trusted_root not in source.parents:
        raise ModelRuntimeError(f"ViTPose source escaped the configured repository root: {source}")
    if not source.exists():
        raise ModelRuntimeError(f"Official ViTPose repository not found: {repo_root}")

    class _Registry:
        @staticmethod
        def register_module() -> Any:
            return lambda cls: cls

    import torch.nn as nn

    class BaseBackbone(nn.Module):
        def init_weights(self, *args: Any, **kwargs: Any) -> None:
            return None

    namespace: dict[str, Any] = {
        "__name__": "tk3d_vitpose_plus_vit_moe",
        "BACKBONES": _Registry(),
        "BaseBackbone": BaseBackbone,
    }
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if "from ..builder import BACKBONES" in line:
            continue
        if "from .base_backbone import BaseBackbone" in line:
            continue
        lines.append(line)
    module = ModuleType("tk3d_vitpose_plus_vit_moe")
    module.__dict__.update(namespace)
    exec("\n".join(lines), module.__dict__)
    return module.__dict__["ViTMoE"]
