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

    input_width = 192
    input_height = 256
    wholebody_head_prefix = "associate_keypoint_heads.4."

    def __init__(self, checkpoint_path: Path, device: str, repo_root: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)
        self.repo_root = repo_root or Path("external/vitpose")
        self._torch = self._import_torch()
        self._model = self._build_model()

    def __call__(self, frame: np.ndarray) -> dict[str, Any]:
        keypoints, scores = self.predict_arrays(frame)
        return {"predictions": [[{"keypoints": keypoints, "keypoint_scores": scores}]]}

    def predict_arrays(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = self._preprocess(frame)
        with self._torch.no_grad():
            heatmaps = self._model(tensor)[0].detach().cpu().numpy()
        return self._decode_heatmaps(heatmaps, frame.shape[1], frame.shape[0])

    def _build_model(self) -> Any:
        torch = self._torch
        nn = torch.nn
        ViTMoE = _load_vit_moe_class(self.repo_root)

        class WholeBodyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = ViTMoE(
                    img_size=(256, 192),
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

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
        model = WholeBodyModel()
        model.backbone.load_state_dict(_strip_prefix(state_dict, "backbone."), strict=True)
        model.head.load_state_dict(_strip_prefix(state_dict, self.wholebody_head_prefix), strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, frame: np.ndarray) -> Any:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return self._torch.from_numpy(image).to(self.device)

    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        keypoints = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        heatmap_height, heatmap_width = heatmaps.shape[1:]
        flat = heatmaps.reshape(COCO_WHOLEBODY_KEYPOINTS, -1)
        indices = np.argmax(flat, axis=1)
        raw_scores = flat[np.arange(COCO_WHOLEBODY_KEYPOINTS), indices]
        xs = indices % heatmap_width
        ys = indices // heatmap_width
        keypoints[:, 0] = (xs + 0.5) * original_width / heatmap_width
        keypoints[:, 1] = (ys + 0.5) * original_height / heatmap_height
        scores[:] = 1.0 / (1.0 + np.exp(-raw_scores))
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


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_vit_moe_class(repo_root: Path) -> Any:
    source = repo_root / "mmpose" / "models" / "backbones" / "vit_moe.py"
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
