from __future__ import annotations

import sys
import types


def install_mmpose_runtime_compat() -> None:
    install_mmcv_ops_stub()
    install_xtcocotools_alias()


def install_mmcv_ops_stub() -> None:
    """Allow MMPose model imports with mmcv-lite on Windows.

    TK3D's pose adapters do not use MultiScaleDeformableAttention, but MMPose imports all
    heads during registry setup and EDPose imports this symbol. Full mmcv is
    difficult to build on Windows, so this stub keeps unrelated imports from
    blocking live pose inference.
    """

    if "mmcv.ops" in sys.modules:
        return
    ops = types.ModuleType("mmcv.ops")
    ops.MultiScaleDeformableAttention = type("MultiScaleDeformableAttention", (), {})
    sys.modules["mmcv.ops"] = ops


def install_xtcocotools_alias() -> None:
    """Use pycocotools when xtcocotools wheels are unavailable.

    MMPose imports xtcocotools for COCO metadata parsing. On newer Windows
    Python builds, xtcocotools may fail to build, while pycocotools provides the
    compatible COCO modules needed on this inference path.
    """

    if "xtcocotools" in sys.modules:
        return
    try:
        import pycocotools
        from pycocotools import coco, cocoeval, mask
    except ModuleNotFoundError:
        return
    xtcoco = types.ModuleType("xtcocotools")
    xtcoco.__path__ = []
    xtcoco.coco = coco
    xtcoco.cocoeval = cocoeval
    xtcoco.mask = mask
    sys.modules["xtcocotools"] = xtcoco
    sys.modules["xtcocotools.coco"] = coco
    sys.modules["xtcocotools.cocoeval"] = cocoeval
    sys.modules["xtcocotools.mask"] = mask
