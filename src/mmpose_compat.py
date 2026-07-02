from __future__ import annotations

import sys
import types


def install_mmcv_ops_stub() -> None:
    """Allow RTMW imports with mmcv-lite on Windows.

    RTMW does not use MultiScaleDeformableAttention, but MMPose imports all
    heads during registry setup and EDPose imports this symbol. Full mmcv is
    difficult to build on Windows, so this stub keeps unrelated imports from
    blocking RTMW inference.
    """

    if "mmcv.ops" in sys.modules:
        return
    ops = types.ModuleType("mmcv.ops")
    ops.MultiScaleDeformableAttention = type("MultiScaleDeformableAttention", (), {})
    sys.modules["mmcv.ops"] = ops
