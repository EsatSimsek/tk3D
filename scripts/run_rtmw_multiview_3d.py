"""Legacy RTMW multi-view redirect — now delegates to ViTPose pipeline.

The default 2D pose backend moved from RTMW to ViTPose-Huge WholeBody.
This script is kept as a backward-compatible alias.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_vitpose_multiview_3d import main

if __name__ == "__main__":
    main()
