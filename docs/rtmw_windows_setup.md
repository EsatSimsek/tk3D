# RTMW Windows Environment

Use Python 3.11 for the real RTMW/MMPose video pipeline. Python 3.13 is kept for the lightweight tests, but the live pose stack needs Python 3.11 compatible wheels.

Known-good local runtime used for the AIST test:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" opencv-python==4.10.0.84 openmim mmengine mmcv-lite==2.1.0
python -m pip install mmpose==1.3.2 --no-deps
python -m pip install mmdet==3.3.0 --no-deps
python -m pip install xtcocotools json_tricks munkres shapely terminaltables scipy pycocotools
```

`chumpy` is listed by MMPose but is not needed for the RTMW whole-image inference path used here. On Windows it can fail to build, so the project installs MMPose with `--no-deps` and installs only the runtime dependencies this path needs.

Validation:

```powershell
python scripts\check_models.py --session data\aist_test\session_front_back.yaml
python scripts\run_rtmw_multiview_3d.py --session data\aist_test\session_front_back.yaml --max-frames 3 --stride 60
```
