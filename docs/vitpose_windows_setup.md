# ViTPose Windows Environment

TK3D'nin varsayılan 2D pose yolu ViTPose-Huge WholeBody'dir. Yerel canlı test, bu repodaki `.venv` ortamında doğrulanmıştır.

Aktif yerel ortam:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The 2D config defaults to:

```text
config/mmpose_configs/wholebody_2d_keypoint/vitpose/coco-wholebody/td-hm_ViTPose-huge_8xb64-210e_coco-wholebody-256x192.py
```

Place the matching ViTPose-Huge whole-body checkpoint at:

```text
weights/vitpose_huge_wholebody_256x192.pth
```

TK3D also expects the official ViTPose source tree at:

```text
external/vitpose
```

Official source: the ViTPose repository lists the `ViTPose++-H COCO+AIC+MPII+AP10K+APT36K+WholeBody 256x192`
weight in its WholeBody table:

```text
https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR
```

If OneDrive saves a small HTML file instead of a large `.pth` file, delete it and download again from a browser. `check_models.py` rejects these invalid HTML downloads.

Validation:

```powershell
python scripts\check_models.py --session data\aist_test\session.yaml
python scripts\run_pose2d_overlays.py --session data\aist_test\session.yaml --camera c01 --stride 10
python scripts\run_vitpose_multiview_3d.py --session data\aist_test\session.yaml --stride 10
```

Use `--max-frames` only for a short preview. Omit it when the output video must preserve the full source video duration.

Do not use the MMPose COCO-only `vitpose-h` checkpoint for TK3D. It predicts 17 body keypoints, while TK3D requires 133 COCO-WholeBody keypoints.
