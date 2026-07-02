# SMPL Mesh Setup

This project can render a real SMPL human mesh for AIST++ motion annotations, but the SMPL body model files are license-controlled and are not included in the repository.

Required local files:

```text
models/smpl/SMPL_MALE.pkl
models/smpl/SMPL_FEMALE.pkl   # optional
```

AIST++ motion annotations are expected under:

```text
data/aist_test/annotations/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl
```

Install/runtime packages used in `.venv311`:

```powershell
python -m pip install smplx trimesh
```

Render command after placing `SMPL_MALE.pkl`:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv311\Scripts\Activate.ps1
python scripts\render_aist_smpl_mesh.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 120 --stride 1
```

Outputs:

```text
outputs/aist_test/videos/aist_smpl_mesh.mp4
outputs/aist_test/figures/aist_smpl_mesh_frame0.obj
outputs/aist_test/json/aist_smpl_mesh_report.json
```

If the SMPL model file is missing, the script stops before rendering and prints the exact expected path. This is intentional: the licensed SMPL model cannot be redistributed or auto-downloaded by the project.
