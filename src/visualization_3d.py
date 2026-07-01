from __future__ import annotations

from pathlib import Path

import numpy as np


def save_reprojection_timeline(reprojection_error: np.ndarray, output_path: str | Path) -> None:
    mean_error = np.nanmean(reprojection_error, axis=1) if reprojection_error.size else np.array([])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _save_fallback_png(path, mean_error.reshape(1, -1), scale_to_255=True)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_error)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean reprojection error (px)")
    ax.set_title("Reprojection Error Timeline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(data: np.ndarray, output_path: str | Path, title: str, ylabel: str = "Frame") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _save_fallback_png(path, data, scale_to_255=True)
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Keypoint index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_placeholder_3d_video(path: str | Path) -> None:
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, 30.0, (1280, 720))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()


def _save_fallback_png(path: Path, data: np.ndarray, scale_to_255: bool) -> None:
    import cv2

    image = np.asarray(data, dtype=float)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim == 1:
        image = image.reshape(1, -1)
    if scale_to_255:
        max_value = float(np.max(image)) if image.size else 0.0
        min_value = float(np.min(image)) if image.size else 0.0
        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
    image = cv2.resize(image, (max(320, image.shape[1] * 8), max(120, image.shape[0] * 8)))
    cv2.imwrite(str(path), image)
