from __future__ import annotations

from pathlib import Path

import numpy as np

COCO_BODY_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

def save_reprojection_timeline(reprojection_error: np.ndarray, output_path: str | Path) -> None:
    mean_error = _safe_nanmean_axis1(reprojection_error) if reprojection_error.size else np.array([])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
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
        import matplotlib
        matplotlib.use("Agg", force=True)
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

def write_3d_skeleton_video(
    keypoints_3d_world: np.ndarray,
    path: str | Path,
    fps: float = 30.0,
    size: tuple[int, int] = (1280, 720),
    edges: list[tuple[int, int]] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if keypoints_3d_world.size == 0:
        _write_blank_video(target, size=size, fps=fps)
        return

    try:
        frames = _render_matplotlib_frames(keypoints_3d_world, size=size, edges=edges or COCO_BODY_EDGES)
    except ModuleNotFoundError:
        frames = _render_fallback_frames(keypoints_3d_world, size=size, edges=edges or COCO_BODY_EDGES)

    _write_frames_to_video(frames, target, fps=fps, size=size)

def write_placeholder_3d_video(path: str | Path) -> None:
    _write_blank_video(Path(path))

def _safe_nanmean_axis1(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=1)
    sums = np.nansum(values, axis=1)
    return np.divide(sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)

def _write_blank_video(path: Path, size: tuple[int, int] = (1280, 720), fps: float = 30.0) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()

def _write_frames_to_video(frames: list[np.ndarray], path: Path, fps: float, size: tuple[int, int]) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                frame = cv2.resize(frame, size)
            writer.write(frame)
    finally:
        writer.release()

def _render_matplotlib_frames(
    keypoints_3d_world: np.ndarray,
    size: tuple[int, int],
    edges: list[tuple[int, int]],
) -> list[np.ndarray]:
    import cv2
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    render_points = _center_body_for_render(keypoints_3d_world)
    limits = _axis_limits(render_points)
    frames: list[np.ndarray] = []
    width, height = size
    dpi = 100
    for frame_idx, keypoints in enumerate(render_points):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f7f7f7")
        fig.patch.set_facecolor("#ffffff")
        _draw_3d_pose(ax, keypoints, edges)
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])
        ax.set_zlim(*limits["z"])
        ax.set_xlabel("X body")
        ax.set_ylabel("Y body")
        ax.set_zlabel("Z body")
        ax.set_title(f"TK3D 3D human skeleton - frame {frame_idx}")
        ax.view_init(elev=18, azim=-65)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        frames.append(bgr)
        plt.close(fig)
    return frames

def _draw_3d_pose(ax: object, keypoints: np.ndarray, edges: list[tuple[int, int]]) -> None:
    valid = np.all(np.isfinite(keypoints), axis=1)
    if np.any(valid):
        xyz = keypoints[valid]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=18, c="#0f766e", depthshade=True)
    for start, end in edges:
        if start >= keypoints.shape[0] or end >= keypoints.shape[0]:
            continue
        if not valid[start] or not valid[end]:
            continue
        segment = keypoints[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#1d4ed8", linewidth=2)

def _render_fallback_frames(
    keypoints_3d_world: np.ndarray,
    size: tuple[int, int],
    edges: list[tuple[int, int]],
) -> list[np.ndarray]:
    import cv2

    width, height = size
    render_points = _center_body_for_render(keypoints_3d_world)
    points_2d = _normalize_points_for_image(render_points, size=size)
    frames = []
    for frame_idx in range(keypoints_3d_world.shape[0]):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        pts = points_2d[frame_idx]
        valid = np.all(np.isfinite(pts), axis=1)
        for start, end in edges:
            if start < pts.shape[0] and end < pts.shape[0] and valid[start] and valid[end]:
                cv2.line(frame, tuple(pts[start].astype(int)), tuple(pts[end].astype(int)), (180, 80, 20), 2)
        for point, is_valid in zip(pts, valid):
            if is_valid:
                cv2.circle(frame, tuple(point.astype(int)), 3, (20, 120, 110), -1)
        cv2.putText(frame, f"TK3D world skeleton frame {frame_idx}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
        frames.append(frame)
    return frames

def _center_body_for_render(keypoints_3d_world: np.ndarray) -> np.ndarray:
    """Return a human-centered copy for visualization only.

    WholeBody triangulation can contain noisy face/hand outliers. The exported
    JSON/CSV keeps the original world coordinates; the video render centers on
    stable COCO body joints so the person is visible instead of becoming a dot
    in a huge world-coordinate box.
    """

    points = np.asarray(keypoints_3d_world, dtype=float).copy()
    if points.ndim != 3 or points.shape[1] == 0:
        return points

    body_count = min(points.shape[1], 17)
    body = points[:, :body_count, :]
    centered = np.full_like(points, np.nan, dtype=float)

    anchor_indices = [idx for idx in (5, 6, 11, 12) if idx < body_count]
    for frame_idx in range(points.shape[0]):
        anchors = body[frame_idx, anchor_indices] if anchor_indices else body[frame_idx]
        valid_anchors = anchors[np.all(np.isfinite(anchors), axis=1)]
        if valid_anchors.size:
            center = np.nanmean(valid_anchors, axis=0)
        else:
            valid_body = body[frame_idx][np.all(np.isfinite(body[frame_idx]), axis=1)]
            if not valid_body.size:
                continue
            center = np.nanmedian(valid_body, axis=0)
        centered[frame_idx] = points[frame_idx] - center

    return centered

def _axis_limits(keypoints_3d_world: np.ndarray) -> dict[str, tuple[float, float]]:
    finite = keypoints_3d_world[np.all(np.isfinite(keypoints_3d_world), axis=-1)]
    if finite.size == 0:
        return {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 2.0)}
    mins = np.nanpercentile(finite, 5, axis=0)
    maxs = np.nanpercentile(finite, 95, axis=0)
    centers = (mins + maxs) / 2.0
    # TK3D analysis coordinates are meters. A 0.5 m minimum keeps a human
    # visible without expanding a normal skeleton into a 50 m scene.
    spans = np.maximum(maxs - mins, 0.5)
    spans = np.full(3, float(np.max(spans) * 1.25))
    return {
        "x": (float(centers[0] - spans[0] / 2), float(centers[0] + spans[0] / 2)),
        "y": (float(centers[1] - spans[1] / 2), float(centers[1] + spans[1] / 2)),
        "z": (float(centers[2] - spans[2] / 2), float(centers[2] + spans[2] / 2)),
    }

def _normalize_points_for_image(keypoints_3d_world: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    xy = keypoints_3d_world[..., [0, 1]].copy()
    finite = np.all(np.isfinite(xy), axis=-1)
    output = np.full_like(xy, np.nan, dtype=float)
    if not np.any(finite):
        return output
    values = xy[finite]
    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (xy - mins) / span
    output[..., 0] = 80 + normalized[..., 0] * (width - 160)
    output[..., 1] = height - (80 + normalized[..., 1] * (height - 160))
    return output

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
