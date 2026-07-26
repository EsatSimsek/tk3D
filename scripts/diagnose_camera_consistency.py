from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import load_calibration_bundle
from src.coordinate_system import transform_points
from src.data_structures import COCO_BODY_JOINT_NAMES


LEFT_RIGHT_SWAP = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=int,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose one camera against an existing multi-view 3D run."
    )
    parser.add_argument("--run", required=True, help="Run output directory")
    parser.add_argument("--camera", required=True, help="Camera id, for example c05")
    parser.add_argument(
        "--calibration",
        default="outputs/aist_test/calibration/cameras.json",
    )
    parser.add_argument("--max-shift", type=int, default=12)
    args = parser.parse_args()

    run_root = Path(args.run)
    bundle = load_calibration_bundle(Path(args.calibration))
    if args.camera not in bundle.calibrations:
        raise SystemExit(f"Unknown camera: {args.camera}")
    with (run_root / "json" / "vitpose_session_3d.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        session = json.load(file)
    points_analysis = np.asarray(session["keypoints_3d_world"], dtype=float)[:, :17]
    source_to_analysis = np.asarray(bundle.metadata["source_to_analysis"], dtype=float)
    points_source = transform_points(points_analysis, np.linalg.inv(source_to_analysis))

    csv_candidates = (
        run_root / "csv" / "vitpose_keypoints_2d_geometry_flat.csv",
        run_root / "csv" / "vitpose_keypoints_2d_flat.csv",
    )
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        raise SystemExit("Run has no readable 2D keypoint CSV")
    all_table = pd.read_csv(csv_path)
    table = all_table[
        (all_table["camera_id"] == args.camera)
        & (all_table["joint_idx"] < len(COCO_BODY_JOINT_NAMES))
    ].sort_values(["frame_idx", "joint_idx"])
    frame_ids = np.asarray(session["frame_indices"], dtype=int)
    observed = _body_array(table, frame_ids)
    frame_count = min(observed.shape[0], points_source.shape[0])
    observed = observed[:frame_count]
    points_source = points_source[:frame_count]
    projected = _project(points_source, bundle.calibrations[args.camera])
    valid = np.all(np.isfinite(observed), axis=-1) & np.all(
        np.isfinite(projected),
        axis=-1,
    )
    direct_error = np.linalg.norm(observed - projected, axis=-1)
    swapped_error = np.linalg.norm(observed[:, LEFT_RIGHT_SWAP] - projected, axis=-1)

    delta = observed - projected
    median_translation = np.median(delta[valid], axis=0) if np.any(valid) else np.zeros(2)
    translated_error = np.linalg.norm(observed - (projected + median_translation), axis=-1)
    affine, affine_error = _affine_alignment(projected, observed, valid)
    shifts = _temporal_shift_scan(
        observed,
        projected,
        max_shift=max(int(args.max_shift), 0),
    )
    best_shift = min(shifts, key=lambda item: item["median_error_px"]) if shifts else None
    best_shift_windows = (
        _temporal_shift_windows(
            observed,
            projected,
            int(best_shift["shift_frames"]),
        )
        if best_shift is not None
        else []
    )
    motion_sync = _pose_motion_sync(
        all_table,
        frame_ids,
        args.camera,
        max_shift=max(int(args.max_shift), 0),
    )
    zero_geometric = next(
        (item for item in shifts if item["shift_frames"] == 0),
        None,
    )
    shift_assessment = _assess_temporal_shift(
        best_shift,
        zero_geometric,
        motion_sync,
    )

    regions = {
        "face": list(range(0, 5)),
        "arms": list(range(5, 11)),
        "legs": list(range(11, 17)),
        "torso_and_legs": [5, 6, 11, 12, 13, 14, 15, 16],
    }
    output = {
        "camera_id": args.camera,
        "run": str(run_root.resolve()),
        "frame_count": frame_count,
        "direct_error_px": _distribution(direct_error[valid]),
        "left_right_swapped_error_px": _distribution(swapped_error[valid]),
        "median_translation_xy_px": median_translation.tolist(),
        "translation_corrected_error_px": _distribution(translated_error[valid]),
        "partial_affine_xy": None if affine is None else affine.tolist(),
        "partial_affine_corrected_error_px": _distribution(affine_error),
        "temporal_shift_scan": shifts,
        "best_temporal_shift": best_shift,
        "best_temporal_shift_windows": best_shift_windows,
        "independent_pose_motion_sync": motion_sync,
        "temporal_shift_assessment": shift_assessment,
        "regions": {
            name: {
                "direct_error_px": _distribution(
                    direct_error[:, indices][valid[:, indices]]
                ),
                "left_right_swapped_error_px": _distribution(
                    swapped_error[:, indices][valid[:, indices]]
                ),
            }
            for name, indices in regions.items()
        },
    }
    print(json.dumps(output, indent=2))


def _body_array(table: pd.DataFrame, frame_ids: np.ndarray) -> np.ndarray:
    output = np.full((frame_ids.size, 17, 2), np.nan, dtype=float)
    frame_lookup = {int(frame_idx): index for index, frame_idx in enumerate(frame_ids)}
    for row in table.itertuples(index=False):
        array_idx = frame_lookup.get(int(row.frame_idx))
        joint_idx = int(row.joint_idx)
        if array_idx is not None and 0 <= joint_idx < 17:
            output[array_idx, joint_idx] = [float(row.x), float(row.y)]
    return output


def _pose_motion_sync(
    table: pd.DataFrame,
    frame_ids: np.ndarray,
    target_camera: str,
    max_shift: int,
) -> dict[str, object]:
    camera_ids = sorted(str(value) for value in table["camera_id"].unique())
    if target_camera not in camera_ids or len(camera_ids) < 2:
        return {"available": False, "reason": "need at least two cameras"}
    signatures = {}
    for camera_id in camera_ids:
        camera_table = table[table["camera_id"] == camera_id]
        features = _motion_feature_array(camera_table, frame_ids)
        signatures[camera_id] = _motion_signature(features)
    reference_stack = np.stack(
        [_robust_standardize(signatures[camera_id]) for camera_id in camera_ids if camera_id != target_camera],
        axis=0,
    )
    reference = np.nanmedian(reference_stack, axis=0)
    target = _robust_standardize(signatures[target_camera])
    scan = []
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            shifted_target = target[shift:]
            shifted_reference = reference[: target.size - shift]
        else:
            shifted_target = target[: target.size + shift]
            shifted_reference = reference[-shift:]
        valid = np.isfinite(shifted_target) & np.isfinite(shifted_reference)
        correlation = None
        if np.count_nonzero(valid) >= 12:
            first = shifted_target[valid]
            second = shifted_reference[valid]
            if np.std(first) > 1e-8 and np.std(second) > 1e-8:
                correlation = float(np.corrcoef(first, second)[0, 1])
        if correlation is not None and np.isfinite(correlation):
            scan.append({"shift_frames": shift, "correlation": correlation})
    best = max(scan, key=lambda item: item["correlation"]) if scan else None
    zero = next(
        (item["correlation"] for item in scan if item["shift_frames"] == 0),
        None,
    )
    return {
        "available": best is not None,
        "best_shift": best,
        "zero_shift_correlation": zero,
        "scan": scan,
    }


def _assess_temporal_shift(
    geometric_best: dict[str, float | int] | None,
    geometric_zero: dict[str, float | int] | None,
    motion_sync: dict[str, object],
) -> dict[str, object]:
    motion_best = motion_sync.get("best_shift")
    motion_zero = motion_sync.get("zero_shift_correlation")
    if (
        geometric_best is None
        or geometric_zero is None
        or not isinstance(motion_best, dict)
        or motion_zero is None
    ):
        return {
            "status": "insufficient_evidence",
            "recommended_frame_offset": None,
        }
    geometric_ratio = float(geometric_best["median_error_px"]) / max(
        float(geometric_zero["median_error_px"]),
        1e-9,
    )
    motion_improvement = float(motion_best["correlation"]) - float(motion_zero)
    shifts_agree = (
        abs(
            int(geometric_best["shift_frames"])
            - int(motion_best["shift_frames"])
        )
        <= 3
    )
    if geometric_ratio < 0.75 and motion_improvement >= 0.15 and shifts_agree:
        return {
            "status": "supported",
            "recommended_frame_offset": -int(geometric_best["shift_frames"]),
            "reason": "geometry and independent pose-motion signals agree",
        }
    if geometric_ratio < 0.75 and not shifts_agree:
        return {
            "status": "ambiguous_periodic_motion",
            "recommended_frame_offset": None,
            "reason": (
                "geometry and independent motion favor different shifts; "
                "do not change session synchronization automatically"
            ),
        }
    return {
        "status": "not_supported",
        "recommended_frame_offset": None,
        "reason": "a non-zero shift is not supported by both independent signals",
    }


def _motion_signature(body_xy: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(body_xy), axis=-1)
    mins = np.nanmin(np.where(valid[..., None], body_xy, np.nan), axis=1)
    maxs = np.nanmax(np.where(valid[..., None], body_xy, np.nan), axis=1)
    scale = np.linalg.norm(maxs - mins, axis=-1)
    finite_scale = scale[np.isfinite(scale) & (scale > 1e-6)]
    fallback = float(np.median(finite_scale)) if finite_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    velocity = np.linalg.norm(np.diff(body_xy, axis=0), axis=-1)
    velocity_valid = valid[1:] & valid[:-1] & np.isfinite(velocity)
    normalized = np.where(velocity_valid, velocity / scale[1:, None], np.nan)
    signature = np.r_[np.nan, np.nanmedian(normalized, axis=1)]
    finite = np.isfinite(signature)
    if np.count_nonzero(finite) >= 5:
        filled = np.interp(
            np.arange(signature.size),
            np.flatnonzero(finite),
            signature[finite],
        )
        kernel = np.ones(7, dtype=float) / 7.0
        signature = np.convolve(filled, kernel, mode="same")
    return signature


def _motion_feature_array(
    table: pd.DataFrame,
    frame_ids: np.ndarray,
) -> np.ndarray:
    joint_count = 133
    points = np.full((frame_ids.size, joint_count, 2), np.nan, dtype=float)
    frame_lookup = {int(frame_idx): index for index, frame_idx in enumerate(frame_ids)}
    for row in table.itertuples(index=False):
        array_idx = frame_lookup.get(int(row.frame_idx))
        joint_idx = int(row.joint_idx)
        row_valid = bool(getattr(row, "valid", True))
        if (
            array_idx is not None
            and 0 <= joint_idx < joint_count
            and row_valid
        ):
            points[array_idx, joint_idx] = [float(row.x), float(row.y)]
    body_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
    features = [points[:, body_indices]]
    for start, end in ((91, 112), (112, 133)):
        hand = points[:, start:end]
        finite = np.all(np.isfinite(hand), axis=-1)
        hand_center = np.full((points.shape[0], 2), np.nan, dtype=float)
        for frame_idx in range(points.shape[0]):
            if np.count_nonzero(finite[frame_idx]) >= 3:
                hand_center[frame_idx] = np.median(
                    hand[frame_idx, finite[frame_idx]],
                    axis=0,
                )
        features.append(hand_center[:, None, :])
    return np.concatenate(features, axis=1)


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    finite = result[np.isfinite(result)]
    if finite.size == 0:
        return result
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if scale <= 1e-8:
        scale = float(np.std(finite))
    if scale <= 1e-8:
        scale = 1.0
    return (result - center) / scale


def _project(points_source: np.ndarray, calibration) -> np.ndarray:
    output = np.full((*points_source.shape[:2], 2), np.nan, dtype=float)
    for frame_idx, frame_points in enumerate(points_source):
        valid = np.all(np.isfinite(frame_points), axis=1)
        if not np.any(valid):
            continue
        projected, _ = cv2.projectPoints(
            frame_points[valid].reshape(-1, 1, 3),
            calibration.rotation_vector.reshape(3, 1),
            calibration.translation_vector.reshape(3, 1),
            calibration.intrinsic_matrix,
            calibration.distortion_coefficients.reshape(-1),
        )
        output[frame_idx, valid] = projected.reshape(-1, 2)
    return output


def _affine_alignment(
    projected: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    if np.count_nonzero(valid) < 6:
        return None, np.empty(0, dtype=float)
    affine, _ = cv2.estimateAffinePartial2D(
        projected[valid].astype(np.float32),
        observed[valid].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=15.0,
        maxIters=5000,
    )
    if affine is None:
        return None, np.empty(0, dtype=float)
    predicted = cv2.transform(
        projected.reshape(-1, 1, 2).astype(np.float32),
        affine,
    ).reshape(projected.shape)
    error = np.linalg.norm(observed - predicted, axis=-1)
    return affine, error[valid & np.isfinite(error)]


def _temporal_shift_scan(
    observed: np.ndarray,
    projected: np.ndarray,
    max_shift: int,
) -> list[dict[str, float | int]]:
    joint_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    output = []
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            shifted_observed = observed[shift:]
            shifted_projected = projected[: observed.shape[0] - shift]
        else:
            shifted_observed = observed[: observed.shape[0] + shift]
            shifted_projected = projected[-shift:]
        valid = np.all(np.isfinite(shifted_observed), axis=-1) & np.all(
            np.isfinite(shifted_projected),
            axis=-1,
        )
        error = np.linalg.norm(
            shifted_observed[:, joint_indices] - shifted_projected[:, joint_indices],
            axis=-1,
        )
        values = error[valid[:, joint_indices]]
        if values.size:
            output.append(
                {
                    "shift_frames": shift,
                    "median_error_px": float(np.median(values)),
                    "p95_error_px": float(np.percentile(values, 95.0)),
                }
            )
    return output


def _temporal_shift_windows(
    observed: np.ndarray,
    projected: np.ndarray,
    shift: int,
    window_size: int = 60,
) -> list[dict[str, float | int | None]]:
    joint_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    if shift >= 0:
        aligned_observed = observed[shift:]
        aligned_projected = projected[: observed.shape[0] - shift]
        projected_start = 0
    else:
        aligned_observed = observed[: observed.shape[0] + shift]
        aligned_projected = projected[-shift:]
        projected_start = -shift
    output = []
    for start in range(0, aligned_observed.shape[0], max(window_size, 1)):
        end = min(start + max(window_size, 1), aligned_observed.shape[0])
        observed_window = aligned_observed[start:end, joint_indices]
        projected_window = aligned_projected[start:end, joint_indices]
        valid = np.all(np.isfinite(observed_window), axis=-1) & np.all(
            np.isfinite(projected_window),
            axis=-1,
        )
        error = np.linalg.norm(observed_window - projected_window, axis=-1)
        values = error[valid]
        output.append(
            {
                "projected_start_frame": int(projected_start + start),
                "projected_end_frame_exclusive": int(projected_start + end),
                "median_error_px": (
                    float(np.median(values)) if values.size else None
                ),
                "p95_error_px": (
                    float(np.percentile(values, 95.0)) if values.size else None
                ),
            }
        )
    return output


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


if __name__ == "__main__":
    main()
