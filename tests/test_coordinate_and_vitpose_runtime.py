from __future__ import annotations

import numpy as np

from src.coordinate_system import aist_world_to_analysis, opencv_reference_to_analysis, transform_points
from src.pose2d_estimator import _bbox_from_pose, pose2d_from_arrays
from src.vitpose_plus_runtime import (
    ViTPosePlusWholeBodyInferencer,
    _aspect_correct_bbox,
    _refine_heatmap_peaks_udp,
)


def test_coordinate_transforms_produce_meter_z_up_analysis_space() -> None:
    opencv = np.array([[[1.0, 2.0, 3.0]]])
    np.testing.assert_allclose(transform_points(opencv, opencv_reference_to_analysis()), [[[1.0, 3.0, -2.0]]])

    aist_cm = np.array([[[100.0, 200.0, 300.0]]])
    np.testing.assert_allclose(transform_points(aist_cm, aist_world_to_analysis()), [[[1.0, 3.0, 2.0]]])


def test_initial_vitpose_crop_preserves_model_aspect_ratio() -> None:
    bbox = _aspect_correct_bbox(None, image_width=1920, image_height=1080, target_aspect=192 / 256)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    assert abs(width / height - 192 / 256) < 1e-9
    assert abs((bbox[0] + bbox[2]) / 2 - 960) < 1e-9

    portrait = _aspect_correct_bbox(None, image_width=400, image_height=1000, target_aspect=192 / 256)
    portrait_width = portrait[2] - portrait[0]
    portrait_height = portrait[3] - portrait[1]
    assert abs(portrait_width / portrait_height - 192 / 256) < 1e-9


def test_zero_heatmap_response_is_zero_confidence_not_half() -> None:
    runtime = object.__new__(ViTPosePlusWholeBodyInferencer)
    runtime.input_width = 192
    runtime.input_height = 256
    heatmaps = np.zeros((133, 64, 48), dtype=float)

    _, scores = runtime._decode_heatmaps(heatmaps, (0.0, 0.0, 192.0, 256.0))

    assert np.all(scores == 0.0)


def test_udp_peak_refinement_recovers_subpixel_gaussian_center() -> None:
    height, width = 32, 24
    target = np.asarray([10.35, 18.60])
    yy, xx = np.mgrid[:height, :width]
    heatmap = np.exp(-((xx - target[0]) ** 2 + (yy - target[1]) ** 2) / (2.0 * 2.0**2))
    peak = np.asarray([[float(np.argmax(heatmap) % width), float(np.argmax(heatmap) // width)]])

    refined = _refine_heatmap_peaks_udp(heatmap[None, ...], peak, kernel_size=11)

    np.testing.assert_allclose(refined[0], target, atol=0.08)


def test_tracked_pose_bbox_preserves_fast_motion_margin() -> None:
    keypoints = np.zeros((133, 2), dtype=float)
    scores = np.zeros(133, dtype=float)
    keypoints[5:17, 0] = np.linspace(100.0, 200.0, 12)
    keypoints[5:17, 1] = np.linspace(50.0, 250.0, 12)
    scores[5:17] = 1.0
    pose = pose2d_from_arrays("C0", 0, keypoints, scores, score_threshold=0.3)

    bbox = _bbox_from_pose(pose, image_width=512, image_height=384)

    assert bbox is not None
    np.testing.assert_allclose(bbox, [82.5, 15.0, 217.5, 285.0])


def test_heatmap_offsets_are_applied_in_heatmap_coordinates() -> None:
    runtime = ViTPosePlusWholeBodyInferencer.__new__(ViTPosePlusWholeBodyInferencer)
    runtime.heatmap_offsets_xy = np.tile([1.0, -0.5], (133, 1))
    heatmaps = np.zeros((133, 64, 48), dtype=float)
    heatmaps[:, 30, 20] = 1.0

    shifted, scores = runtime._decode_heatmaps(heatmaps, (0.0, 0.0, 94.0, 126.0))
    runtime.heatmap_offsets_xy = np.zeros((133, 2), dtype=float)
    baseline, _ = runtime._decode_heatmaps(heatmaps, (0.0, 0.0, 94.0, 126.0))

    np.testing.assert_allclose(
        shifted - baseline,
        np.tile([2.0, -1.0], (133, 1)),
        atol=1e-6,
    )
    assert np.all(scores == 1.0)
