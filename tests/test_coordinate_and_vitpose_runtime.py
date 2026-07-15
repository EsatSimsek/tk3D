from __future__ import annotations

import numpy as np

from src.coordinate_system import aist_world_to_analysis, opencv_reference_to_analysis, transform_points
from src.vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer, _aspect_correct_bbox


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
