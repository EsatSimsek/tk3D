from __future__ import annotations

import numpy as np

from src.person_tracking import (
    PersonDetectorConfig,
    _best_initial_track_index,
    _pad_bbox,
    _stabilize_bbox,
    person_detector_config_from_mapping,
)


def test_best_initial_track_prefers_confident_large_athlete_over_decoy() -> None:
    boxes = np.asarray(
        [
            [360.0, 100.0, 445.0, 365.0],
            [195.0, 140.0, 230.0, 260.0],
        ]
    )
    confidences = np.asarray([0.95, 0.56])

    selected = _best_initial_track_index(boxes, confidences, minimum_confidence=0.65)

    assert selected == 0


def test_initial_track_refuses_low_confidence_human_shaped_decoy() -> None:
    boxes = np.asarray([[195.0, 140.0, 230.0, 260.0]])
    confidences = np.asarray([0.56])

    assert _best_initial_track_index(boxes, confidences, minimum_confidence=0.65) is None


def test_detector_bbox_padding_is_clipped_to_frame() -> None:
    padded = _pad_bbox(
        np.asarray([0.0, 10.0, 100.0, 200.0]),
        image_width=512,
        image_height=384,
        padding=0.20,
    )

    np.testing.assert_allclose(padded, [0.0, 0.0, 120.0, 238.0])


def test_person_detector_mapping_uses_effective_sample_frame_rate() -> None:
    config = person_detector_config_from_mapping(
        {"enabled": True, "model_variant": "small"},
        frame_rate=15.0,
    )

    assert config == PersonDetectorConfig(enabled=True, model_variant="small", frame_rate=15)


def test_bbox_stabilizer_suppresses_small_detector_jitter() -> None:
    previous = np.asarray([100.0, 50.0, 200.0, 250.0])
    current = np.asarray([102.0, 49.0, 202.0, 249.0])

    stabilized = _stabilize_bbox(
        current,
        previous,
        stationary_alpha=0.35,
        motion_scale_ratio=0.12,
    )

    assert np.linalg.norm(stabilized - previous) < np.linalg.norm(current - previous)
    assert np.linalg.norm(stabilized - current) < np.linalg.norm(current - previous)


def test_bbox_stabilizer_follows_large_motion_without_lag() -> None:
    previous = np.asarray([100.0, 50.0, 200.0, 250.0])
    current = np.asarray([180.0, 50.0, 280.0, 250.0])

    stabilized = _stabilize_bbox(
        current,
        previous,
        stationary_alpha=0.35,
        motion_scale_ratio=0.12,
    )

    np.testing.assert_allclose(stabilized, current)
