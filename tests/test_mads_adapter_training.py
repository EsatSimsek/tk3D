from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_mads_vitpose_offsets import coordinate_metrics, robust_joint_offsets
from scripts.train_mads_vitpose_adapter import heatmap_targets, split_sequences
from src.mads_dataset import MadsSequence


def _sequence(action: str, sequence: str) -> MadsSequence:
    return MadsSequence(
        modality="multiview",
        action=action,
        sequence=sequence,
        videos={},
        ground_truth_path=Path(f"{action}_{sequence}_GT.mat"),
        auxiliary_paths=[],
    )


def test_adapter_split_never_leaks_held_out_test_sequence() -> None:
    sequences = [
        _sequence("Kata", "F2"),
        _sequence("Kata", "F3"),
        _sequence("Kata", "F4"),
        _sequence("Taichi", "S6"),
    ]

    train, validation = split_sequences(
        sequences,
        actions=["Kata", "Taichi"],
        test_labels=["Kata:F2"],
        validation_labels=["Kata:F3", "Taichi:S6"],
    )

    assert [(item.action, item.sequence) for item in train] == [("Kata", "F4")]
    assert {(item.action, item.sequence) for item in validation} == {
        ("Kata", "F3"),
        ("Taichi", "S6"),
    }


def test_adapter_split_rejects_test_validation_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        split_sequences(
            [_sequence("Kata", "F2")],
            actions=["Kata"],
            test_labels=["Kata:F2"],
            validation_labels=["Kata:F2"],
        )


def test_projected_mads_points_generate_supervised_gaussian_heatmaps() -> None:
    projected = np.zeros((15, 2), dtype=float)
    projected[:, 0] = 50.0
    projected[:, 1] = 60.0

    targets, weights = heatmap_targets(
        projected,
        crop=(0.0, 0.0, 100.0, 120.0),
        heatmap_size=(48, 64),
        sigma=2.0,
    )

    assert targets.shape == (12, 64, 48)
    assert weights.shape == (12,)
    assert np.all(weights == 1.0)
    assert np.all(np.max(targets, axis=(1, 2)) > 0.90)


def test_robust_offsets_improve_held_out_coordinate_bias_without_exceeding_limit() -> None:
    predicted = np.zeros((20, 2, 2), dtype=float)
    truth = np.zeros_like(predicted)
    truth[:, 0] = [0.5, -1.0]
    truth[:, 1] = [4.0, 0.0]

    offsets = robust_joint_offsets(predicted, truth, max_offset_heatmap_px=2.0)
    metrics = coordinate_metrics(predicted, truth, offsets)

    np.testing.assert_allclose(offsets[0], [0.5, -1.0])
    assert np.linalg.norm(offsets[1]) == pytest.approx(2.0)
    assert metrics["calibrated_mean_px"] < metrics["base_mean_px"]
