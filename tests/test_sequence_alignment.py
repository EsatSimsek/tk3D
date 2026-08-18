"""sequence_alignment için testler (Aşama 4, Yol 1)."""
from src.poomsae_scoring.sequence_alignment import align_segments_to_movements


def _seg(segment_id, start, end):
    return {"segment_id": segment_id, "start_frame": start, "end_frame": end}


def test_equal_counts_align_in_order():
    segments = [_seg(0, 140, 229), _seg(1, 230, 353), _seg(2, 354, 420)]
    result = align_segments_to_movements(segments, ["M01", "M02", "M03"])
    assert result["status"] == "aligned"
    assert result["anomalies"] == []
    assert [a["movement_id"] for a in result["aligned"]] == ["M01", "M02", "M03"]
    assert result["aligned"][0]["source_segment_id"] == 0


def test_missing_segment_is_reported_not_swallowed():
    segments = [_seg(0, 140, 229), _seg(1, 230, 353)]
    result = align_segments_to_movements(segments, ["M01", "M02", "M03"])
    assert result["status"] == "missing_segments"
    assert len(result["aligned"]) == 2
    assert result["anomalies"] == [{"issue": "unmatched_movement", "movement_id": "M03"}]


def test_extra_segment_is_reported_not_crash():
    segments = [_seg(0, 140, 229), _seg(1, 230, 353), _seg(2, 354, 420), _seg(3, 421, 500)]
    result = align_segments_to_movements(segments, ["M01", "M02", "M03"])
    assert result["status"] == "extra_segments"
    assert len(result["aligned"]) == 3
    assert result["anomalies"] == [{"issue": "unmatched_segment", "segment_id": 3}]


def test_dtw_detects_segment_spanning_multiple_movements():
    from src.poomsae_scoring.sequence_alignment import (
        _cost_matrix,
        _dtw_align,
        _detect_anomalies_from_pairs,
    )
    # 2 segment (baş ve son), 3 hareket bekleniyor: ortadaki M02 için segment yok
    segments = [_seg(0, 0, 100), _seg(1, 400, 500)]
    cost = _cost_matrix(segments, 3, 500)
    result = _dtw_align(cost)
    anomalies = _detect_anomalies_from_pairs(result["pairs"], 2, ["M01", "M02", "M03"])
    # Segment 0'ın M01+M02'ye yayıldığı tespit edilmeli
    spanning = [a for a in anomalies if a["issue"] == "segment_spans_multiple_movements"]
    assert len(spanning) == 1
    assert spanning[0]["segment_id"] == 0
    assert spanning[0]["movement_ids"] == ["M01", "M02"]


def test_dtw_clean_case_has_no_anomalies():
    from src.poomsae_scoring.sequence_alignment import (
        _cost_matrix,
        _dtw_align,
        _detect_anomalies_from_pairs,
    )
    # 3 segment, 3 hareket, temiz eşleşme
    segments = [_seg(0, 0, 100), _seg(1, 200, 300), _seg(2, 400, 500)]
    cost = _cost_matrix(segments, 3, 500)
    result = _dtw_align(cost)
    anomalies = _detect_anomalies_from_pairs(result["pairs"], 3, ["M01", "M02", "M03"])
    assert anomalies == []
    assert result["pairs"] == [(0, 0), (1, 1), (2, 2)]


def test_pose_dtw_skips_truly_missing_movement():
    from src.synthetic_poses import build_frame
    from src.poomsae_scoring.sequence_alignment import (
        _pose_cost_matrix, _dtw_align_with_skip, _auto_skip_penalty,
    )
    expected = [build_frame(90, 90), build_frame(180, 180), build_frame(90, 90)]
    penalty = _auto_skip_penalty(expected)
    # Ap-seogi hic yapilmadi: iki juchum segmenti
    segments = [build_frame(90, 90), build_frame(90, 90)]
    cost = _pose_cost_matrix(segments, expected)
    result = _dtw_align_with_skip(cost, skip_penalty=penalty)
    assert result["skipped_movements"] == [1]
    assert result["pairs"] == [(0, 0), (1, 2)]


def test_pose_dtw_keeps_imperfect_but_present_movement():
    from src.synthetic_poses import build_frame
    from src.poomsae_scoring.sequence_alignment import (
        _pose_cost_matrix, _dtw_align_with_skip, _auto_skip_penalty,
    )
    expected = [build_frame(90, 90), build_frame(180, 180), build_frame(90, 90)]
    penalty = _auto_skip_penalty(expected)
    # Ap-seogi yapildi ama kusurlu (175 derece)
    segments = [build_frame(90, 90), build_frame(175, 175), build_frame(90, 90)]
    cost = _pose_cost_matrix(segments, expected)
    result = _dtw_align_with_skip(cost, skip_penalty=penalty)
    assert result["skipped_movements"] == []
    assert result["pairs"] == [(0, 0), (1, 1), (2, 2)]


def test_uncertain_band_flags_borderline_kept_movement():
    from src.synthetic_poses import build_frame
    from src.poomsae_scoring.sequence_alignment import (
        _pose_cost_matrix, _dtw_align_with_skip, _auto_skip_penalty, _flag_uncertain_movements,
    )
    expected = [build_frame(90, 90), build_frame(180, 180), build_frame(90, 90)]
    penalty = _auto_skip_penalty(expected)
    margin = 0.03
    # Ortadaki hareket sinirda (125 derece): tutulmali ama belirsiz isaretlenmeli
    segments = [build_frame(90, 90), build_frame(125, 125), build_frame(90, 90)]
    cost = _pose_cost_matrix(segments, expected)
    result = _dtw_align_with_skip(cost, skip_penalty=penalty)
    assert result["skipped_movements"] == []  # sporcu lehine tutuldu
    flagged = _flag_uncertain_movements(cost, result["pairs"], penalty, margin)
    assert any(f["movement_index"] == 1 for f in flagged)  # belirsiz isaretlendi


def test_clear_movement_not_flagged_uncertain():
    from src.synthetic_poses import build_frame
    from src.poomsae_scoring.sequence_alignment import (
        _pose_cost_matrix, _dtw_align_with_skip, _auto_skip_penalty, _flag_uncertain_movements,
    )
    expected = [build_frame(90, 90), build_frame(180, 180), build_frame(90, 90)]
    penalty = _auto_skip_penalty(expected)
    margin = 0.03
    # Ortadaki hareket iyi (175 derece): tutulmali, belirsiz OLMAMALI
    segments = [build_frame(90, 90), build_frame(175, 175), build_frame(90, 90)]
    cost = _pose_cost_matrix(segments, expected)
    result = _dtw_align_with_skip(cost, skip_penalty=penalty)
    flagged = _flag_uncertain_movements(cost, result["pairs"], penalty, margin)
    assert not any(f["movement_index"] == 1 for f in flagged)  # temiz, isaret yok


def _synthetic_spec(movement_count):
    """Minimal PoomsaeSpec-like dict with `movement_count` movements."""
    return {
        "schema_version": 1,
        "poomsae_id": "synthetic_poomsae",
        "version": "0.0.1-test",
        "status": "draft",
        "display_name": "Synthetic",
        "rule_pack_id": "wt_recognized_2024-09-30",
        "sequence_status": "source_transcribed",
        "source_documents": [
            {
                "source_id": "synthetic_source",
                "authority": "Test",
                "title": "Synthetic",
                "url": "https://example.invalid/synthetic",
                "effective_date": "2026-01-01",
                "accessed_at": "2026-01-01",
                "language": "test",
                "access": "public",
                "content_sha256": None,
                "sections": ["synthetic"],
            }
        ],
        "movements": [
            {
                "movement_id": f"M{index+1:02d}",
                "sequence_index": index + 1,
                "display_name": f"Synthetic movement {index+1}",
                "direction": "initial_forward",
                "stance": "left_ap_seogi",
                "techniques": [
                    {"technique_id": "arae_makki", "side": "left", "order": 1}
                ],
                "phases": ["preparation", "execution", "fixation"],
                "measurable_criteria": ["balance.torso_vertical"],
                "source_refs": ["synthetic_source#sec1"],
            }
            for index in range(movement_count)
        ],
        "blocked_reasons": ["synthetic spec used for tests only"],
    }


def test_build_automatic_movement_timeline_maps_segments_and_reports_missing():
    from src.synthetic_poses import build_frame
    from src.poomsae_scoring.sequence_alignment import build_automatic_movement_timeline

    spec = _synthetic_spec(5)
    expected = [
        build_frame(160, 160),
        build_frame(120, 120),
        build_frame(150, 150),
        build_frame(90, 90),
        build_frame(85, 85),
    ]
    segments = [
        {"segment_id": 0, "start_frame": 10, "end_frame": 60, "mean_pose": build_frame(158, 158)},
        {"segment_id": 1, "start_frame": 80, "end_frame": 140, "mean_pose": build_frame(122, 122)},
        {"segment_id": 2, "start_frame": 160, "end_frame": 220, "mean_pose": build_frame(148, 148)},
    ]

    timeline = build_automatic_movement_timeline(
        segments=segments,
        expected_poses=expected,
        poomsae_spec=spec,
        frame_count=300,
        fps=60.0,
        source_binding={
            "session_id": "auto-session",
            "run_id": "auto-run",
            "pose_file": "outputs/auto/pose.json",
            "pose_file_sha256": None,
        },
        timeline_id="auto-align-test",
    )

    assert timeline["label_source"] == "automatic"
    assert timeline["coverage"]["recording_scope"] == "partial_sequence"
    assert timeline["coverage"]["observed_movement_ids"] == ["M01", "M02", "M03"]
    assert timeline["coverage"]["missing_movement_ids"] == ["M04", "M05"]
    assert len(timeline["segments"]) == 3
    first = timeline["segments"][0]
    assert first["movement_id"] == "M01"
    assert first["start_frame"] == 10
    assert first["end_frame"] == 60
    anchors = first["anchors"]
    assert list(anchors) == ["preparation", "execution", "fixation"]
    assert 10 <= anchors["preparation"] <= anchors["execution"] <= anchors["fixation"] <= 60
    assert first["label_status"] in {"confirmed", "ambiguous"}
    assert 0.5 <= first["confidence"] <= 0.99


def test_build_automatic_movement_timeline_rejects_overlapping_segments():
    from src.poomsae_scoring.contracts import ScoringContractError
    from src.poomsae_scoring.sequence_alignment import build_automatic_movement_timeline
    from src.synthetic_poses import build_frame
    import pytest

    spec = _synthetic_spec(2)
    expected = [build_frame(160, 160), build_frame(120, 120)]
    bad_segments = [
        {"segment_id": 0, "start_frame": 10, "end_frame": 60, "mean_pose": build_frame(158, 158)},
        {"segment_id": 1, "start_frame": 60, "end_frame": 100, "mean_pose": build_frame(122, 122)},
    ]
    with pytest.raises(ScoringContractError, match="segments must not overlap"):
        build_automatic_movement_timeline(
            segments=bad_segments,
            expected_poses=expected,
            poomsae_spec=spec,
            frame_count=200,
            fps=60.0,
            source_binding={
                "session_id": "auto-session",
                "run_id": "auto-run",
                "pose_file": "outputs/auto/pose.json",
                "pose_file_sha256": None,
            },
            timeline_id="auto-align-bad",
        )


def _auto_timeline_kwargs(spec, segments, expected, frame_count=200):
    """Shared valid kwargs for build_automatic_movement_timeline edge-case tests."""
    return {
        "segments": segments,
        "expected_poses": expected,
        "poomsae_spec": spec,
        "frame_count": frame_count,
        "fps": 60.0,
        "source_binding": {
            "session_id": "auto-session",
            "run_id": "auto-run",
            "pose_file": "outputs/auto/pose.json",
            "pose_file_sha256": None,
        },
        "timeline_id": "auto-align-edge",
    }


def test_build_automatic_movement_timeline_validates_inputs_fail_closed():
    import pytest

    from src.poomsae_scoring.contracts import ScoringContractError
    from src.poomsae_scoring.sequence_alignment import build_automatic_movement_timeline
    from src.synthetic_poses import build_frame

    spec = _synthetic_spec(2)
    expected = [build_frame(160, 160), build_frame(120, 120)]
    good_segment = {
        "segment_id": 0,
        "start_frame": 10,
        "end_frame": 60,
        "mean_pose": build_frame(158, 158),
    }

    with pytest.raises(ScoringContractError, match="expected_poses length"):
        build_automatic_movement_timeline(
            **_auto_timeline_kwargs(spec, [good_segment], expected[:1])
        )
    with pytest.raises(ScoringContractError, match="non-empty list"):
        build_automatic_movement_timeline(**_auto_timeline_kwargs(spec, [], expected))
    incomplete = {"segment_id": 0, "start_frame": 10, "end_frame": 60}
    with pytest.raises(ScoringContractError, match="missing required key"):
        build_automatic_movement_timeline(**_auto_timeline_kwargs(spec, [incomplete], expected))
    inverted = dict(good_segment, start_frame=60, end_frame=10)
    with pytest.raises(ScoringContractError, match="cannot precede"):
        build_automatic_movement_timeline(**_auto_timeline_kwargs(spec, [inverted], expected))
    beyond = dict(good_segment, end_frame=205)
    with pytest.raises(ScoringContractError, match="exceeds frame_count"):
        build_automatic_movement_timeline(**_auto_timeline_kwargs(spec, [beyond], expected))


def test_build_automatic_movement_timeline_keeps_best_movement_per_shared_segment():
    from src.poomsae_scoring.sequence_alignment import (
        build_automatic_movement_timeline,
        pose_distance,
    )
    from src.synthetic_poses import build_frame

    spec = _synthetic_spec(2)
    expected = [build_frame(160, 160), build_frame(90, 90)]
    # Mean pose sits between both expected poses (biased toward M01) so DTW may
    # pair both movements onto the single segment; the timeline contract forces
    # exactly one winner. The winner must be the prefix movement M01 — the
    # coverage contract only accepts observed ids that form a spec prefix.
    shared_pose = build_frame(135, 135)
    segments = [
        {"segment_id": 0, "start_frame": 10, "end_frame": 60, "mean_pose": shared_pose}
    ]

    costs = [pose_distance(shared_pose, pose) for pose in expected]
    assert costs[0] < costs[1]  # geometry sanity: M01 really is the closer match

    timeline = build_automatic_movement_timeline(
        **_auto_timeline_kwargs(spec, segments, expected)
    )

    assert len(timeline["segments"]) == 1
    assert timeline["segments"][0]["movement_id"] == "M01"
    assert timeline["coverage"]["observed_movement_ids"] == ["M01"]
    assert timeline["coverage"]["missing_movement_ids"] == ["M02"]
    assert timeline["coverage"]["recording_scope"] == "partial_sequence"
    assert (
        timeline["coverage"]["source_end_reason"]
        == "auto_alignment_missing_movements_detected"
    )


def test_build_automatic_movement_timeline_survives_segment_shorter_than_phase_count():
    from src.poomsae_scoring.sequence_alignment import build_automatic_movement_timeline
    from src.synthetic_poses import build_frame

    spec = _synthetic_spec(1)  # synthetic movements carry 3 phases each
    expected = [build_frame(150, 150)]
    segments = [
        {"segment_id": 0, "start_frame": 10, "end_frame": 11, "mean_pose": build_frame(150, 150)}
    ]

    timeline = build_automatic_movement_timeline(
        **_auto_timeline_kwargs(spec, segments, expected)
    )

    anchors = timeline["segments"][0]["anchors"]
    values = list(anchors.values())
    assert list(anchors) == ["preparation", "execution", "fixation"]
    assert all(10 <= value <= 11 for value in values)
    assert values == sorted(values)  # non-decreasing keeps the timeline contract happy
    assert timeline["coverage"]["recording_scope"] == "complete_performance"
    # Contract: a complete performance must carry no end reason.
    assert timeline["coverage"]["source_end_reason"] is None


def test_distribute_anchors_edge_cases():
    import pytest

    from src.poomsae_scoring.contracts import ScoringContractError
    from src.poomsae_scoring.sequence_alignment import _distribute_anchors

    assert _distribute_anchors([], 5, 9) == {}
    assert _distribute_anchors(["only"], 5, 9) == {"only": 5}
    spread = _distribute_anchors(["a", "b", "c"], 0, 100)
    assert spread == {"a": 0, "b": 50, "c": 100}
    with pytest.raises(ScoringContractError, match="inverted segment"):
        _distribute_anchors(["a"], 9, 5)
