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
