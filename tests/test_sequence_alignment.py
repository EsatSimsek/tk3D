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
