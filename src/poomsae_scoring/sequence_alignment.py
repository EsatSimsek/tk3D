"""Segment listesini bilinen Taegeuk hareket sırasına hizalar (Aşama 4, Yol 1)."""
from typing import Any


def align_segments_to_movements(
    segments: list[dict[str, Any]],
    movement_ids: list[str],
) -> dict[str, Any]:
    """Her segmenti sıradaki harekete eşler ve sayı uyuşmazlığını raporlar."""
    segment_count = len(segments)
    movement_count = len(movement_ids)
    pair_count = min(segment_count, movement_count)

    aligned = []
    for index in range(pair_count):
        segment = segments[index]
        aligned.append(
            {
                "sequence_index": index + 1,
                "movement_id": movement_ids[index],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "source_segment_id": segment["segment_id"],
            }
        )

    if segment_count == movement_count:
        status = "aligned"
        anomalies = []
    elif segment_count < movement_count:
        status = "missing_segments"
        anomalies = [
            {"issue": "unmatched_movement", "movement_id": movement_ids[i]}
            for i in range(pair_count, movement_count)
        ]
    else:
        status = "extra_segments"
        anomalies = [
            {"issue": "unmatched_segment", "segment_id": segments[i]["segment_id"]}
            for i in range(pair_count, segment_count)
        ]

    return {
        "status": status,
        "segment_count": segment_count,
        "movement_count": movement_count,
        "aligned": aligned,
        "anomalies": anomalies,
    }


