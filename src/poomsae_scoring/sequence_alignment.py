"""Segment list aligned to known Taegeuk movement order (Stage 4)."""
from typing import Any


def align_segments_to_movements(
    segments: list[dict[str, Any]],
    movement_ids: list[str],
) -> dict[str, Any]:
    """Her segmenti s�radaki harekete e�ler ve say� uyu�mazl���n� raporlar."""
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




def _cost_matrix(segments, movement_count, frame_count):
    """Her segment-hareket �ifti i�in konum fark� maliyetini hesaplar."""
    matrix = []
    for segment in segments:
        mid_frame = (segment["start_frame"] + segment["end_frame"]) / 2
        segment_pos = mid_frame / frame_count if frame_count > 0 else 0.0
        row = []
        for movement_index in range(movement_count):
            movement_pos = (movement_index + 0.5) / movement_count
            cost = abs(segment_pos - movement_pos)
            row.append(cost)
        matrix.append(row)
    return matrix


def _dtw_align(cost_matrix):
    """Maliyet tablosunda en d���k toplam maliyetli monoton yolu bulur."""
    n_segments = len(cost_matrix)
    n_movements = len(cost_matrix[0]) if n_segments else 0

    INF = float("inf")
    # Birikimli maliyet tablosu (kenarlarda sonsuz, ba�lang�� i�in +1 boyut)
    acc = [[INF] * (n_movements + 1) for _ in range(n_segments + 1)]
    acc[0][0] = 0.0

    for i in range(1, n_segments + 1):
        for j in range(1, n_movements + 1):
            cost = cost_matrix[i - 1][j - 1]
            best_prev = min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1])
            acc[i][j] = cost + best_prev

    # �z s�rme: sa�-alttan sol-�ste
    pairs = []
    i, j = n_segments, n_movements
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        diag = acc[i - 1][j - 1]
        up = acc[i - 1][j]
        left = acc[i][j - 1]
        best = min(diag, up, left)
        if best == diag:
            i, j = i - 1, j - 1
        elif best == up:
            i = i - 1
        else:
            j = j - 1

    pairs.reverse()
    total_cost = acc[n_segments][n_movements]
    return {"pairs": pairs, "total_cost": total_cost}


def _detect_anomalies_from_pairs(pairs, n_segments, movement_ids):
    """DTW e�le�melerinden atlanan hareketleri ve b�l�nen segmentleri bulur."""
    # Her hareket ka� segment ald�?
    movements_seen = {}
    for seg_idx, mov_idx in pairs:
        movements_seen.setdefault(mov_idx, []).append(seg_idx)
    # Her segment ka� harekete e�lendi?
    segments_used = {}
    for seg_idx, mov_idx in pairs:
        segments_used.setdefault(seg_idx, []).append(mov_idx)

    anomalies = []
    # Hi� segment almayan hareket = atlanm��
    for mov_idx in range(len(movement_ids)):
        if mov_idx not in movements_seen:
            anomalies.append({"issue": "skipped_movement", "movement_id": movement_ids[mov_idx]})
    # Birden �ok harekete e�lenen segment = b�l�nm��/��pheli
    for seg_idx, mov_list in segments_used.items():
        if len(mov_list) > 1:
            anomalies.append({
                "issue": "segment_spans_multiple_movements",
                "segment_id": seg_idx,
                "movement_ids": [movement_ids[m] for m in mov_list],
            })
    return anomalies


def _dtw_align_with_skip(cost_matrix, skip_penalty=0.5):
    """Atlama se�enekli DTW: bir hareketi e�lemek yerine ceza �deyip atlayabilir."""
    n_segments = len(cost_matrix)
    n_movements = len(cost_matrix[0]) if n_segments else 0

    INF = float("inf")
    acc = [[INF] * (n_movements + 1) for _ in range(n_segments + 1)]
    move = [[None] * (n_movements + 1) for _ in range(n_segments + 1)]
    acc[0][0] = 0.0

    # �lk sat�r: hi� segment yokken hareketleri atlayarak ilerle
    for j in range(1, n_movements + 1):
        acc[0][j] = acc[0][j - 1] + skip_penalty
        move[0][j] = "skip"

    for i in range(1, n_segments + 1):
        for j in range(1, n_movements + 1):
            cost = cost_matrix[i - 1][j - 1]
            match = min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1]) + cost
            skip = acc[i][j - 1] + skip_penalty
            if skip < match:
                acc[i][j] = skip
                move[i][j] = "skip"
            else:
                acc[i][j] = match
                move[i][j] = "match"

    # �z s�rme
    pairs = []
    skipped = []
    i, j = n_segments, n_movements
    while j > 0:
        if move[i][j] == "skip":
            skipped.append(j - 1)
            j = j - 1
        else:
            pairs.append((i - 1, j - 1))
            diag = acc[i - 1][j - 1]
            up = acc[i - 1][j]
            left = acc[i][j - 1]
            best = min(diag, up, left)
            if best == diag:
                i, j = i - 1, j - 1
            elif best == up:
                i = i - 1
            else:
                j = j - 1
        if i == 0:
            while j > 0:
                skipped.append(j - 1)
                j = j - 1

    pairs.reverse()
    skipped.reverse()
    return {"pairs": pairs, "skipped_movements": skipped, "total_cost": acc[n_segments][n_movements]}


def pose_distance(frame_a, frame_b):
    """�ki poz karesi aras�ndaki ortalama eklem uzakl��� (NaN eklemler atlan�r)."""
    import numpy as np
    a = np.asarray(frame_a, dtype=float)
    b = np.asarray(frame_b, dtype=float)
    # Her eklemin 3D uzakl���
    diff = a - b
    per_joint = np.sqrt(np.sum(diff * diff, axis=1))
    # �ki karede de dolu (NaN olmayan) eklemleri se�
    valid = np.isfinite(per_joint)
    if not np.any(valid):
        return float("inf")
    return float(np.mean(per_joint[valid]))


def _pose_cost_matrix(segment_poses, expected_poses):
    """Her segment-hareket �ifti i�in duru� fark� maliyeti (konum yerine poz)."""
    matrix = []
    for seg_pose in segment_poses:
        row = []
        for exp_pose in expected_poses:
            row.append(pose_distance(seg_pose, exp_pose))
        matrix.append(row)
    return matrix


def _auto_skip_penalty(expected_poses, fraction=0.7):
    """Ceza = beklenen duru�lar aras� en k���k fark�n yar�s� (veriden otomatik)."""
    min_gap = float("inf")
    for i in range(len(expected_poses)):
        for j in range(i + 1, len(expected_poses)):
            gap = pose_distance(expected_poses[i], expected_poses[j])
            if gap > 0 and gap < min_gap:
                min_gap = gap
    if min_gap == float("inf"):
        return 0.1  # tek hareket varsa varsayilan
    return min_gap * fraction



def _flag_uncertain_movements(cost_matrix, pairs, skip_penalty, uncertainty_margin):
    """Tutulan (atlanmayan) ama atlama esigine yakin hareketleri belirsiz isaretler."""
    uncertain = []
    for seg_idx, mov_idx in pairs:
        cost = cost_matrix[seg_idx][mov_idx]
        lower = skip_penalty - uncertainty_margin
        upper = skip_penalty + uncertainty_margin
        if lower <= cost <= upper:
            uncertain.append({
                "movement_index": mov_idx,
                "segment_index": seg_idx,
                "cost": round(cost, 4),
                "skip_threshold": round(skip_penalty, 4),
                "reason": "cost_within_uncertainty_band_below_skip_threshold",
                "decision": "kept_in_favor_of_athlete_flagged_for_review",
            })
    return uncertain
