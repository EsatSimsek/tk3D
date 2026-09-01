"""Segment list aligned to known Taegeuk movement order (Stage 4)."""
from typing import Any

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)


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


def normalize_pose_for_matching(frame):
    """Put a pose in a place- and size-free frame so it can be matched across recordings.

    A template is only useful if it still matches the same movement in a *different*
    recording. Raw world coordinates cannot do that: if the athlete starts a metre to
    the left, every joint is a metre away and the correct movement looks as wrong as
    the others. Two normalisations remove that without touching what actually
    distinguishes the movements:

    * **Place.** The pose is centred on the pelvis midpoint when both hips are
      measured, and on the mean of the measured joints otherwise. Where the athlete
      stood stops mattering.
    * **Size.** Coordinates are divided by the root-mean-square distance of the
      measured joints from that centre. Body size and how far the athlete happened to
      step stop mattering.

    Facing is deliberately *not* normalised. Which way the athlete turns is what
    separates one movement of the form from another, so removing it would make the
    templates match each other. Recordings whose world axes differ must instead be
    brought into a common frame with the session's athlete-local direction reference.

    Joints that were not measured stay NaN and are ignored downstream.
    """
    import numpy as np

    points = np.asarray(frame, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ScoringContractError("a pose frame must be shaped [joints, 3]")
    measured = np.all(np.isfinite(points), axis=1)
    if not np.any(measured):
        return points
    hips = [index for index in (11, 12) if index < len(points) and measured[index]]
    centre = points[hips].mean(axis=0) if len(hips) == 2 else points[measured].mean(axis=0)
    centred = points - centre
    scale = float(np.sqrt(np.mean(np.sum(centred[measured] ** 2, axis=1))))
    return centred if not np.isfinite(scale) or scale <= 1e-9 else centred / scale


def pose_distance(frame_a, frame_b):
    """Mean joint distance between two poses, place- and size-free (NaN joints skipped).

    Both poses go through :func:`normalize_pose_for_matching` first, so the result is
    a shape difference rather than a difference in where the athlete stood or how big
    they are. The value is in normalised units, not metres.
    """
    import numpy as np
    a = normalize_pose_for_matching(frame_a)
    b = normalize_pose_for_matching(frame_b)
    if a.shape != b.shape:
        raise ScoringContractError("pose frames must have the same joint count")
    # Her eklemin 3B uzaklığı
    diff = a - b
    per_joint = np.sqrt(np.sum(diff * diff, axis=1))
    # İki karede de dolu (NaN olmayan) eklemleri seç
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
    """Tutulan (atlanmayan) ama atlama esigine yakin hareketleri belirsiz isaretler.

    ``uncertainty_margin`` esigin bir orani olarak okunur; poz maliyetleri
    normalize edildigi icin mutlak bir bant olcek degistiginde anlamini yitirir.
    """
    uncertain = []
    band = abs(skip_penalty) * uncertainty_margin
    for seg_idx, mov_idx in pairs:
        cost = cost_matrix[seg_idx][mov_idx]
        lower = skip_penalty - band
        upper = skip_penalty + band
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


def build_automatic_movement_timeline(
    segments: list[dict[str, Any]],
    expected_poses: list[Any],
    poomsae_spec: dict[str, Any],
    *,
    frame_count: int,
    fps: float,
    source_binding: dict[str, Any],
    timeline_id: str,
    uncertainty_margin: float = 0.25,
) -> dict[str, Any]:
    """Return only the validated MovementTimeline; see :func:`build_automatic_timeline_report`.

    Kept as the narrow entry point for callers that want the timeline alone. Anything
    that should surface *how well* the alignment went must use
    :func:`build_automatic_timeline_report` instead, because the anomalies are
    discarded here.
    """
    return build_automatic_timeline_report(
        segments,
        expected_poses,
        poomsae_spec,
        frame_count=frame_count,
        fps=fps,
        source_binding=source_binding,
        timeline_id=timeline_id,
        uncertainty_margin=uncertainty_margin,
    )["timeline"]


def build_automatic_timeline_report(
    segments: list[dict[str, Any]],
    expected_poses: list[Any],
    poomsae_spec: dict[str, Any],
    *,
    frame_count: int,
    fps: float,
    source_binding: dict[str, Any],
    timeline_id: str,
    uncertainty_margin: float = 0.25,
) -> dict[str, Any]:
    """Auto-align detected pose segments to a PoomsaeSpec and return a validated MovementTimeline.

    This is the glue that lets the accuracy pipeline consume a recording without
    a hand-labelled timeline: given the per-segment mean poses and the expected
    reference pose of every spec movement, it runs the pose-based DTW with an
    automatic skip penalty, marks borderline matches as ambiguous, and emits a
    ``label_source=automatic`` MovementTimeline that passes the same contract
    checks as a manual one.

    Parameters
    ----------
    segments
        Each item must carry ``segment_id``, ``start_frame``, ``end_frame`` and
        ``mean_pose`` (a ``[133, 3]`` array-like usable by :func:`pose_distance`).
        Segments must be sorted by ``start_frame`` and must not overlap.
    expected_poses
        Reference pose (one per spec movement, same length and order as
        ``poomsae_spec['movements']``). Missing joints must be represented as
        NaN so :func:`pose_distance` can ignore them.
    frame_count, fps, source_binding, timeline_id
        Straight passthroughs to the MovementTimeline schema.
    uncertainty_margin
        Fraction of the auto skip penalty that still counts as ambiguous
        (matched movements land as ``label_status='ambiguous'`` instead of
        ``'provisional'``).

    Returns
    -------
    dict
        ``{"timeline": ..., "alignment_anomalies": [...]}``. The timeline has already
        been through :func:`validate_movement_timeline` and is ready to feed
        ``build_source_bound_accuracy_decisions`` and the rest of the pipeline.

        ``alignment_anomalies`` records every way the match was *not* clean. These are
        review candidates, never deductions: a movement without a segment may mean the
        athlete skipped it, but it may equally mean the segment detector merged two
        movements, the recording is partial, or the pose was too poor to match. The
        timeline itself cannot carry them — its schema is closed by
        ``_require_exact_keys`` — so they are returned alongside it and it is the
        caller's job to surface them.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    movements = spec["movements"]
    if len(expected_poses) != len(movements):
        raise ScoringContractError(
            "expected_poses length must equal the number of PoomsaeSpec movements"
        )
    if not isinstance(segments, list) or not segments:
        raise ScoringContractError("segments must be a non-empty list")
    previous_end = -1
    for index, segment in enumerate(segments):
        for key in ("segment_id", "start_frame", "end_frame", "mean_pose"):
            if key not in segment:
                raise ScoringContractError(f"segment {index} missing required key {key!r}")
        if segment["start_frame"] <= previous_end:
            raise ScoringContractError("segments must not overlap or move backward")
        if segment["end_frame"] < segment["start_frame"]:
            raise ScoringContractError("segment end_frame cannot precede start_frame")
        if segment["end_frame"] >= frame_count:
            raise ScoringContractError("segment end_frame exceeds frame_count")
        previous_end = segment["end_frame"]

    segment_poses = [segment["mean_pose"] for segment in segments]
    cost = _pose_cost_matrix(segment_poses, expected_poses)
    skip_penalty = _auto_skip_penalty(expected_poses)
    aligned = _dtw_align_with_skip(cost, skip_penalty=skip_penalty)
    uncertain = _flag_uncertain_movements(cost, aligned["pairs"], skip_penalty, uncertainty_margin)
    uncertain_movement_indices = {item["movement_index"] for item in uncertain}

    # Enforce 1-to-1: MovementTimeline forbids overlapping segments, so a source
    # segment can back at most one movement. When DTW pairs several movements to
    # the same segment (a common "spans multiple movements" case in partial
    # recordings), keep the movement with the lowest pose cost and treat the
    # rest as missing.
    best_by_segment: dict[int, tuple[int, float]] = {}
    pairs_by_segment: dict[int, list[int]] = {}
    for seg_idx, mov_idx in aligned["pairs"]:
        pair_cost = float(cost[seg_idx][mov_idx])
        pairs_by_segment.setdefault(seg_idx, []).append(mov_idx)
        current = best_by_segment.get(seg_idx)
        if current is None or pair_cost < current[1]:
            best_by_segment[seg_idx] = (mov_idx, pair_cost)
    candidate_pairs = [
        (seg_idx, mov_idx, pair_cost)
        for seg_idx, (mov_idx, pair_cost) in best_by_segment.items()
    ]
    best_by_movement: dict[int, tuple[int, float]] = {}
    for seg_idx, mov_idx, pair_cost in candidate_pairs:
        current = best_by_movement.get(mov_idx)
        if current is None or pair_cost < current[1]:
            best_by_movement[mov_idx] = (seg_idx, pair_cost)

    # The reverse direction matters just as much: a motion-energy detector routinely
    # splits one movement into several bursts, because the quiet fixation frames in the
    # middle fall below its threshold. Several segments then match the same movement,
    # and a timeline may not repeat a movement_id. Those fragments are merged back into
    # the span they jointly cover, which also restores the low-motion frames the
    # detector dropped -- exactly the frames the fixation measurements need.
    segments_by_movement: dict[int, list[int]] = {}
    for seg_idx, (mov_idx, _) in best_by_segment.items():
        segments_by_movement.setdefault(mov_idx, []).append(seg_idx)
    fragmented: dict[int, list[int]] = {
        mov_idx: sorted(seg_indices)
        for mov_idx, seg_indices in segments_by_movement.items()
        if len(seg_indices) > 1
    }
    matched_movement_indices = set(best_by_movement)
    missing_indices = {
        idx for idx in range(len(movements)) if idx not in matched_movement_indices
    }
    # One timeline segment per matched movement, in spec order so sequence_index is
    # contiguous and monotonic.
    ordered_movements = sorted(segments_by_movement)
    timeline_segments: list[dict[str, Any]] = []
    for order_index, mov_idx in enumerate(ordered_movements, start=1):
        movement = movements[mov_idx]
        seg_indices = sorted(segments_by_movement[mov_idx])
        start = min(int(segments[idx]["start_frame"]) for idx in seg_indices)
        end = max(int(segments[idx]["end_frame"]) for idx in seg_indices)
        anchors = _distribute_anchors(movement["phases"], start, end)
        # Confidence follows the best fragment; a merge cannot make a match stronger.
        pair_cost = best_by_movement[mov_idx][1]
        is_uncertain = mov_idx in uncertain_movement_indices
        if skip_penalty > 0:
            base = max(0.0, 1.0 - pair_cost / (2.0 * skip_penalty))
        else:
            base = 1.0
        confidence = round(min(max(base, 0.5 if not is_uncertain else 0.6), 0.99), 4)
        timeline_segments.append(
            {
                "sequence_index": order_index,
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": end,
                "anchors": anchors,
                "confidence": confidence,
                "label_status": "ambiguous" if is_uncertain else "provisional",
            }
        )

    observed_ids = [movements[mov_idx]["movement_id"] for mov_idx in ordered_movements]
    # Missing ids must be strictly ordered by sequence_index for the timeline
    # contract; they cover both skipped and out-of-recording movements.
    missing_ids = [
        movements[idx]["movement_id"] for idx in sorted(missing_indices)
    ]
    expected_ids = [movement["movement_id"] for movement in movements]
    if observed_ids + missing_ids != expected_ids:
        raise ScoringContractError(
            "automatic alignment can only represent an observed prefix; "
            "missing prefix or middle movements require manual timeline review"
        )
    recording_scope = "complete_performance" if not missing_ids else "partial_sequence"
    # The timeline contract requires source_end_reason to be None for a complete
    # performance and a non-empty string for a partial one; violating that would
    # make this function raise on its own output.
    source_end_reason = (
        None if not missing_ids else "auto_alignment_missing_movements_detected"
    )

    timeline_draft = {
        "schema_version": 2,
        "timeline_id": timeline_id,
        "poomsae_id": spec["poomsae_id"],
        "poomsae_version": spec["version"],
        "status": "draft",
        "label_source": "automatic",
        "frame_index_space": "sample_index",
        "frame_count": int(frame_count),
        "fps": float(fps),
        "source_binding": {
            "session_id": source_binding["session_id"],
            "run_id": source_binding["run_id"],
            "pose_file": source_binding["pose_file"],
            "pose_file_sha256": source_binding.get("pose_file_sha256"),
        },
        "coverage": {
            "recording_scope": recording_scope,
            "observed_movement_ids": observed_ids,
            "missing_movement_ids": missing_ids,
            "source_end_reason": source_end_reason,
        },
        "segments": timeline_segments,
    }
    timeline = validate_movement_timeline(timeline_draft, spec)
    anomalies = _collect_alignment_anomalies(
        movements=movements,
        segments=segments,
        timeline_segments=timeline_segments,
        best_by_segment=best_by_segment,
        pairs_by_segment=pairs_by_segment,
        fragmented=fragmented,
        missing_indices=missing_indices,
        uncertain_movement_indices=uncertain_movement_indices,
        cost=cost,
    )
    return {"timeline": timeline, "alignment_anomalies": anomalies}


def _collect_alignment_anomalies(
    *,
    movements: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    timeline_segments: list[dict[str, Any]],
    best_by_segment: dict[int, tuple[int, float]],
    pairs_by_segment: dict[int, list[int]],
    fragmented: dict[int, list[int]],
    missing_indices: set[int],
    uncertain_movement_indices: set[int],
    cost: Any,
) -> list[dict[str, Any]]:
    """Record every way the segment-to-movement match was not clean.

    None of these is a deduction. Each says only "the alignment is not certain here",
    and the honest reading is that the cause is unknown: athlete error, a segment
    detector that merged or split a movement, a partial recording, or poor pose
    quality all produce the same symptom.
    """
    anomalies: list[dict[str, Any]] = []
    frames_by_movement = {item["movement_id"]: item for item in timeline_segments}

    # A movement that ended up with no segment. Separate the two causes we can tell
    # apart: DTW never paired it at all, or it lost the one-segment-one-movement
    # contest to a closer movement.
    dedup_losers: dict[int, int] = {}
    for seg_idx, movement_indices in pairs_by_segment.items():
        winner = best_by_segment.get(seg_idx, (None, None))[0]
        for mov_idx in movement_indices:
            if mov_idx != winner:
                dedup_losers[mov_idx] = seg_idx
    for mov_idx in sorted(missing_indices):
        movement = movements[mov_idx]
        if mov_idx in dedup_losers:
            seg_idx = dedup_losers[mov_idx]
            winner_idx = best_by_segment[seg_idx][0]
            anomalies.append(
                {
                    "issue": "segment_spans_multiple_movements",
                    "movement_id": movement["movement_id"],
                    "segment_id": segments[seg_idx].get("segment_id", seg_idx),
                    "start_frame": int(segments[seg_idx]["start_frame"]),
                    "end_frame": int(segments[seg_idx]["end_frame"]),
                    "competing_movement_id": movements[winner_idx]["movement_id"],
                    "cost": round(float(cost[seg_idx][mov_idx]), 4),
                    "detail": (
                        f"{movement['movement_id']} ile "
                        f"{movements[winner_idx]['movement_id']} aynı segmente eşleşti; "
                        "segment başına tek hareket kuralı gereği daha yakın olan tutuldu."
                    ),
                }
            )
        else:
            anomalies.append(
                {
                    "issue": "unmatched_movement",
                    "movement_id": movement["movement_id"],
                    "segment_id": None,
                    "start_frame": None,
                    "end_frame": None,
                    "competing_movement_id": None,
                    "cost": None,
                    "detail": (
                        f"{movement['movement_id']} için eşleşen segment bulunamadı. "
                        "Sebep hareketin yapılmamış olması olabileceği gibi segmentin "
                        "hiç tespit edilememesi de olabilir."
                    ),
                }
            )

    # One movement that several detected segments matched: the detector fragmented it.
    for mov_idx, seg_indices in sorted(fragmented.items()):
        movement_id = movements[mov_idx]["movement_id"]
        placed = frames_by_movement.get(movement_id)
        spans = ", ".join(
            f"{segments[idx]['start_frame']}-{segments[idx]['end_frame']}" for idx in seg_indices
        )
        anomalies.append(
            {
                "issue": "movement_split_across_segments",
                "movement_id": movement_id,
                "segment_id": [segments[idx].get("segment_id", idx) for idx in seg_indices],
                "start_frame": None if placed is None else int(placed["start_frame"]),
                "end_frame": None if placed is None else int(placed["end_frame"]),
                "competing_movement_id": None,
                "cost": None,
                "detail": (
                    f"{movement_id} {len(seg_indices)} ayrı segmente bölünmüş olarak tespit "
                    f"edildi ({spans}); parçalar tek aralığa birleştirildi. Segment tespiti "
                    "hareketin düşük hareketli bölümlerini kaçırmış olabilir."
                ),
            }
        )

    # A detected segment that no movement claimed.
    for seg_idx, segment in enumerate(segments):
        if seg_idx in best_by_segment:
            continue
        anomalies.append(
            {
                "issue": "unmatched_segment",
                "movement_id": None,
                "segment_id": segment.get("segment_id", seg_idx),
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "competing_movement_id": None,
                "cost": None,
                "detail": (
                    "Bu kare aralığında hareket tespit edildi ama beklenen hiçbir "
                    "harekete oturmadı."
                ),
            }
        )

    # A match that was kept but sits inside the uncertainty band.
    for mov_idx in sorted(uncertain_movement_indices):
        movement_id = movements[mov_idx]["movement_id"]
        placed = frames_by_movement.get(movement_id)
        if placed is None:
            continue
        anomalies.append(
            {
                "issue": "ambiguous_match",
                "movement_id": movement_id,
                "segment_id": None,
                "start_frame": int(placed["start_frame"]),
                "end_frame": int(placed["end_frame"]),
                "competing_movement_id": None,
                "cost": None,
                "detail": (
                    f"{movement_id} eşleşmesi belirsizlik bandında kaldı; sporcu lehine "
                    "korundu ama insan incelemesi gerekiyor."
                ),
            }
        )
    return anomalies


def _distribute_anchors(phases: list[str], start_frame: int, end_frame: int) -> dict[str, int]:
    """Spread phase anchors evenly across a segment while preserving phase order."""
    if end_frame < start_frame:
        raise ScoringContractError("cannot distribute anchors on inverted segment")
    if not phases:
        return {}
    if len(phases) == 1:
        return {phases[0]: int(start_frame)}
    span = end_frame - start_frame
    anchors: dict[str, int] = {}
    previous = start_frame - 1
    for index, phase in enumerate(phases):
        raw = start_frame + round(span * index / (len(phases) - 1))
        raw = int(max(start_frame, min(end_frame, raw)))
        if raw <= previous:
            raw = previous + 1
        if raw > end_frame:
            raw = end_frame
        anchors[phase] = raw
        previous = raw
    return anchors
