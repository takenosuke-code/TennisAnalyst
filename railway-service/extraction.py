"""Pose-extraction pipeline: YOLO11n person crop + RTMPose-s.

This module is the lean import surface used by both Railway's FastAPI
service (`main.py`) and the Modal GPU endpoint (`modal_inference.py`).
By design it does NOT import mediapipe, supabase, fastapi, or anything
Railway-server-specific — only stdlib, cv2, numpy, and the local
`pose_rtmpose` / `tracking` / `racket_detector` / `extract_clip_keypoints`
modules. Cold-start cost on Modal drops by ~2-3s vs. importing through
`main.py`, since mediapipe alone is ~3s of transitive imports.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def angle_between(a, b, c):
    """Compute angle at joint b given three 2D points a, b, c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles_from_dicts(
    landmarks: list[dict], min_visibility: float = 0.0
) -> dict:
    """Compute key joint angles from a list of landmark dicts with x, y keys.

    `min_visibility` gates each landmark before it's used. Default 0
    preserves legacy behavior (every landmark contributes regardless of
    confidence). The RTMPose path passes a low positive value so the
    visibility=0 placeholder landmarks (face detail, hands, heels,
    foot_index) don't produce meaningless angles like
    `right_wrist: 0.0` derived from (0,0)-(0,0)-(0,0) triplets.
    """
    def get(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
            if lm.get("visibility", 1.0) < min_visibility:
                return None
            return (lm["x"], lm["y"])
        return None

    angles = {}
    pts = {
        "ls": get(11), "rs": get(12),
        "le": get(13), "re": get(14),
        "lw": get(15), "rw": get(16),
        "li": get(19), "ri": get(20),
        "lh": get(23), "rh": get(24),
        "lk": get(25), "rk": get(26),
        "la": get(27), "ra": get(28),
    }

    if all(pts[k] for k in ["rs", "re", "rw"]):
        angles["right_elbow"] = round(angle_between(pts["rs"], pts["re"], pts["rw"]), 1)
    if all(pts[k] for k in ["ls", "le", "lw"]):
        angles["left_elbow"] = round(angle_between(pts["ls"], pts["le"], pts["lw"]), 1)
    if all(pts[k] for k in ["ls", "rs", "re"]):
        angles["right_shoulder"] = round(angle_between(pts["ls"], pts["rs"], pts["re"]), 1)
    if all(pts[k] for k in ["rs", "ls", "le"]):
        angles["left_shoulder"] = round(angle_between(pts["rs"], pts["ls"], pts["le"]), 1)
    # Wrist flexion: elbow -> wrist -> index finger
    if all(pts[k] for k in ["re", "rw", "ri"]):
        angles["right_wrist"] = round(angle_between(pts["re"], pts["rw"], pts["ri"]), 1)
    if all(pts[k] for k in ["le", "lw", "li"]):
        angles["left_wrist"] = round(angle_between(pts["le"], pts["lw"], pts["li"]), 1)
    if all(pts[k] for k in ["rh", "rk", "ra"]):
        angles["right_knee"] = round(angle_between(pts["rh"], pts["rk"], pts["ra"]), 1)
    if all(pts[k] for k in ["lh", "lk", "la"]):
        angles["left_knee"] = round(angle_between(pts["lh"], pts["lk"], pts["la"]), 1)
    if pts["lh"] and pts["rh"]:
        hip_vec = (pts["rh"][0] - pts["lh"][0], pts["rh"][1] - pts["lh"][1])
        angles["hip_rotation"] = round(
            abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]))), 1
        )
    if pts["ls"] and pts["rs"]:
        sh_vec = (pts["rs"][0] - pts["ls"][0], pts["rs"][1] - pts["ls"][1])
        angles["trunk_rotation"] = round(
            abs(math.degrees(math.atan2(sh_vec[1], sh_vec[0]))), 1
        )

    return angles


def _open_video_autorotated(video_path: str) -> cv2.VideoCapture:
    """cv2.VideoCapture with CAP_PROP_ORIENTATION_AUTO on. Phone portrait
    clips carry a 90° rotation tag that OpenCV won't apply by default —
    YOLO and RTMPose both collapse to zero detections on sideways frames.
    See extract_clip_keypoints._open_video_autorotated for the full
    rationale and the measured impact on IMG_1097.mov.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:  # noqa: BLE001
        pass
    return cap


def _try_load_racket_detector():
    """Best-effort import of the racket detector.

    Returns either the callable or None when ultralytics/onnxruntime
    isn't fully wired (e.g. unit-test environments without GPU/CPU
    weights). Logged but never raised so extraction still runs.
    """
    try:
        from racket_detector import detect_racket  # type: ignore

        return detect_racket
    except Exception as e:  # noqa: BLE001
        print(f"[extract] racket_detector unavailable: {e}")
        return None


def _detect_racket_for_frame(
    detect_racket,
    frame_bgr,
    landmarks: list[dict],
    processed_index: int,
    timestamp_ms: float | None = None,
    racket_tracker=None,
    skip_yolo: bool = False,
):
    """Shared racket-head detection. Picks the higher-visibility wrist as the
    dominant hand and feeds the racket detector. Routes the result through a
    Kalman `racket_tracker` (when provided) to smooth YOLO jitter and coast
    short motion-blur gaps. Returns None on any failure (logged but never
    raised).

    `skip_yolo=True` bypasses the YOLO call entirely and only feeds None
    to the tracker (which coasts on its previous estimate). Use when the
    wrist hasn't moved much between frames — running YOLO on a static
    arm is wasted compute."""
    yolo_point = None
    if detect_racket is not None and not skip_yolo:
        h, w = frame_bgr.shape[:2]
        lw = landmarks[15] if len(landmarks) > 15 else None
        rw = landmarks[16] if len(landmarks) > 16 else None
        dominant = None
        if lw and rw:
            dominant = rw if rw["visibility"] >= lw["visibility"] else lw
        elif lw:
            dominant = lw
        elif rw:
            dominant = rw
        wrist_xy = (
            (dominant["x"] * w, dominant["y"] * h)
            if dominant is not None
            else None
        )
        # Crop to landmark envelope + reach margin. See
        # extract_clip_keypoints._person_crop_bbox for rationale.
        from extract_clip_keypoints import _person_crop_bbox
        person_bbox = _person_crop_bbox(landmarks, w, h)
        try:
            yolo_point = detect_racket(
                frame_bgr, wrist_xy, person_bbox=person_bbox,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[extract] racket detect failed on frame {processed_index}: {e}")
            yolo_point = None

    if racket_tracker is not None and timestamp_ms is not None:
        return racket_tracker.update(yolo_point, timestamp_ms=timestamp_ms)
    return yolo_point


# Below this threshold (in normalized [0,1] frame coords) the dominant
# wrist is considered "stationary" and we skip the racket-detector
# YOLO call. 0.005 ≈ 0.5% of frame width; on a 320-wide clip that's
# ~1.6px of wrist drift per sampled frame, which is well below any
# real swing motion (a forehand swing moves the wrist ~0.05-0.1 norm
# per frame at 30fps). The Kalman tracker coasts through skipped
# frames so the racket trail stays continuous.
RACKET_MOTION_GATE_NORM = 0.005

# Two-pass extraction kicks in for clips longer than this. Below the
# threshold the pass-1 overhead isn't worth it — short clips just run
# the heavy single-pass path directly. 30s matches the lib/poseExtraction
# adaptive-sampling first-tier boundary, so the boundary is consistent
# across browser and server paths.
TWO_PASS_MIN_DURATION_SEC = 30.0

# Pass 1 sample rate (fps). Lightweight YOLO + RTMPose-s wrist-only
# inference. 8fps is well above the Nyquist limit for tennis swing
# detection: a forehand contact event spans ~120-180ms of high wrist
# velocity, and 8fps = 125ms/sample gives us 1-2 samples on the peak
# even for the shortest swings. Going lower (5fps) starts to risk
# aliasing the peak; going higher (15fps) doubles the pass-1 cost
# without changing the candidate-window output meaningfully.
PASS1_SAMPLE_FPS = 8

# Stroke-detection refractory after a peak (ms). Two real strokes can't
# happen closer than ~500ms in tennis (even rapid volleys at the net
# pause briefly between hits). Used to suppress secondary peaks from a
# single swing's wind-up + follow-through showing up as separate peaks.
# MUST match the JS default in lib/swingDetect.ts STROKE_DEFAULTS.
STROKE_REFRACTORY_MS = 500.0

# Pre/post pad (ms) around each detected peak. The window must include
# the wind-up before contact and the follow-through after. MUST match
# the JS defaults in lib/swingDetect.ts (prePadMs=1000, postPadMs=500).
STROKE_PRE_PAD_MS = 1000.0
STROKE_POST_PAD_MS = 500.0

# Smoothing window (frames) for the centered moving average over wrist
# speed. MUST match JS STROKE_DEFAULTS.smoothingFrames (5).
STROKE_SMOOTHING_FRAMES = 5

# Adaptive threshold multiplier on the (p90 - median) spread. MUST
# match JS STROKE_DEFAULTS.thresholdK (1.0).
STROKE_THRESHOLD_K = 1.0

# Minimum frame distance between two raw peaks before refractory dedup.
# MUST match JS STROKE_DEFAULTS.minPeakDistanceFrames (3). Refractory
# does the heavy lifting; this exists so two real peaks 300ms apart
# (≈9 frames at 30fps) can both make it past peak-finding before
# refractory chooses one.
STROKE_MIN_PEAK_DISTANCE_FRAMES = 3

# Tuple of all peak-pick + padding constants. Order is load-bearing:
# the SHA256 of str(TWO_PASS_PARAMS) is appended to Modal's per-clip
# cache key on two-pass requests, so any tuning of these values
# implicitly invalidates stale cached keypoints. Don't reorder fields
# without bumping POSE_CACHE_MODEL_VERSION as well.
TWO_PASS_PARAMS = (
    STROKE_REFRACTORY_MS,
    STROKE_PRE_PAD_MS,
    STROKE_POST_PAD_MS,
    STROKE_SMOOTHING_FRAMES,
    STROKE_THRESHOLD_K,
)


def two_pass_params_hash() -> str:
    """Return an 8-char SHA256 prefix of the TWO_PASS_PARAMS tuple.

    Used to extend Modal's per-clip cache key when a request runs the
    two-pass orchestrator. Tuning any peak-pick constant rotates this
    hash, so a same-blob retry after a tuning deploy gets fresh
    keypoints rather than stale cached ones from the old constants.
    """
    import hashlib

    return hashlib.sha256(str(TWO_PASS_PARAMS).encode()).hexdigest()[:8]


# Two-pass falls back to a full single-pass when fewer than this many
# strokes are detected in pass 1. Small candidate counts mean the
# overhead of a separate pass-1 + pass-2 outweighs the saved compute,
# AND it's possible pass 1 missed strokes (low confidence wrist
# tracking on a tough clip) — running the heavy model on everything
# is the safe fallback.
MIN_STROKES_FOR_TWO_PASS = 5


# ---------------------------------------------------------------------------
# Pass 1: lightweight wrist-trajectory extraction
# ---------------------------------------------------------------------------


def _run_wrist_pass(
    video_path: str,
    pass1_fps: int = PASS1_SAMPLE_FPS,
) -> tuple[list[float], list[tuple[float, float] | None], float, float]:
    """Pass 1: lightweight YOLO + RTMPose-s wrist-trajectory extraction.

    Samples the video at `pass1_fps` (default 8fps), runs YOLO + RTMPose
    on each sampled frame, and emits ONLY the dominant wrist's normalized
    (x, y) position per frame. Skips:

      * joint-angle compute (pure CPU, ~0.1ms/frame but not free at
        scale)
      * the racket detector + Kalman tracker (the racket head isn't
        used by stroke detection)
      * the COCO-17 → BlazePose-33 remap (pass 1 only needs ids 9/10
        from the raw COCO-17 output — the dominant-wrist pick happens
        directly on COCO scores)

    Returns:
      timestamps_ms — list of millisecond timestamps for each sampled
        frame (regardless of whether wrist tracking succeeded)
      wrists       — list of (x_norm, y_norm) tuples, or None when no
        wrist landmark cleared the visibility gate
      video_fps    — original video framerate (needed to convert
        candidate windows to original-frame indices)
      duration_sec — total video duration in seconds

    The function intentionally returns `None` for frames where wrist
    tracking failed rather than dropping them, so the stroke detector
    can use timestamp continuity to decide whether to interpolate or
    skip a peak candidate.
    """
    from pose_rtmpose import (
        _yolo_person_candidates,
        _ensure_rtmpose,
        expand_bbox,
    )
    import pose_rtmpose

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

    sample_interval = max(1, int(round(video_fps / pass1_fps)))

    timestamps_ms: list[float] = []
    wrists: list[tuple[float, float] | None] = []

    # Materialize the rtmlib session once before we loop. This is the
    # lazy import that pulls onnxruntime; doing it here means cold-start
    # overhead is paid once for both passes (the heavy pass reuses the
    # same _rtm_session).
    _ensure_rtmpose()

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                ts_ms = (frame_idx / video_fps) * 1000.0
                timestamps_ms.append(ts_ms)

                wrist = _extract_dominant_wrist(frame)
                wrists.append(wrist)
            frame_idx += 1
    finally:
        cap.release()

    return timestamps_ms, wrists, video_fps, duration_sec


def _extract_dominant_wrist(
    frame_bgr: np.ndarray,
) -> tuple[float, float] | None:
    """Run YOLO + RTMPose-s on a frame and return the dominant wrist's
    normalized (x, y), or None if neither wrist passed the visibility gate.

    Internally short-circuits the BlazePose-33 remap — we only need
    COCO ids 9 (left wrist) and 10 (right wrist) and their scores. This
    saves a tiny bit per frame but more importantly avoids the bbox
    expansion/visibility-zeroing logic that the heavy path needs but
    that we don't, for a wrist-only signal.
    """
    from pose_rtmpose import (
        _yolo_person_candidates,
        expand_bbox,
    )
    import pose_rtmpose

    h, w = frame_bgr.shape[:2]

    try:
        candidates, _w, _h = _yolo_person_candidates(frame_bgr)
    except Exception as e:  # noqa: BLE001
        print(f"[extract:pass1] YOLO failed: {e}")
        return None
    if not candidates:
        return None

    # Pick the highest-confidence person. Pass 1 doesn't need the
    # PersonTracker's identity stickiness — wrist-trajectory peak
    # detection is robust to occasional player-id flips, and skipping
    # the tracker keeps pass 1 closer to "pure" lightweight cost.
    bbox = max(candidates, key=lambda c: c[4])
    expanded = expand_bbox(bbox, w, h)

    # Run RTMPose on the bbox. We use the existing session (already
    # warmed by _ensure_rtmpose). This is the same model as pass 2
    # currently — see `summary` doc / future_work for the upgrade path
    # to RTMPose-S → RTMPose-M for pass 2.
    session = pose_rtmpose._rtm_session
    if session is None:
        return None

    try:
        keypoints, scores = session(frame_bgr, bboxes=[list(expanded)])
    except Exception as e:  # noqa: BLE001
        print(f"[extract:pass1] RTMPose failed: {e}")
        return None

    if keypoints is None or len(keypoints) == 0:
        return None

    # COCO-17: id 9 = L wrist, id 10 = R wrist. Pick the higher-score
    # one as dominant — same dominant-hand rule the racket detector uses.
    kpts = keypoints[0]  # shape (17, 2) in image-pixel coords
    sc = scores[0]       # shape (17,)
    lw_score = float(sc[9])
    rw_score = float(sc[10])

    if lw_score < 0.1 and rw_score < 0.1:
        return None

    if rw_score >= lw_score:
        x_px, y_px = kpts[10]
    else:
        x_px, y_px = kpts[9]

    x_norm = float(np.clip(x_px / w, 0.0, 1.0))
    y_norm = float(np.clip(y_px / h, 0.0, 1.0))
    return (x_norm, y_norm)


# ---------------------------------------------------------------------------
# Stroke detection from wrist trajectory
# ---------------------------------------------------------------------------


def _detect_strokes_from_wrist(
    timestamps_ms: list[float],
    wrists: list[tuple[float, float] | None],
    video_fps: float,
    refractory_ms: float = STROKE_REFRACTORY_MS,
    pre_pad_ms: float = STROKE_PRE_PAD_MS,
    post_pad_ms: float = STROKE_POST_PAD_MS,
    smoothing_window: int = STROKE_SMOOTHING_FRAMES,
    threshold_k: float = STROKE_THRESHOLD_K,
    min_peak_distance_frames: int = STROKE_MIN_PEAK_DISTANCE_FRAMES,
) -> list[tuple[int, int, int]]:
    """Faithful Python port of `detectStrokes` in lib/swingDetect.ts.

    The JS swing detector is the source of truth: this function
    reproduces its algorithm bit-for-bit so the server-side stroke
    candidate windows (used to gate Modal's heavy pose pass) match
    what the browser would compute on the same wrist signal.

      1. Compute frame-to-frame wrist speed (||Δposition|| / Δt). Frames
         where either endpoint is missing wrist tracking contribute 0.
      2. Smooth with a centered moving-average of `smoothing_window`
         frames (default 5; matches JS STROKE_DEFAULTS.smoothingFrames).
      3. Compute the adaptive threshold from the SMOOTHED signal:
         `median + threshold_k * (p90 - median)`. Empty spread =>
         Infinity threshold (no peaks).
      4. Find local maxima: a sample is a local max iff it dominates
         BOTH immediate neighbors (`v >= left and v >= right` with at
         least one strict inequality, which lets plateaus' first index
         survive). Apply a min-distance dedup: if a candidate is within
         `min_peak_distance_frames` of the previously accepted peak,
         keep whichever has the larger smoothed speed.
      5. Refractory: sort accepted candidates by smoothed speed
         DESCENDING and walk in that order, accepting a peak only if
         no already-accepted peak is within `refractoryFrames`.
         `refractoryFrames = max(1, round(refractory_ms/1000 * sample_fps))`,
         where `sample_fps` is derived from the input timestamps' median
         delta — NOT `video_fps`. (The detector operates on pass-1
         sample indices; `video_fps` is only used at the end to convert
         peak/pad timestamps back to original-video frame indices.)
      6. Pad each surviving peak to a (peak - pre_pad_ms, peak +
         post_pad_ms) window in milliseconds, then convert to
         original-video frame indices via `video_fps`. Pre/post pad
         frames use math.ceil (matches JS Math.ceil) so a small ms pad
         never clips to 0 frames.

    Returns a list of (start_frame, end_frame, peak_frame) tuples in
    ORIGINAL-VIDEO frame indices, sorted by start_frame. Frame indices
    are clamped at 0 below; callers clamp at the upper bound (total
    frames) as needed.
    """
    n = len(timestamps_ms)
    if n < 2:
        return []
    if len(wrists) != n:
        raise ValueError("timestamps_ms and wrists must be same length")

    # Step 1: frame-to-frame wrist speed in normalized-coords-per-second.
    # Speeds[0] = 0 (no previous frame to diff against). Mirrors
    # computeWristSpeeds() in lib/swingDetect.ts: a missing/occluded
    # wrist endpoint yields 0 rather than NaN so the smoother doesn't
    # poison neighbours.
    speeds: list[float] = [0.0] * n
    for i in range(1, n):
        a = wrists[i - 1]
        b = wrists[i]
        if a is None or b is None:
            speeds[i] = 0.0
            continue
        dt_s = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        if dt_s <= 0:
            speeds[i] = 0.0
            continue
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        speeds[i] = ((dx * dx + dy * dy) ** 0.5) / dt_s

    # Step 2: centered moving average. Half-window = floor(W/2) so a
    # window of 5 looks at i±2. Edges use whatever neighbours exist
    # (not zero-padded). Matches smooth() in lib/swingDetect.ts.
    half = max(0, smoothing_window // 2)
    smoothed: list[float] = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        s = 0.0
        c = 0
        for j in range(lo, hi + 1):
            s += speeds[j]
            c += 1
        smoothed[i] = s / c if c > 0 else 0.0

    # Step 3: adaptive threshold over smoothed signal.
    # threshold = median + threshold_k * (p90 - median). Spread <= 0
    # (perfectly flat clip) bumps to +inf so no peaks fire — same
    # flat-signal early-out the JS path uses.
    sorted_s = sorted(smoothed)
    median = sorted_s[len(sorted_s) // 2]
    p90 = sorted_s[min(len(sorted_s) - 1, int(len(sorted_s) * 0.9))]
    spread = max(0.0, p90 - median)
    if spread <= 0:
        return []
    threshold = median + threshold_k * spread

    # Refractory operates on the sample-series cadence, not the original
    # video fps. Derive sample_fps from the timestamps' median delta —
    # mirrors deriveFps() in lib/swingDetect.ts. Falls back to video_fps
    # (and ultimately 30) when timestamps are too short / degenerate.
    sample_fps = _derive_sample_fps(timestamps_ms, fallback=video_fps)

    refractory_frames = max(1, round((refractory_ms / 1000.0) * sample_fps))
    min_peak_distance = min(min_peak_distance_frames, refractory_frames)

    # Step 4: local maxima with min-distance dedup. Mirrors
    # findLocalMaxima() in lib/swingDetect.ts.
    raw_peaks: list[int] = []
    for i in range(n):
        v = smoothed[i]
        if v < threshold:
            continue
        # Local max: not less than either immediate neighbour AND
        # strictly greater than at least one of them (handles plateaus
        # by keeping the first index of the plateau). Edge frames see
        # -infinity on the missing side, so they pass the >=
        # comparison automatically.
        left = smoothed[i - 1] if i > 0 else float("-inf")
        right = smoothed[i + 1] if i < n - 1 else float("-inf")
        is_local_max = (
            v >= left
            and v >= right
            and (v > left or v > right or n == 1)
        )
        if not is_local_max:
            continue
        # Min-distance dedup: if the previous accepted peak is within
        # min_peak_distance, keep the taller. Greedy forward pass.
        if raw_peaks:
            last = raw_peaks[-1]
            if i - last < min_peak_distance:
                if smoothed[i] > smoothed[last]:
                    raw_peaks[-1] = i
                continue
        raw_peaks.append(i)

    if not raw_peaks:
        return []

    # Step 5: refractory. Sort candidates by smoothed speed DESCENDING
    # and greedily accept the strongest, blocking any candidate within
    # refractory_frames of an already-accepted peak. Mirrors
    # applyRefractory() in lib/swingDetect.ts.
    by_strength = sorted(raw_peaks, key=lambda i: smoothed[i], reverse=True)
    accepted: list[int] = []
    for p in by_strength:
        blocked = False
        for a in accepted:
            if abs(p - a) < refractory_frames:
                blocked = True
                break
        if not blocked:
            accepted.append(p)
    accepted.sort()

    # Step 6: pad each surviving peak to (start_ms, end_ms) and convert
    # to original-video frame indices via video_fps. math.ceil for
    # pre/post pad matches JS Math.ceil.
    if video_fps <= 0:
        video_fps = 30.0  # defensive — caller should always pass real fps

    pre_pad_frames = math.ceil((pre_pad_ms / 1000.0) * video_fps)
    post_pad_frames = math.ceil((post_pad_ms / 1000.0) * video_fps)

    windows: list[tuple[int, int, int]] = []
    for sample_idx in accepted:
        peak_ts_ms = timestamps_ms[sample_idx]
        peak_frame = round((peak_ts_ms / 1000.0) * video_fps)
        start_frame = peak_frame - pre_pad_frames
        end_frame = peak_frame + post_pad_frames
        windows.append((max(0, start_frame), max(0, end_frame), max(0, peak_frame)))

    return windows


def _derive_sample_fps(timestamps_ms: list[float], fallback: float) -> float:
    """Derive sample-series fps from timestamp deltas.

    Faithful port of `deriveFps` in lib/swingDetect.ts. Uses the median
    Δt (robust to a few duplicated/dropped frames) and snaps integer
    nearby (within 0.05fps) so canonical 8/30/60fps captures hit
    integer frame arithmetic. Falls back to `fallback` (and ultimately
    30) when timestamps are too short or all equal.
    """
    if len(timestamps_ms) < 2:
        return fallback if fallback > 0 else 30.0
    diffs = [
        timestamps_ms[i] - timestamps_ms[i - 1]
        for i in range(1, len(timestamps_ms))
        if (timestamps_ms[i] - timestamps_ms[i - 1]) > 0
        and math.isfinite(timestamps_ms[i] - timestamps_ms[i - 1])
    ]
    if not diffs:
        return fallback if fallback > 0 else 30.0
    diffs.sort()
    median_dt = diffs[len(diffs) // 2]
    if not median_dt or median_dt <= 0:
        return fallback if fallback > 0 else 30.0
    fps = 1000.0 / median_dt
    if abs(fps - round(fps)) < 0.05:
        return float(round(fps))
    return fps


def _merge_windows(
    windows: list[tuple[int, int, int]],
    merge_gap_frames: int = 0,
) -> list[tuple[int, int, int]]:
    """Merge overlapping or adjacent stroke windows.

    Two windows are merged if their frame ranges overlap OR if the gap
    between them (start of next - end of prev) is <= `merge_gap_frames`.
    The merged window's peak is the peak from whichever input window had
    the LATER peak — closer to the end of the merged window, which gives
    downstream pose-aware stroke quality scoring a more conservative
    contact estimate.

    Sorting + merging is critical before pass 2: without it, two wrist
    velocity peaks ~600ms apart (overlapping post-pad of the first with
    pre-pad of the second) would produce two windows whose frame ranges
    overlap, and the heavy pose extraction would run YOLO + RTMPose on
    the overlapping frames TWICE.

    Returns a new list, sorted by start_frame, with no overlaps and no
    gaps below `merge_gap_frames`.
    """
    if not windows:
        return []

    sorted_w = sorted(windows, key=lambda w: w[0])
    merged: list[tuple[int, int, int]] = [sorted_w[0]]
    for s, e, p in sorted_w[1:]:
        last_s, last_e, last_p = merged[-1]
        if s <= last_e + merge_gap_frames:
            new_s = last_s
            new_e = max(last_e, e)
            # Prefer the LATER peak — see docstring.
            new_p = p if p >= last_p else last_p
            merged[-1] = (new_s, new_e, new_p)
        else:
            merged.append((s, e, p))
    return merged


def _dominant_wrist_norm(landmarks: list[dict]) -> tuple[float, float] | None:
    """Pick the higher-visibility wrist from a 33-entry landmark list and
    return its normalized (x, y) coords, or None if neither wrist is
    visible. Mirrors _detect_racket_for_frame's dominant-hand pick so
    the motion gate uses the same wrist the racket detector would."""
    if len(landmarks) <= 16:
        return None
    lw = landmarks[15]
    rw = landmarks[16]
    if not lw and not rw:
        return None
    if lw and rw:
        dominant = rw if rw["visibility"] >= lw["visibility"] else lw
    else:
        dominant = lw or rw
    if dominant is None or dominant.get("visibility", 0) < 0.1:
        return None
    return (dominant["x"], dominant["y"])


def _extract_with_rtmpose(
    video_path: str,
    sample_fps: int,
    max_seconds: float,
    candidate_windows: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Pose extraction via YOLO11n person crop + RTMPose-s.

    Schema_version=3: COCO-17 keypoints remapped to BlazePose-33 ids, with
    the unmapped ids (face details, hands, heels, foot_index) at
    visibility=0. wrist-flexion joint angles are dropped because the
    index-finger landmarks (BlazePose 19/20) aren't filled.

    Pipeline shape — chunked, staged-parallel:

      1. Decode N frames sequentially into a chunk (cv2.VideoCapture
         is not thread-safe).
      2. Stage 1 (parallel): YOLO11n person detection per frame in the
         chunk. Stateless — safe to fan out.
      3. Stage 2 (sequential, in submit order): PersonTracker.update.
         The tracker's IoU association needs frame N's candidates to
         see frame N-1's reference bbox, so this stage MUST stay
         sequential. It's also cheap (microseconds per call), so this
         doesn't bottleneck.
      4. Stage 3 (parallel): RTMPose-s inference per (frame, bbox).
         Stateless given inputs — rtmlib's wrapper allocates per-call
         numpy arrays and the underlying ORT session is thread-safe.
      5. Stage 4 (sequential): joint angles + racket Kalman tracker.
         Racket Kalman is order-dependent.

    Net on a 2-4 core Railway CPU: ~2-3x throughput vs. the prior
    fully-sequential per-frame loop, no quality regression. The
    previous all-at-once parallel attempt (commit b802df4, reverted
    in aceca57) broke because it parallelized the tracker too —
    tracker.update saw frames out of order. This staged design fixes
    the structural problem.

    `candidate_windows`, when provided, is a list of (start_frame,
    end_frame, peak_frame) triples in original-video frame indices.
    The heavy loop only processes frames whose ORIGINAL-VIDEO index
    falls inside one of those windows. This is the pass-2 entry point
    for two-pass orchestration: caller has already detected stroke
    candidates from a cheap pass-1 wrist trajectory and merged
    overlapping windows. When `None`, every sampled frame is processed
    (single-pass behavior, used by the short-clip fast path and
    pre-existing callers).
    """
    from pose_rtmpose import (
        _yolo_person_candidates,
        _ensure_rtmpose,
        infer_rtmpose_for_bbox,
    )
    from tracking import PersonTracker, RacketTracker

    detect_racket = _try_load_racket_detector()

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Adaptive sampling: long clips at 30fps overflow the 180s extraction
    # timeout (a 2-min clip = 3600 frames * ~30ms/frame = 108s of
    # inference alone, before download/decode/serialization). Drop the
    # effective sample rate for long clips so user-uploaded full match
    # videos still fit the budget. Mirrors the browser-side adaptive
    # sampling at lib/poseExtraction.ts:184-198.
    duration_sec_est = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / video_fps

    # Frame-level filtering: when `candidate_windows` is provided, only
    # process frames whose ORIGINAL-VIDEO index falls inside a window.
    # Replaces the previous frame-difference motion pre-pass — the
    # two-pass orchestrator passes pose-derived stroke windows here,
    # which are tighter and more discriminative than raw motion peaks.
    total_window_sec = 0.0
    if candidate_windows:
        # Extract just (start, end) for the in-window check; peak isn't
        # used here.
        check_windows: list[tuple[int, int]] = [
            (s, e) for s, e, _p in candidate_windows
        ]
        total_window_frames = sum(e - s for s, e in check_windows)
        total_window_sec = total_window_frames / video_fps if video_fps > 0 else 0.0
        clip_pct = (
            100 * total_window_sec / duration_sec_est
            if duration_sec_est > 0
            else 0.0
        )
        print(
            f"[extract:rtmpose] pass 2 candidate windows: "
            f"{len(check_windows)} windows covering {total_window_sec:.1f}s "
            f"of {duration_sec_est:.1f}s "
            f"({clip_pct:.0f}%)"
        )
    else:
        check_windows = []

    # Choose the duration that drives the fps cap. Single-pass uses the
    # full clip duration (existing behavior). Two-pass with candidate
    # windows uses the SUMMED window duration: with ~15-25% coverage on
    # match tape, even a 5-min clip's actual heavy work is ~45-75s, so
    # capping at 8fps the way we do for full 5-min single-pass leaves
    # the natural sample_fps on the table without budget pressure.
    cap_basis_sec = total_window_sec if candidate_windows else duration_sec_est

    if cap_basis_sec > 120:
        effective_sample_fps = min(sample_fps, 8)
    elif cap_basis_sec > 60:
        effective_sample_fps = min(sample_fps, 12)
    elif cap_basis_sec > 30:
        effective_sample_fps = min(sample_fps, 18)
    else:
        # Short candidate-window total OR short single-pass clip: don't
        # cap — let the caller's requested sample_fps through.
        effective_sample_fps = sample_fps

    frame_interval = max(1, int(video_fps / effective_sample_fps))
    max_frame = int(max_seconds * video_fps) if max_seconds > 0 else 0
    if effective_sample_fps != sample_fps:
        print(
            f"[extract:rtmpose] adaptive sample fps: clip is {duration_sec_est:.1f}s, "
            f"sampling at {effective_sample_fps}fps (was {sample_fps})"
        )
    frames: list[dict] = []
    frame_index = 0
    processed_index = 0

    # See _extract_rtmpose in extract_clip_keypoints.py for rationale.
    person_tracker = PersonTracker()
    racket_tracker = RacketTracker()

    # Materialize the rtmlib session once before workers race for it.
    # Avoids N threads simultaneously hitting the lazy-init lock on the
    # first chunk.
    _ensure_rtmpose()

    # Previous-frame dominant wrist position (normalized). Used by the
    # motion gate to skip the racket-detector YOLO call on frames where
    # the wrist has barely moved (player setting up / between shots).
    # Persists across chunk boundaries so the first frame of chunk N+1
    # compares against the last frame of chunk N rather than blindly
    # running YOLO on it.
    prev_wrist_norm: tuple[float, float] | None = None
    racket_yolo_skipped = 0  # debug telemetry only
    person_yolo_skipped = 0  # debug telemetry for the person-YOLO frame skip

    # Cached frame dimensions from the most recent YOLO call. Used on
    # coast frames (where person YOLO is skipped) so PersonTracker.update
    # still receives a sensible (w, h) — these aren't load-bearing during
    # coasting (the tracker just stores them) but passing 0,0 is unsafe
    # if a future tracker change starts validating dimensions.
    last_known_wh: tuple[int, int] | None = None

    # 16-frame chunks bound peak buffered memory (~16 * 1080p ≈ 100MB
    # worst case) while keeping the inference pool busy enough to
    # amortize submit/collect overhead.
    CHUNK_SIZE = 16
    # 4 workers matches a typical 4-core Railway CPU. With
    # intra_op_num_threads=1 on each ORT session, 4 concurrent
    # inferences saturate the cores cleanly without scheduler thrash.
    POOL_WORKERS = 4
    # Run person-YOLO every Nth chunk frame; the rest coast on the
    # PersonTracker's smoothed bbox. The tracker already absorbs missed
    # detections by holding the previous reference (see tracking.py
    # PersonTracker.update — coast path returns the previous bbox when
    # no candidates pass the IoU gate). On Modal T4 YOLO is ~25-40% of
    # per-frame cost; at N=3 we run YOLO 33% of frames, saving ~17-27%
    # wall time with no quality regression on a stable subject. K=3 also
    # bounds the worst-case bbox drift to 3 frames, well below the
    # PersonTracker's max_coast_frames default.
    YOLO_DETECT_EVERY_N = 3

    def _process_chunk(
        chunk: list[tuple[int, int, np.ndarray]],
        pool: ThreadPoolExecutor,
    ) -> None:
        nonlocal processed_index, prev_wrist_norm, racket_yolo_skipped
        nonlocal last_known_wh, person_yolo_skipped

        # Stage 1: parallel YOLO, but only on every Nth chunk frame.
        # Skipped frames feed empty candidates to the PersonTracker on
        # Stage 2 so it coasts on its previous reference bbox.
        yolo_futures: list = []
        for i, (_, _, fb) in enumerate(chunk):
            if i % YOLO_DETECT_EVERY_N == 0:
                yolo_futures.append(pool.submit(_yolo_person_candidates, fb))
            else:
                yolo_futures.append(None)
                person_yolo_skipped += 1

        # Stage 2: sequential tracker.update in SUBMIT ORDER. The
        # tracker's IoU-association invariant requires this — frame N's
        # candidates must update against frame N-1's reference. Wait
        # for each YOLO future in order rather than `as_completed`.
        bboxes: list = []
        for fut, (fidx, _, _) in zip(yolo_futures, chunk):
            if fut is None:
                # Coast frame: skip YOLO, pass empty candidates so the
                # tracker coasts on its previous reference. Use cached
                # frame dimensions — `last_known_wh` is set on the most
                # recent successful YOLO result.
                if last_known_wh is None:
                    # Defensive: never coast before we've seen a frame.
                    bboxes.append(None)
                    continue
                w, h = last_known_wh
                try:
                    bbox = person_tracker.update([], w, h)
                except Exception as e:  # noqa: BLE001
                    print(f"[extract:rtmpose] tracker coast failed on frame {fidx}: {e}")
                    bbox = None
                bboxes.append(bbox)
                continue
            try:
                candidates, w, h = fut.result()
                last_known_wh = (w, h)
            except Exception as e:  # noqa: BLE001
                print(f"[extract:rtmpose] YOLO failed on frame {fidx}: {e}")
                bboxes.append(None)
                continue
            try:
                bbox = person_tracker.update(candidates, w, h)
            except Exception as e:  # noqa: BLE001
                print(f"[extract:rtmpose] tracker failed on frame {fidx}: {e}")
                bbox = None
            bboxes.append(bbox)

        # Stage 3: parallel RTMPose. Submit only frames with a bbox.
        rtm_futures: list = []
        for (_, _, fb), bbox in zip(chunk, bboxes):
            if bbox is None:
                rtm_futures.append(None)
            else:
                rtm_futures.append(pool.submit(infer_rtmpose_for_bbox, fb, bbox))

        # Stage 4: sequential post-processing in submit order. Racket
        # Kalman + joint angles need temporal order.
        for (fidx, ts, fb), fut in zip(chunk, rtm_futures):
            if fut is None:
                continue
            try:
                landmarks = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[extract:rtmpose] RTMPose failed on frame {fidx}: {e}")
                continue
            if landmarks is None:
                continue

            # min_visibility=0.1 skips the v=0 placeholder ids that
            # exist in v3 only because COCO-17 doesn't fill them.
            joint_angles = compute_joint_angles_from_dicts(
                landmarks, min_visibility=0.1
            )

            # Motion gate: if the dominant wrist hasn't moved much
            # since the previous processed frame, skip the racket
            # YOLO call. The Kalman tracker coasts through.
            cur_wrist_norm = _dominant_wrist_norm(landmarks)
            skip_yolo = False
            if cur_wrist_norm is not None and prev_wrist_norm is not None:
                dx = cur_wrist_norm[0] - prev_wrist_norm[0]
                dy = cur_wrist_norm[1] - prev_wrist_norm[1]
                motion = (dx * dx + dy * dy) ** 0.5
                if motion < RACKET_MOTION_GATE_NORM:
                    skip_yolo = True
                    racket_yolo_skipped += 1
            prev_wrist_norm = cur_wrist_norm

            racket_head = _detect_racket_for_frame(
                detect_racket, fb, landmarks, processed_index,
                timestamp_ms=ts,
                racket_tracker=racket_tracker,
                skip_yolo=skip_yolo,
            )
            frames.append({
                "frame_index": processed_index,
                "timestamp_ms": round(ts, 1),
                "landmarks": landmarks,
                "joint_angles": joint_angles,
                "racket_head": racket_head,
            })
            processed_index += 1

    def _in_candidate_window(idx: int) -> bool:
        """True when `idx` is inside any candidate stroke window (or
        when no windows were provided, in which case every frame is
        in — i.e. single-pass mode)."""
        if not check_windows:
            return True
        for s, e in check_windows:
            if s <= idx <= e:
                return True
        return False

    chunk: list[tuple[int, int, np.ndarray]] = []
    window_skipped = 0
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frame > 0 and frame_index >= max_frame:
                break

            if frame_index % frame_interval == 0:
                if not _in_candidate_window(frame_index):
                    window_skipped += 1
                    frame_index += 1
                    continue
                timestamp_ms = int((frame_index / video_fps) * 1000)
                # .copy() so cv2's reused frame buffer doesn't change
                # under us while a worker thread still holds the ref.
                chunk.append((frame_index, timestamp_ms, frame.copy()))
                if len(chunk) >= CHUNK_SIZE:
                    _process_chunk(chunk, pool)
                    chunk = []

            frame_index += 1

        if chunk:
            _process_chunk(chunk, pool)

    if check_windows:
        print(
            f"[extract:rtmpose] pass-2 window filter skipped {window_skipped} sampled frames "
            f"({person_yolo_skipped} of the processed frames also skipped person YOLO)"
        )

    cap.release()

    total_frames = frame_index
    total_duration_ms = (total_frames / video_fps) * 1000

    return {
        # Report the EFFECTIVE sample rate so downstream code (joint
        # angle smoothing, swing detection) knows the actual frame
        # cadence rather than the requested-but-not-applied rate.
        "fps_sampled": effective_sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        # v3: COCO-17 backbone via RTMPose. Wrist-flexion absent.
        "schema_version": 3,
        # Identifies which backend produced these keypoints. The browser
        # surfaces this as a diagnostic chip so we can tell at a glance
        # whether a "tracing is off" complaint is coming from server
        # rtmpose vs browser fallback.
        "pose_backend": "rtmpose",
    }


def extract_keypoints_from_video(
    video_path: str,
    sample_fps: int = 30,
    max_seconds: float = 0,
    two_pass: bool = True,
) -> dict:
    """Extract pose keypoints from a video file via YOLO11n + RTMPose-s.

    Args:
        max_seconds: Stop after this many seconds. 0 = process entire video.
        two_pass: When True (default), use the two-pass orchestration on
            long clips: a cheap pass-1 wrist-trajectory extraction at
            PASS1_SAMPLE_FPS to detect candidate stroke windows, then a
            heavy pass-2 pose extraction restricted to those windows.
            For short clips (< TWO_PASS_MIN_DURATION_SEC) or when pass 1
            finds too few candidate strokes, we transparently fall back
            to a single-pass full extraction. Set False to force single-
            pass on every clip (used by callers that already know the
            input is short, like the live-coach batch extractor, or
            tests that want deterministic single-pass behavior).
    """
    if not two_pass:
        return _extract_with_rtmpose(video_path, sample_fps, max_seconds)
    return _two_pass_extract(video_path, sample_fps, max_seconds)


def _two_pass_extract(
    video_path: str,
    sample_fps: int,
    max_seconds: float,
) -> dict:
    """Two-pass orchestration: cheap pass-1 wrist trajectory → stroke
    detection → heavy pass-2 on candidate windows only.

    Goal: ~10-15× compute reduction on long match-tape clips. A 5-min
    clip at 30fps has 9000 frames; the heavy YOLO+RTMPose pipeline runs
    at ~30ms/frame on Modal T4, so single-pass = ~270s of inference,
    over the 600s Modal timeout and well past the user's patience.
    Real-world tennis tape is ~70-85% non-stroke time (walking between
    points, ball pickup, talking) — running the heavy pipeline on those
    frames is wasted compute and the user never sees skeletons there.

    Pass 1: ~8fps wrist-only YOLO+RTMPose-s. ~75% cost reduction vs.
        the heavy pass at the same fps (no joint angles, no racket
        detector, no Kalman tracker, no remap), and ~73% vs. heavy
        pass at the default 30fps tier (8fps vs 30fps × 1.0 cost).
        For a 5-min clip: ~5min × 8fps × ~10ms = ~24s on T4.

    Pass 2: existing chunked-parallel YOLO+RTMPose-s on ONLY the frames
        inside candidate windows. Typical candidate-window coverage on
        match tape: 15-25% of total clip duration. Same per-frame cost,
        ~70-85% fewer frames.

    Combined: ~24s pass 1 + ~50s pass 2 ≈ ~75s total, vs ~270s for
    single pass. Real-world speedup ~3.5×, with the headroom to grow
    (an RTMPose-S → -M upgrade for pass 2 would still come in under
    single-pass timing).

    Falls back to single-pass when:
      * Clip duration < TWO_PASS_MIN_DURATION_SEC (overhead not worth it)
      * Pass 1 detected fewer than MIN_STROKES_FOR_TWO_PASS strokes
        (either a true low-stroke-count clip or pass 1 missed strokes;
        either way, single-pass is the safe fallback)
      * Pass 1 raised an exception (we log and continue with single-pass)
    """
    import time as _time

    # Probe the video for duration. Same autorotate path the heavy
    # extractor uses so the duration matches what _extract_with_rtmpose
    # will see internally.
    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0.0
    cap.release()

    # Short-clip fast path. The two-pass overhead (pass-1 model load
    # + wrist scan + stroke detect) costs ~5-10s on cold Modal even on
    # a tiny clip; that's a net loss if the clip itself is only 15s of
    # heavy single-pass work. The boundary matches the lib/poseExtraction
    # adaptive-sampling first tier so users see consistent behavior
    # across browser and server paths.
    if duration_sec < TWO_PASS_MIN_DURATION_SEC:
        print(
            f"[extract:two-pass] short clip ({duration_sec:.1f}s < "
            f"{TWO_PASS_MIN_DURATION_SEC:.0f}s) — skipping pass 1, "
            f"running single-pass directly"
        )
        result = _extract_with_rtmpose(video_path, sample_fps, max_seconds)
        result["pass_summary"] = {
            "mode": "single-pass-short-clip",
            "duration_sec": round(duration_sec, 1),
            "pass1_ms": 0,
            "pass2_ms": None,  # captured by inference_ms in caller
            "candidate_count": None,
        }
        return result

    # Pass 1: wrist trajectory.
    t_pass1 = _time.perf_counter()
    try:
        timestamps_ms, wrists, p1_video_fps, _p1_duration = _run_wrist_pass(
            video_path,
            pass1_fps=PASS1_SAMPLE_FPS,
        )
    except Exception as e:  # noqa: BLE001
        # If pass 1 blew up (rare — usually a transient YOLO/RTMPose
        # init failure), don't take the user down with us. Fall back to
        # single-pass and log loudly.
        print(f"[extract:two-pass] pass 1 failed: {e} — falling back to single-pass")
        result = _extract_with_rtmpose(video_path, sample_fps, max_seconds)
        result["pass_summary"] = {
            "mode": "single-pass-pass1-failed",
            "duration_sec": round(duration_sec, 1),
            "pass1_error": str(e),
        }
        return result
    pass1_ms = round((_time.perf_counter() - t_pass1) * 1000, 1)

    # Detect candidate stroke windows from the wrist series.
    candidate_windows = _detect_strokes_from_wrist(
        timestamps_ms, wrists, video_fps=p1_video_fps
    )
    candidate_windows = _merge_windows(candidate_windows)

    # Clip windows to the actual frame range (the detector pads peaks
    # by pre/post_pad_ms which can extend beyond the video on edge
    # peaks).
    if total_frames > 0:
        candidate_windows = [
            (max(0, s), min(total_frames - 1, e), max(0, min(total_frames - 1, p)))
            for s, e, p in candidate_windows
            if s < total_frames
        ]

    candidate_count = len(candidate_windows)
    print(
        f"[extract:two-pass] pass 1 ({pass1_ms}ms): "
        f"{candidate_count} candidate stroke windows"
    )

    # Few-strokes fallback. Either a true low-stroke clip (in which
    # case single-pass cost on a ~30-60s clip is acceptable), or pass 1
    # missed strokes. Either way, falling back to single-pass is the
    # safer choice — we don't want to silently drop a real stroke
    # because pass 1 mis-tracked the wrist on a tough clip.
    if candidate_count < MIN_STROKES_FOR_TWO_PASS:
        print(
            f"[extract:two-pass] only {candidate_count} candidates "
            f"(< {MIN_STROKES_FOR_TWO_PASS}) — falling back to single-pass"
        )
        result = _extract_with_rtmpose(video_path, sample_fps, max_seconds)
        result["pass_summary"] = {
            "mode": "single-pass-few-strokes",
            "duration_sec": round(duration_sec, 1),
            "pass1_ms": pass1_ms,
            "candidate_count": candidate_count,
        }
        return result

    # Pass 2: heavy extraction restricted to candidate windows. The
    # ThreadPoolExecutor inside _extract_with_rtmpose batches frames
    # 16 at a time across all candidate windows in a SINGLE invocation
    # — sequential per-stroke calls would each pay the chunk-pool
    # spin-up overhead, so we batch them.
    t_pass2 = _time.perf_counter()
    result = _extract_with_rtmpose(
        video_path,
        sample_fps,
        max_seconds,
        candidate_windows=candidate_windows,
    )
    pass2_ms = round((_time.perf_counter() - t_pass2) * 1000, 1)

    result["pass_summary"] = {
        "mode": "two-pass",
        "duration_sec": round(duration_sec, 1),
        "pass1_ms": pass1_ms,
        "pass2_ms": pass2_ms,
        "candidate_count": candidate_count,
        "candidate_windows": candidate_windows,
    }
    return result
