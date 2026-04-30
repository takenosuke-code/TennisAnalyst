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

# Above this clip duration the cheap motion pre-pass kicks in so we
# only pose-extract within motion windows instead of every sampled
# frame of e.g. a 5-min match (most of which is the player walking
# between points). Below the threshold the pre-pass overhead isn't
# worth it.
MOTION_PREPASS_MIN_DURATION_SEC = 90.0


def _detect_motion_windows(
    video_path: str,
    sample_fps: int = 5,
    threshold_percentile: float = 70.0,
    padding_sec: float = 1.0,
    min_window_sec: float = 0.5,
) -> list[tuple[int, int]] | None:
    """Cheap motion-peak detection over the whole video.

    Goal: identify the frame ranges where the player is actively
    swinging so the expensive YOLO+RTMPose pipeline only runs on those
    windows. A 5-min match clip might be ~80% non-swing time (walking
    between points, ball retrieval, talking to partner) — paying full
    inference cost on those frames is wasted compute and the user
    doesn't see skeletons there anyway, since the multi-shot grid
    only surfaces detected swings.

    Algorithm:
      1. Sample frames at `sample_fps` (default 5fps).
      2. Downsample each to 320x240 grayscale, frame-diff against the
         previous, take mean abs-difference as the motion score.
      3. Threshold at the `threshold_percentile` of motion scores.
      4. Group consecutive above-threshold frames into windows with
         `padding_sec` of slack on each side so a swing's wind-up and
         follow-through (lower motion than peak contact) still land
         inside the window.

    Cost: ~1ms per sampled frame on CPU. A 5-min clip at 5fps motion
    sampling is 1500 frames × ~1ms = ~1.5s of pre-pass overhead in
    exchange for avoiding pose extraction on ~70% of frames.

    Returns a list of (start_frame, end_frame) tuples in original-
    video frame indices, or None when no usable motion was detected
    (caller falls back to processing every frame). Returning None is
    intentionally conservative — we'd rather over-process than
    silently drop swings.
    """
    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(1, int(fps / sample_fps))

    motion_scores: list[tuple[int, float]] = []
    prev_gray = None
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                small = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    score = float(np.mean(diff))
                    motion_scores.append((frame_idx, score))
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    if len(motion_scores) < 3:
        return None

    scores = np.array([s for _, s in motion_scores])
    threshold_score = float(np.percentile(scores, threshold_percentile))

    # If the threshold itself is tiny, the video is essentially
    # uniform-motion (camera shake, a static scene with no clear peaks)
    # — don't filter. Process whole video.
    if threshold_score < 1.0:
        return None

    motion_frames = [f for f, s in motion_scores if s >= threshold_score]
    if not motion_frames:
        return None

    pad = int(padding_sec * fps)
    min_window_frames = int(min_window_sec * fps)
    windows: list[tuple[int, int]] = []
    cur_start = max(0, motion_frames[0] - pad)
    cur_end = motion_frames[0] + pad

    for f in motion_frames[1:]:
        if f - pad <= cur_end:
            # Adjacent or overlapping — extend current window.
            cur_end = f + pad
        else:
            # Gap — close current window if it meets the minimum length.
            if cur_end - cur_start >= min_window_frames:
                windows.append((cur_start, cur_end))
            cur_start = max(0, f - pad)
            cur_end = f + pad
    if cur_end - cur_start >= min_window_frames:
        windows.append((cur_start, cur_end))

    return windows or None


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
    video_path: str, sample_fps: int, max_seconds: float
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

    # Motion pre-pass for long clips: detect motion windows up front so
    # we only pose-extract within them. The cheap motion scan runs in
    # ~1.5s for a 5-min clip and typically reduces the inference set to
    # ~20-30% of the original frames (most match footage is non-swing
    # dead time). For shorter clips the pre-pass overhead isn't worth
    # it — the threshold matches the existing adaptive-sampling tier.
    motion_windows: list[tuple[int, int]] | None = None
    if duration_sec_est > MOTION_PREPASS_MIN_DURATION_SEC:
        import time as _time
        t_motion = _time.perf_counter()
        # Release our cap so the motion-prepass cap doesn't fight for
        # the same video file handle on platforms that gate concurrent
        # opens (rare but observed on some FS combinations).
        cap.release()
        motion_windows = _detect_motion_windows(video_path)
        motion_ms = round((_time.perf_counter() - t_motion) * 1000, 1)
        if motion_windows:
            total_window_frames = sum(e - s for s, e in motion_windows)
            total_window_sec = total_window_frames / video_fps
            print(
                f"[extract:rtmpose] motion pre-pass ({motion_ms}ms): "
                f"{len(motion_windows)} windows covering {total_window_sec:.1f}s "
                f"of {duration_sec_est:.1f}s "
                f"({100 * total_window_sec / duration_sec_est:.0f}%)"
            )
        else:
            print(
                f"[extract:rtmpose] motion pre-pass ({motion_ms}ms): "
                f"no usable windows; processing all sampled frames"
            )
        # Reopen for the main loop.
        cap = _open_video_autorotated(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot reopen video after motion prepass: {video_path}")

    if duration_sec_est > 120:
        effective_sample_fps = min(sample_fps, 8)
    elif duration_sec_est > 60:
        effective_sample_fps = min(sample_fps, 12)
    elif duration_sec_est > 30:
        effective_sample_fps = min(sample_fps, 18)
    else:
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

    def _in_motion_window(idx: int) -> bool:
        """True when `idx` is inside any pre-detected motion window
        (or when no pre-pass ran, in which case every frame is in)."""
        if motion_windows is None:
            return True
        for s, e in motion_windows:
            if s <= idx <= e:
                return True
        return False

    chunk: list[tuple[int, int, np.ndarray]] = []
    motion_skipped = 0
    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frame > 0 and frame_index >= max_frame:
                break

            if frame_index % frame_interval == 0:
                if not _in_motion_window(frame_index):
                    motion_skipped += 1
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

    if motion_windows is not None:
        print(
            f"[extract:rtmpose] motion pre-pass skipped {motion_skipped} sampled frames "
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
    video_path: str, sample_fps: int = 30, max_seconds: float = 0
) -> dict:
    """Extract pose keypoints from a video file via YOLO11n + RTMPose-s.

    Args:
        max_seconds: Stop after this many seconds. 0 = process entire video.
    """
    return _extract_with_rtmpose(video_path, sample_fps, max_seconds)
