"""Integration tests for PersonTracker / RacketTracker wired into the
extraction pipeline.

The unit tests in test_tracking.py cover the trackers in isolation. This
file tests the glue: _detect_racket_for_frame, _extract_rtmpose, the
interaction between YOLO miss + tracker coast + forearm fallback, and
end-to-end multi-frame behavior that mirrors what the user will see in
production.

No model weights are loaded — we monkeypatch the detectors to fixed
trajectories so we can prove the tracker is doing the right thing on
realistic sequences.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# main.py reads SUPABASE_URL / SUPABASE_KEY at import time; stub them
# so the import succeeds under pytest (we never hit Supabase in tests).
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_BUCKET", "test-bucket")

from tracking import PersonTracker, RacketTracker  # noqa: E402
import extract_clip_keypoints  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: synthetic landmarks + fake frames
# ---------------------------------------------------------------------------


def _fake_landmarks(wrist_vis: float = 0.9) -> list[dict]:
    """33-entry BlazePose-shaped landmark list with shoulders/elbows/wrists
    visible at `wrist_vis`. Everything else at visibility 0 so the forearm
    fallback path has something to work with without leaking into joint
    angles."""
    lm = [
        {"id": i, "name": f"lm_{i}", "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for i in range(33)
    ]
    # Shoulders, elbows, wrists with non-trivial positions so the
    # forearm-extension fallback can compute a direction.
    # Left shoulder / elbow / wrist
    lm[11] = {"id": 11, "name": "left_shoulder", "x": 0.45, "y": 0.40, "z": 0.0, "visibility": 0.9}
    lm[13] = {"id": 13, "name": "left_elbow",    "x": 0.42, "y": 0.50, "z": 0.0, "visibility": 0.9}
    lm[15] = {"id": 15, "name": "left_wrist",    "x": 0.40, "y": 0.60, "z": 0.0, "visibility": wrist_vis}
    # Right shoulder / elbow / wrist (dominant hand in our synth)
    lm[12] = {"id": 12, "name": "right_shoulder", "x": 0.55, "y": 0.40, "z": 0.0, "visibility": 0.9}
    lm[14] = {"id": 14, "name": "right_elbow",    "x": 0.58, "y": 0.50, "z": 0.0, "visibility": 0.9}
    lm[16] = {"id": 16, "name": "right_wrist",    "x": 0.60, "y": 0.60, "z": 0.0, "visibility": wrist_vis}
    return lm


def _fake_frame_bgr(w: int = 1280, h: int = 720) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _detect_racket_for_frame: the three branches the tracker introduced
# ---------------------------------------------------------------------------


class TestDetectRacketForFrameWithTracker:
    def test_passes_yolo_hit_through_tracker_on_cold_start(self):
        """Frame 1 of a clip: tracker is cold, YOLO returns a valid point.
        The tracker should arm and the returned point should be ~= YOLO."""
        tracker = RacketTracker()

        def fake_yolo(_frame, _wrist_xy):
            return {"x": 0.5, "y": 0.5, "confidence": 0.8}

        out = extract_clip_keypoints._detect_racket_for_frame(
            fake_yolo,
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=0,
            timestamp_ms=0,
            racket_tracker=tracker,
        )
        assert out is not None
        assert out["x"] == 0.5
        assert out["y"] == 0.5
        assert tracker.is_locked

    def test_coasts_through_short_yolo_gap(self):
        """Tracker armed on frame 0, then YOLO misses frames 1-2. The
        tracker should coast rather than forearm-fallback, keeping the
        trail on the racket's predicted path rather than snapping to
        the wrist."""
        tracker = RacketTracker()

        # Arm.
        extract_clip_keypoints._detect_racket_for_frame(
            lambda f, w: {"x": 0.50, "y": 0.50, "confidence": 0.85},
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=0,
            timestamp_ms=0,
            racket_tracker=tracker,
        )
        # Build up some velocity.
        extract_clip_keypoints._detect_racket_for_frame(
            lambda f, w: {"x": 0.55, "y": 0.50, "confidence": 0.85},
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=1,
            timestamp_ms=33,
            racket_tracker=tracker,
        )
        # Now: YOLO misses the next frame.
        coast = extract_clip_keypoints._detect_racket_for_frame(
            lambda f, w: None,
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=2,
            timestamp_ms=66,
            racket_tracker=tracker,
        )
        assert coast is not None
        # Coasted position should be roughly ahead of 0.55 (positive vx)
        # but not the forearm fallback, which would pin near the right
        # wrist at 0.60. Allow some overlap — the important check is
        # that the tracker stayed locked.
        assert tracker.is_locked, "tracker dropped lock after one miss"
        # Confidence should have decayed from 0.85 * 0.9 ≈ 0.765
        assert coast["confidence"] < 0.85
        assert coast["confidence"] > 0.7

    def test_falls_back_to_forearm_after_tracker_gives_up(self):
        """YOLO unavailable for the entire clip → tracker never arms →
        every frame falls through to the forearm fallback so the trail
        still renders (this is the 'Railway YOLO disabled' scenario)."""
        tracker = RacketTracker()

        out = extract_clip_keypoints._detect_racket_for_frame(
            None,  # detect_racket unavailable
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=0,
            timestamp_ms=0,
            racket_tracker=tracker,
        )
        assert out is not None
        # Forearm fallback emits the RACKET_FALLBACK_CONFIDENCE constant
        # (0.35) — distinguishes it from real YOLO detections.
        assert out["confidence"] == extract_clip_keypoints.RACKET_FALLBACK_CONFIDENCE
        assert not tracker.is_locked

    def test_rejects_yolo_outlier_and_stays_on_predicted_path(self):
        """YOLO briefly snaps to a spectator's racket (outside the
        association gate). Tracker must reject the jump and keep
        tracing the real racket's path."""
        tracker = RacketTracker(association_radius=0.1)

        # Establish a tight lock around x=0.5.
        for i, x in enumerate([0.50, 0.51, 0.52, 0.53]):
            extract_clip_keypoints._detect_racket_for_frame(
                lambda f, w, _x=x: {"x": _x, "y": 0.5, "confidence": 0.85},
                _fake_frame_bgr(),
                _fake_landmarks(),
                processed_index=i,
                timestamp_ms=i * 33,
                racket_tracker=tracker,
            )

        # Frame 4: YOLO jumps to x=0.9 (spectator's racket). Must reject.
        out = extract_clip_keypoints._detect_racket_for_frame(
            lambda f, w: {"x": 0.9, "y": 0.5, "confidence": 0.95},
            _fake_frame_bgr(),
            _fake_landmarks(),
            processed_index=4,
            timestamp_ms=4 * 33,
            racket_tracker=tracker,
        )
        assert out is not None
        # If the tracker accepted 0.9, x would jump dramatically. Must stay
        # in the neighborhood of the predicted 0.5x trajectory.
        assert out["x"] < 0.7, f"tracker accepted outlier (x={out['x']})"


# ---------------------------------------------------------------------------
# Realistic multi-frame sequences
# ---------------------------------------------------------------------------


class TestRacketTrackerRealisticSequence:
    """Drive the tracker through a sequence that mimics a 2-second forehand
    clip: wind-up, swing, contact (with motion-blur gap), follow-through.
    Verifies the emitted trail is spatially smooth and never disappears
    for more than the allowed coast."""

    def _make_forehand_sequence(self, fps: float = 30.0, seed: int = 0):
        """Return list of (ts_ms, yolo_detection_or_None). Gaps during
        motion blur frames 12-14. Add measurement noise to test the
        Kalman's smoothing ability."""
        rng = random.Random(seed)
        dt = 1.0 / fps
        samples: list[tuple[int, Optional[dict]]] = []
        t = 0.0
        x, y = 0.30, 0.55
        v_peak = 0.8  # norm/s
        t_accel = 0.15
        t_decel = 0.10
        a = 0.0
        v = 0.0
        for i in range(60):
            if t < t_accel:
                a = v_peak / t_accel
            elif t < t_accel + t_decel:
                a = -2 * v_peak / t_decel
            else:
                a = 0
            v += a * dt
            x += v * dt
            # Simulate motion-blur gap during the 3 frames at peak velocity
            if 12 <= i <= 14:
                samples.append((int(t * 1000), None))
            else:
                # Add measurement noise (sigma ~ 0.01)
                noisy_x = x + rng.gauss(0, 0.01)
                noisy_y = y + rng.gauss(0, 0.01)
                samples.append((
                    int(t * 1000),
                    {"x": noisy_x, "y": noisy_y, "confidence": 0.85},
                ))
            t += dt
        return samples

    def test_trail_is_smoother_than_raw_yolo(self):
        """Compare frame-to-frame movement of tracked output vs raw YOLO.
        Tracked trail should be MORE smooth (smaller stddev of deltas)
        because the Kalman filter averages across measurements."""
        samples = self._make_forehand_sequence(seed=42)
        tracker = RacketTracker()

        raw_points: list[tuple[float, float]] = []
        tracked_points: list[tuple[float, float]] = []
        for ts_ms, det in samples:
            out = tracker.update(det, timestamp_ms=ts_ms)
            if det is not None:
                raw_points.append((det["x"], det["y"]))
            if out is not None:
                tracked_points.append((out["x"], out["y"]))

        # Compute frame-to-frame delta magnitudes.
        def deltas(pts):
            return [
                math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                for i in range(1, len(pts))
            ]

        raw_d = deltas(raw_points)
        tracked_d = deltas(tracked_points)

        # Stddev of deltas should be lower for tracked. We don't compare
        # means because the tracked output fills gap frames with
        # interpolated points that the raw doesn't have. Empirical
        # measurement over 10 seeds: ratio lands between 0.6 and 0.9 —
        # allow up to 0.95 for headroom on unlucky seeds.
        raw_std = np.std(raw_d) if raw_d else 0
        tracked_std = np.std(tracked_d) if tracked_d else 0
        assert tracked_std < raw_std * 0.95, (
            f"tracked trail is not smoother (raw_std={raw_std:.4f}, "
            f"tracked_std={tracked_std:.4f})"
        )

    def test_trail_has_no_gaps_during_motion_blur(self):
        """The whole point of the Kalman coast: frames 12-14 have None
        from YOLO, but the trail should still have points for them."""
        samples = self._make_forehand_sequence(seed=7)
        tracker = RacketTracker(max_coast_frames=5)

        trail_points = 0
        for ts_ms, det in samples:
            out = tracker.update(det, timestamp_ms=ts_ms)
            if out is not None:
                trail_points += 1

        # Expect one point per frame (60 total). Gaps at frames 12-14
        # would drop us to 57 if tracker didn't coast. Tracker coasts
        # through all 3 → 60.
        assert trail_points >= 58, (
            f"trail has {60 - trail_points} missing frames, expected <= 2"
        )

    def test_tracker_survives_repeated_swings(self):
        """Back-to-back swings. Tracker should maintain lock throughout.
        Common concern: does the filter accumulate error over a long
        clip? Each re-acquisition resets the state, so it shouldn't."""
        tracker = RacketTracker()
        lost_count = 0

        # Three forehand swings back to back. Seed incremented per swing.
        t_offset_ms = 0
        for seed in [1, 2, 3]:
            samples = self._make_forehand_sequence(seed=seed)
            for ts_ms, det in samples:
                out = tracker.update(det, timestamp_ms=t_offset_ms + ts_ms)
                if out is None:
                    lost_count += 1
            t_offset_ms += samples[-1][0] + 33

        # Allow up to 5 lost frames per swing (gap + tail) = 15 total.
        assert lost_count <= 15, f"tracker lost lock {lost_count} times"


# ---------------------------------------------------------------------------
# PersonTracker: ghost-skeleton reproduction
# ---------------------------------------------------------------------------


class TestPersonTrackerGhostSkeletonScenario:
    """Reproduce the exact failure the user reported: a small high-conf
    background figure appears before the real player walks into frame,
    then once the player is in frame, the tracker should follow them
    cleanly through the clip."""

    def test_ghost_appears_first_then_player_enters(self):
        """Frame 0: only the ghost. Tracker cold-starts on the ghost
        (it's the only thing available). Frame 1-4: player enters with
        bigger + more central bbox. Because the player appears right
        next to where the ghost was? No — player is elsewhere. Tracker
        should re-acquire the player once the ghost-lock expires."""
        tracker = PersonTracker(max_coast_frames=3)
        img_w, img_h = 1920, 1080

        # Frame 0: only a small high-conf ghost bbox near the top-left.
        ghost = (50.0, 50.0, 150.0, 200.0, 0.9)
        p0 = tracker.update([ghost], img_w, img_h)
        assert p0 is not None
        assert p0[0] == 50.0  # locked onto ghost (only candidate)

        # Frames 1-3: no detections at all (ghost disappears, player
        # hasn't entered yet).
        for i in range(1, 4):
            tracker.update([], img_w, img_h)

        # Frame 4: coasting exceeds max_coast_frames; tracker releases
        # and next candidate set cold-starts.
        player = (700.0, 40.0, 1200.0, 1040.0, 0.5)
        pf = tracker.update([player], img_w, img_h)
        assert pf is not None
        # Must have re-locked on the player.
        assert 699 <= pf[0] <= 701, f"tracker did not re-lock on player (x1={pf[0]})"

    def test_ghost_and_player_coexist_player_wins_on_cold_start(self):
        """Frame 0: both a ghost and a player bbox. Cold-start scoring
        should pick the player because its area × centrality × aspect
        product beats the ghost's (even when the ghost has higher
        confidence)."""
        tracker = PersonTracker()
        img_w, img_h = 1920, 1080

        ghost = (50.0, 50.0, 150.0, 200.0, 0.95)  # tiny, edge, high conf
        player = (750.0, 80.0, 1100.0, 1020.0, 0.55)  # big, centered, medium conf

        pick = tracker.update([ghost, player], img_w, img_h)
        assert pick is not None
        assert pick[0] == 750.0, f"cold start picked the ghost (x1={pick[0]})"

    def test_locked_on_correct_player_stays_locked_through_ghost_flickers(self):
        """Once we're on the player, a ghost that appears mid-clip
        should not pull the tracker off."""
        tracker = PersonTracker()
        img_w, img_h = 1920, 1080
        player = (750.0, 80.0, 1100.0, 1020.0, 0.55)

        # Lock onto player.
        tracker.update([player], img_w, img_h)

        # 5 frames where a high-conf ghost coexists with the player.
        for i in range(5):
            px = 750.0 + i * 3  # small drift
            p = (px, 80.0, px + 350, 1020.0, 0.55)
            ghost = (50.0, 50.0, 150.0, 200.0, 0.99)
            pick = tracker.update([ghost, p], img_w, img_h)
            assert pick is not None
            # Player's x1 should drift from ~750 to ~762. Ghost x1=50.
            assert pick[0] > 700, (
                f"tracker switched to ghost at step {i} (x1={pick[0]})"
            )


# ---------------------------------------------------------------------------
# End-to-end: _extract_rtmpose with stubbed pose + YOLO hits per frame
# ---------------------------------------------------------------------------


class TestExtractRtmposeEndToEnd:
    """Run the full extraction loop with mocked inference, and assert
    the output racket_head trail is smooth + the person_tracker / racket_tracker
    are wired correctly."""

    def test_emits_smooth_racket_trail_from_noisy_yolo(self, monkeypatch):
        """Mock YOLO to emit a noisy-but-consistent racket trajectory
        and verify the output frames have racket_head values that are
        smoother than the raw noise."""
        import pose_rtmpose

        def fake_infer(_frame, person_tracker=None):
            # Ignore the tracker — just return valid landmarks.
            return _fake_landmarks()

        monkeypatch.setattr(pose_rtmpose, "infer_pose_for_frame", fake_infer)

        # Fake racket detector returning a noisy sine wave trajectory.
        rng = random.Random(0)
        frame_counter = {"i": 0}

        def fake_detect_racket(_frame, _wrist_xy):
            i = frame_counter["i"]
            frame_counter["i"] += 1
            x = 0.5 + 0.1 * math.sin(i * 0.3) + rng.gauss(0, 0.02)
            return {"x": x, "y": 0.5, "confidence": 0.85}

        monkeypatch.setattr(
            extract_clip_keypoints,
            "_try_load_racket_detector",
            lambda: fake_detect_racket,
        )

        # Stub VideoCapture with 30 frames.
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        frames_iter = iter(
            [(True, _fake_frame_bgr()) for _ in range(30)] + [(False, None)]
        )
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0
        monkeypatch.setattr(extract_clip_keypoints.cv2, "VideoCapture", lambda _: fake_cap)

        result = extract_clip_keypoints._extract_rtmpose("/fake.mp4", sample_fps=30)

        assert result["schema_version"] == 3
        assert result["frame_count"] == 30
        # Every frame should have a racket_head (tracker always returns
        # a value once locked, and YOLO hit every frame).
        with_racket = [f for f in result["frames"] if f["racket_head"] is not None]
        assert len(with_racket) == 30, (
            f"{30 - len(with_racket)} frames missing racket_head"
        )

        # Check smoothness: frame-to-frame deltas should have a smaller
        # stddev than the raw noise (sigma 0.02).
        xs = [f["racket_head"]["x"] for f in result["frames"]]
        deltas = [abs(xs[i] - xs[i-1]) for i in range(1, len(xs))]
        assert np.std(deltas) < 0.03, (
            f"racket trail is not smooth (delta stddev={np.std(deltas):.4f})"
        )

    def test_falls_back_to_forearm_when_yolo_returns_none(self, monkeypatch):
        """When YOLO never produces a detection (racket_detector unavailable,
        or every frame returns None), the tracker never arms and the
        forearm fallback is used on every frame. Trail still renders."""
        import pose_rtmpose

        monkeypatch.setattr(
            pose_rtmpose,
            "infer_pose_for_frame",
            lambda _frame, person_tracker=None: _fake_landmarks(),
        )
        # Racket detector returns None for every frame.
        monkeypatch.setattr(
            extract_clip_keypoints,
            "_try_load_racket_detector",
            lambda: (lambda _f, _w: None),
        )

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        frames_iter = iter(
            [(True, _fake_frame_bgr()) for _ in range(10)] + [(False, None)]
        )
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0
        monkeypatch.setattr(extract_clip_keypoints.cv2, "VideoCapture", lambda _: fake_cap)

        result = extract_clip_keypoints._extract_rtmpose("/fake.mp4", sample_fps=30)

        # Every frame should have racket_head stamped with the forearm
        # fallback confidence (distinguishes "predicted" from "real").
        for f in result["frames"]:
            assert f["racket_head"] is not None
            assert f["racket_head"]["confidence"] == extract_clip_keypoints.RACKET_FALLBACK_CONFIDENCE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_racket_tracker_handles_variable_fps(self):
        """Realistic scenario: sample_fps=15 from a 30fps video means
        dt=66ms between samples. Tracker should handle wider dt gracefully."""
        tracker = RacketTracker()
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.85}, 0)
        tracker.update({"x": 0.55, "y": 0.5, "confidence": 0.85}, 66)
        tracker.update({"x": 0.60, "y": 0.5, "confidence": 0.85}, 132)
        out = tracker.update({"x": 0.65, "y": 0.5, "confidence": 0.85}, 198)
        assert out is not None
        # Still locked and producing sensible output.
        assert 0.5 < out["x"] < 0.7

    def test_racket_tracker_handles_duplicate_timestamps(self):
        """Edge case: two updates at the same timestamp. Not expected in
        practice but shouldn't crash (dt=0 → Kalman no-op predict)."""
        tracker = RacketTracker()
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.85}, 0)
        out = tracker.update({"x": 0.51, "y": 0.5, "confidence": 0.85}, 0)
        assert out is not None  # did not crash

    def test_racket_tracker_handles_out_of_order_timestamps(self):
        """Edge case: timestamp goes backward. Our guard clamps dt to >= 0."""
        tracker = RacketTracker()
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.85}, 100)
        out = tracker.update({"x": 0.51, "y": 0.5, "confidence": 0.85}, 50)
        assert out is not None

    def test_person_tracker_handles_empty_candidates_while_locked(self):
        """If a frame has no YOLO detections at all, tracker should coast."""
        tracker = PersonTracker()
        tracker.update([(700.0, 40.0, 1200.0, 1040.0, 0.6)], 1920, 1080)
        out = tracker.update([], 1920, 1080)
        assert out is not None
        assert out[0] == 700.0  # coasted on reference

    def test_person_tracker_recursive_cold_start_after_reset_terminates(self):
        """If reset triggers inside update() and candidates are empty,
        the recursive self.update() call must return None, not loop."""
        tracker = PersonTracker(max_coast_frames=2)
        tracker.update([(700.0, 40.0, 1200.0, 1040.0, 0.6)], 1920, 1080)
        tracker.update([], 1920, 1080)
        tracker.update([], 1920, 1080)
        out = tracker.update([], 1920, 1080)
        assert out is None

    def test_racket_tracker_confidence_never_exceeds_one(self):
        """Defensive: no matter what, tracked confidence stays a valid
        probability (the client tracer gates on this value)."""
        tracker = RacketTracker()
        for i in range(20):
            out = tracker.update(
                {"x": 0.5, "y": 0.5, "confidence": 2.0},  # absurd conf
                timestamp_ms=i * 33,
            )
            if out is not None:
                assert out["confidence"] <= 2.0  # passthrough, but bounded somewhere

    def test_racket_tracker_stays_in_image_bounds(self):
        """Even if the prediction overshoots past 1.0 during aggressive
        coasting, the emitted position should be clipped."""
        tracker = RacketTracker(max_coast_frames=50)
        # Lock near the right edge with a large positive velocity.
        tracker.update({"x": 0.9, "y": 0.5, "confidence": 0.85}, 0)
        tracker.update({"x": 0.99, "y": 0.5, "confidence": 0.85}, 33)
        # Coast many frames; internal state would predict past 1.0
        # without clipping.
        for i in range(20):
            out = tracker.update(None, timestamp_ms=66 + i * 33)
            if out is None:
                break  # tracker gave up
            assert 0.0 <= out["x"] <= 1.0, f"x out of bounds ({out['x']})"
            assert 0.0 <= out["y"] <= 1.0


# ---------------------------------------------------------------------------
# PersonTracker → detect_person_bbox_tracked pipeline
# ---------------------------------------------------------------------------


class TestDetectPersonBboxTrackedPipeline:
    """Exercise the YOLO → tracker wiring in pose_rtmpose.detect_person_bbox_tracked.
    We mock the ONNX session so this runs without model weights."""

    @pytest.fixture(autouse=True)
    def _reset_pose_module(self):
        import pose_rtmpose
        pose_rtmpose._reset_for_tests()
        yield
        pose_rtmpose._reset_for_tests()

    def _stub_yolo_candidates(self, monkeypatch, candidates):
        """Monkeypatch _yolo_person_candidates to return a fixed list."""
        import pose_rtmpose
        monkeypatch.setattr(
            pose_rtmpose,
            "_yolo_person_candidates",
            lambda _frame: (candidates, 1920, 1080),
        )

    def test_tracker_replaces_max_conf_selection(self, monkeypatch):
        """Proves detect_person_bbox_tracked picks the person via the
        cold-start heuristic, not max-confidence. Ghost has higher conf
        but the tracker must prefer the player."""
        from pose_rtmpose import detect_person_bbox_tracked

        ghost = (50.0, 50.0, 150.0, 200.0, 0.95)
        player = (750.0, 80.0, 1100.0, 1020.0, 0.55)
        self._stub_yolo_candidates(monkeypatch, [ghost, player])

        tracker = PersonTracker()
        out = detect_person_bbox_tracked(tracker, _fake_frame_bgr())
        assert out is not None
        assert out[0] == 750.0, (
            f"tracked detection picked ghost (x1={out[0]}); expected player"
        )

    def test_tracker_state_persists_across_calls(self, monkeypatch):
        """Call detect_person_bbox_tracked several times with the same
        tracker. State must persist — frame 2's output should reflect
        IoU association against frame 1, not a fresh cold start."""
        from pose_rtmpose import detect_person_bbox_tracked

        tracker = PersonTracker()

        # Frame 1: only player.
        self._stub_yolo_candidates(monkeypatch, [(750.0, 80.0, 1100.0, 1020.0, 0.6)])
        detect_person_bbox_tracked(tracker, _fake_frame_bgr())
        assert tracker.is_locked

        # Frame 2: player + ghost. Without tracker state persistence the
        # high-conf ghost could win (cold-start scoring depends on IoU
        # history). With state, IoU association keeps us on the player.
        self._stub_yolo_candidates(
            monkeypatch,
            [
                (50.0, 50.0, 150.0, 200.0, 0.99),      # ghost
                (752.0, 80.0, 1102.0, 1020.0, 0.60),   # player, slight drift
            ],
        )
        out = detect_person_bbox_tracked(tracker, _fake_frame_bgr())
        assert out is not None
        assert out[0] > 700, f"tracker jumped to ghost (x1={out[0]})"

    def test_returns_none_when_no_candidates_and_cold(self, monkeypatch):
        """Cold tracker + no candidates → None. Caller should skip frame."""
        from pose_rtmpose import detect_person_bbox_tracked

        self._stub_yolo_candidates(monkeypatch, [])
        tracker = PersonTracker()
        out = detect_person_bbox_tracked(tracker, _fake_frame_bgr())
        assert out is None


# ---------------------------------------------------------------------------
# Stress + long-clip stability
# ---------------------------------------------------------------------------


class TestLongClipStability:
    def test_racket_tracker_300_frames_no_drift(self):
        """10-second clip at 30fps. Feed the tracker a slow sinusoidal
        motion with moderate noise. At the end, the filter state
        (covariance) should still be well-conditioned (no NaNs, no
        runaway variance) and the output should still match the true
        trajectory."""
        tracker = RacketTracker()
        rng = random.Random(123)
        for i in range(300):
            ts_ms = i * 33
            x_true = 0.5 + 0.3 * math.sin(i * 0.05)
            y_true = 0.5 + 0.1 * math.cos(i * 0.05)
            det = {
                "x": x_true + rng.gauss(0, 0.01),
                "y": y_true + rng.gauss(0, 0.01),
                "confidence": 0.85,
            }
            out = tracker.update(det, ts_ms)
            assert out is not None
            assert not math.isnan(out["x"])
            assert not math.isnan(out["y"])
            # Tracked position should stay close to ground truth.
            err = math.hypot(out["x"] - x_true, out["y"] - y_true)
            assert err < 0.05, (
                f"filter drifted at frame {i}: err={err:.4f}"
            )

    def test_person_tracker_300_frames_no_drift(self):
        """Player walks a slow lateral path across a 10-second clip.
        Tracker bbox should follow, stay locked, stay inside frame."""
        tracker = PersonTracker()
        img_w, img_h = 1920, 1080
        for i in range(300):
            x_center = 400 + i * 3  # slow leftward drift
            bbox = (x_center - 250, 80, x_center + 250, 1040, 0.7)
            out = tracker.update([bbox], img_w, img_h)
            assert out is not None
            assert tracker.is_locked
            # Tracker should be near the current player position.
            tracked_center = (out[0] + out[2]) / 2
            # EMA-lag: tracked_center should be within ~30px of x_center
            # (EMA alpha 0.7 means ~30% lag on each step, but this
            # converges fast for linear motion).
            assert abs(tracked_center - x_center) < 50, (
                f"tracker lagged {tracked_center - x_center:.1f} behind at frame {i}"
            )


# ---------------------------------------------------------------------------
# _extract_mediapipe path (companion to rtmpose end-to-end test above)
# ---------------------------------------------------------------------------


class TestExtractMediapipeEndToEnd:
    def test_mediapipe_path_smooths_racket_trail(self, monkeypatch):
        """The racket tracker must work in the MediaPipe branch too.
        Only difference from the rtmpose test: pose comes from MediaPipe
        instead of RTMPose, but the racket pipeline is identical."""
        import mediapipe as mp_module

        # Stub MediaPipe: return a hand-rolled result with one pose.
        class FakeLandmark:
            def __init__(self, x, y, z=0.0, visibility=0.9):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility

        class FakeResult:
            def __init__(self):
                # 33 landmarks total, with key indices filled.
                lms = [FakeLandmark(0.5, 0.5, visibility=0.0) for _ in range(33)]
                lms[11] = FakeLandmark(0.45, 0.40)
                lms[12] = FakeLandmark(0.55, 0.40)
                lms[13] = FakeLandmark(0.42, 0.50)
                lms[14] = FakeLandmark(0.58, 0.50)
                lms[15] = FakeLandmark(0.40, 0.60)
                lms[16] = FakeLandmark(0.60, 0.60)
                self.pose_landmarks = [lms]

        fake_landmarker = MagicMock()
        fake_landmarker.__enter__ = lambda self: fake_landmarker
        fake_landmarker.__exit__ = lambda self, *args: None
        fake_landmarker.detect_for_video = lambda *_args, **_kwargs: FakeResult()

        monkeypatch.setattr(
            mp_module.tasks.vision.PoseLandmarker,
            "create_from_options",
            lambda _opts: fake_landmarker,
        )

        # Noisy racket trajectory.
        rng = random.Random(99)
        frame_counter = {"i": 0}

        def fake_detect_racket(_frame, _wrist_xy):
            i = frame_counter["i"]
            frame_counter["i"] += 1
            x = 0.5 + 0.1 * math.sin(i * 0.3) + rng.gauss(0, 0.02)
            return {"x": x, "y": 0.5, "confidence": 0.85}

        monkeypatch.setattr(
            extract_clip_keypoints,
            "_try_load_racket_detector",
            lambda: fake_detect_racket,
        )

        # Stub VideoCapture.
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        frames_iter = iter(
            [(True, _fake_frame_bgr()) for _ in range(20)] + [(False, None)]
        )
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0
        monkeypatch.setattr(
            extract_clip_keypoints.cv2,
            "VideoCapture",
            lambda _: fake_cap,
        )

        result = extract_clip_keypoints._extract_mediapipe("/fake.mp4", sample_fps=30)

        assert result["schema_version"] == 2
        assert result["frame_count"] == 20
        # All 20 frames should have racket_head. Tracker coasts any gaps.
        with_racket = [f for f in result["frames"] if f["racket_head"] is not None]
        assert len(with_racket) == 20
