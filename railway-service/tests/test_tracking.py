"""Unit tests for the single-object trackers.

Tracker logic is pure data — no frames, no YOLO — so these tests run fast
and deterministically. The screenshot bug that motivated this file was
YOLO picking a small high-conf background figure over the large foreground
player; the cold-start and cold-stay-locked tests below are direct
regression guards for that exact failure mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking import PersonTracker, RacketTracker, iou  # noqa: E402


IMG_W, IMG_H = 1920, 1080


# ---------------------------------------------------------------------------
# iou helper
# ---------------------------------------------------------------------------


class TestIou:
    def test_identical_bboxes_iou_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_bboxes_iou_is_zero(self):
        assert iou((0, 0, 10, 10), (100, 100, 120, 120)) == 0.0

    def test_half_overlap_iou(self):
        # Two 10x10 bboxes, 5px horizontal overlap -> 50 / (100+100-50) = 1/3
        result = iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert abs(result - (50 / 150)) < 1e-6

    def test_zero_area_bbox_returns_zero(self):
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# PersonTracker: cold start — the regression that motivated this file
# ---------------------------------------------------------------------------


class TestPersonTrackerColdStart:
    def test_empty_candidates_returns_none(self):
        tracker = PersonTracker()
        assert tracker.update([], IMG_W, IMG_H) is None
        assert not tracker.is_locked

    def test_picks_large_centered_bbox_over_small_edge_bbox_even_with_lower_conf(self):
        """The exact screenshot bug: YOLO returned a small, high-confidence
        bbox on a background figure and a larger, lower-confidence bbox on
        the foreground player. raw-max-conf picked the ghost; the scored
        picker must pick the player."""
        # Foreground player: ~500x1000 centered, conf 0.55
        player = (700.0, 40.0, 1200.0, 1040.0, 0.55)
        # Background ghost: 60x120, upper-left corner, conf 0.95
        ghost = (50.0, 50.0, 110.0, 170.0, 0.95)

        tracker = PersonTracker()
        pick = tracker.update([ghost, player], IMG_W, IMG_H)

        assert pick is not None
        # bbox x1 near 700 = player; near 50 = ghost.
        assert pick[0] == 700.0, "tracker picked the background ghost"
        assert tracker.is_locked

    def test_penalizes_implausible_aspect_ratio(self):
        """A wide low-aspect bbox (fence / banner / bench) loses to a
        tall person-shaped bbox even at similar area + centrality."""
        # Horizontal banner: 800x120 centered, conf 0.9
        banner = (560.0, 480.0, 1360.0, 600.0, 0.9)
        # Player: 300x750 centered, conf 0.6
        player = (810.0, 165.0, 1110.0, 915.0, 0.6)
        tracker = PersonTracker()
        pick = tracker.update([banner, player], IMG_W, IMG_H)
        assert pick is not None
        assert pick[0] == 810.0, "tracker picked the horizontal banner"


# ---------------------------------------------------------------------------
# PersonTracker: association (the second half of the screenshot bug —
# once locked, the tracker must not switch to a different person)
# ---------------------------------------------------------------------------


class TestPersonTrackerAssociation:
    def test_stays_locked_when_background_detection_appears(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker()
        tracker.update([player], IMG_W, IMG_H)

        # Next frame: player moves slightly, plus a random background
        # detection appears at high confidence. The tracker must keep
        # following the player by IoU, not switch to the ghost.
        player_moved = (720.0, 45.0, 1220.0, 1045.0, 0.55)
        ghost = (50.0, 50.0, 110.0, 170.0, 0.99)

        pick = tracker.update([ghost, player_moved], IMG_W, IMG_H)
        assert pick is not None
        # x1 near 700–720 means we followed the player; near 50 means we
        # jumped to the ghost.
        assert pick[0] > 600, f"tracker switched to ghost (x1={pick[0]})"

    def test_coasts_when_no_candidate_clears_iou_gate(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker()
        tracker.update([player], IMG_W, IMG_H)

        # Frame with only far-away detections (no IoU overlap with player).
        far = (50.0, 50.0, 110.0, 170.0, 0.9)
        pick = tracker.update([far], IMG_W, IMG_H)

        assert pick is not None
        # We should still be showing the player's last known position.
        assert pick[0] == 700.0

    def test_resets_after_max_coast_frames(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker(max_coast_frames=3)
        tracker.update([player], IMG_W, IMG_H)

        # Coast for 4 frames with only off-gate detections. On the 4th
        # coast the tracker should reset and cold-start on whatever is
        # available.
        far = (50.0, 50.0, 110.0, 170.0, 0.9)
        # Coast 1, 2, 3 — still returning player's last known.
        for _ in range(3):
            tracker.update([far], IMG_W, IMG_H)
        # Frame 4 exceeds max_coast_frames=3, triggers reset + cold-start
        # on the only candidate (far). Now tracker locks onto it.
        pick = tracker.update([far], IMG_W, IMG_H)
        assert pick is not None
        assert pick[0] == 50.0, "tracker did not cold-start after timeout"

    def test_ema_smooths_jittery_detections(self):
        """When the IoU match is accepted, the returned bbox should be a
        smoothed blend of the old reference and the new detection — not
        the raw detection. This prevents per-frame YOLO jitter from
        making the player-bbox (and thus the pose crop) twitch."""
        ref = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker(ema_alpha=0.7)
        tracker.update([ref], IMG_W, IMG_H)

        # New detection shifted by +100px. EMA-smoothed x1 should land
        # at 0.7*800 + 0.3*700 = 770, not 800 and not 700.
        shifted = (800.0, 40.0, 1300.0, 1040.0, 0.6)
        pick = tracker.update([shifted], IMG_W, IMG_H)
        assert pick is not None
        assert 765 < pick[0] < 775, (
            f"ema smoothing did not blend (expected ~770, got {pick[0]})"
        )

    def test_reset_clears_lock(self):
        tracker = PersonTracker()
        tracker.update([(700.0, 40.0, 1200.0, 1040.0, 0.6)], IMG_W, IMG_H)
        assert tracker.is_locked
        tracker.reset()
        assert not tracker.is_locked


# ---------------------------------------------------------------------------
# RacketTracker: cold start, update, coast, acceleration-following
# ---------------------------------------------------------------------------


class TestRacketTrackerBasics:
    def test_cold_no_detection_returns_none(self):
        tracker = RacketTracker()
        assert tracker.update(None, timestamp_ms=0) is None
        assert not tracker.is_locked

    def test_cold_with_detection_arms_filter(self):
        tracker = RacketTracker()
        out = tracker.update(
            {"x": 0.5, "y": 0.5, "confidence": 0.6}, timestamp_ms=0
        )
        assert out is not None
        assert out["x"] == 0.5
        assert out["y"] == 0.5
        assert tracker.is_locked

    def test_reject_outlier_outside_association_radius(self):
        tracker = RacketTracker(association_radius=0.1)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.8}, timestamp_ms=0)
        # Next frame: a detection 0.3 normalized away — outside the gate.
        # Tracker should coast (return a prediction near 0.5), not jump.
        out = tracker.update(
            {"x": 0.8, "y": 0.8, "confidence": 0.9}, timestamp_ms=33
        )
        assert out is not None
        assert abs(out["x"] - 0.5) < 0.02, (
            f"tracker jumped to outlier (x={out['x']})"
        )

    def test_accepts_in_gate_detection(self):
        tracker = RacketTracker(association_radius=0.15)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.8}, timestamp_ms=0)
        out = tracker.update(
            {"x": 0.55, "y": 0.50, "confidence": 0.8}, timestamp_ms=33
        )
        assert out is not None
        # EMA-ish: filter should land between prediction (0.5 + small v) and
        # measurement (0.55). Not strictly 0.55.
        assert 0.5 < out["x"] < 0.56


class TestRacketTrackerCoast:
    def test_coasts_through_missed_detection(self):
        tracker = RacketTracker(max_coast_frames=4)
        # Two detections to establish velocity.
        tracker.update({"x": 0.4, "y": 0.5, "confidence": 0.8}, timestamp_ms=0)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.8}, timestamp_ms=33)

        # Next frame: no detection. Filter should coast and emit a position
        # ahead of 0.5 (positive vx).
        coast = tracker.update(None, timestamp_ms=66)
        assert coast is not None
        assert coast["x"] > 0.5, (
            f"filter did not project forward during coast (x={coast['x']})"
        )

    def test_gives_up_after_max_coast_frames(self):
        tracker = RacketTracker(max_coast_frames=3)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.8}, timestamp_ms=0)
        # Three coasts allowed.
        for i in range(3):
            out = tracker.update(None, timestamp_ms=33 * (i + 1))
            assert out is not None, f"filter gave up early at coast {i}"
        # Fourth consecutive miss trips the reset.
        final = tracker.update(None, timestamp_ms=33 * 4)
        assert final is None
        assert not tracker.is_locked

    def test_confidence_decays_on_coast(self):
        tracker = RacketTracker(conf_decay=0.8, conf_floor=0.0)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.9}, timestamp_ms=0)
        c1 = tracker.update(None, timestamp_ms=33)
        c2 = tracker.update(None, timestamp_ms=66)
        assert c1 is not None and c2 is not None
        assert c2["confidence"] < c1["confidence"] < 0.9

    def test_confidence_floor_triggers_reset(self):
        tracker = RacketTracker(conf_decay=0.5, conf_floor=0.4, max_coast_frames=100)
        tracker.update({"x": 0.5, "y": 0.5, "confidence": 0.7}, timestamp_ms=0)
        # 0.7 × 0.5 = 0.35, below floor 0.4 -> reset.
        out = tracker.update(None, timestamp_ms=33)
        assert out is None
        assert not tracker.is_locked


class TestRacketTrackerAcceleration:
    """The advisor's key test: synthetic trajectory approximating a
    forehand contact — point accelerates from 0 → ~1500 px/s in ~100ms
    (≈ 0.8 normalized/s at 1920px wide) then reverses. The filter must
    follow without lagging more than ~2 frames behind ground truth.
    A too-small sigma_a silently fails this check.
    """

    def _simulate_swing(self, fps: float = 30.0) -> list[tuple[float, float]]:
        """Return a list of (timestamp_ms, true_x) for a synthetic 1D
        motion: ramp up to peak velocity, then reverse. Y held at 0.5.
        """
        dt = 1.0 / fps
        # Peak velocity in normalized units/sec (approx 1500 px/s on 1920 wide)
        v_peak = 0.8
        # Acceleration phase = 100ms; deceleration+reverse = 100ms
        t_accel = 0.1
        t_decel = 0.1
        a_up = v_peak / t_accel       # 8 norm/s²
        a_down = -2 * v_peak / t_decel  # goes from +v_peak to -v_peak
        samples = []
        x = 0.2
        v = 0.0
        t = 0.0
        for _ in range(12):  # ~400ms of motion
            if t < t_accel:
                a = a_up
            elif t < t_accel + t_decel:
                a = a_down
            else:
                a = 0
            v += a * dt
            x += v * dt
            samples.append((t * 1000, x))
            t += dt
        return samples

    def test_follows_forehand_acceleration_within_two_frames_of_lag(self):
        tracker = RacketTracker(sigma_a=8.0, sigma_m=0.01, association_radius=0.2)
        samples = self._simulate_swing()

        # Simulate perfect YOLO: every frame has a detection at the true
        # position with high confidence. The filter SHOULD track it
        # closely because the association gate is generous and the
        # Kalman update trusts the measurement.
        max_error = 0.0
        for ts_ms, x_true in samples:
            out = tracker.update(
                {"x": x_true, "y": 0.5, "confidence": 0.85}, ts_ms
            )
            assert out is not None
            max_error = max(max_error, abs(out["x"] - x_true))

        # Allow up to 0.05 normalized units of lag (about 2 frames worth
        # of peak-velocity travel). If this fails, sigma_a is probably
        # too small.
        assert max_error < 0.05, (
            f"filter lagged {max_error:.4f} normalized units behind swing — "
            f"increase sigma_a"
        )

    def test_fills_gap_during_motion_blur_plateau(self):
        """Classic failure mode: YOLO sees the racket before the swing
        and after, but misses 2-3 frames in the middle at peak velocity
        (motion blur). Filter must bridge those frames with sensible
        predictions, not emit None."""
        tracker = RacketTracker(max_coast_frames=5)
        samples = self._simulate_swing()

        # Drop detections for frames 3, 4, 5 — those fall in the
        # acceleration plateau.
        for i, (ts_ms, x_true) in enumerate(samples):
            det: Optional[dict]
            if 3 <= i <= 5:
                det = None
            else:
                det = {"x": x_true, "y": 0.5, "confidence": 0.85}
            out = tracker.update(det, ts_ms)
            assert out is not None, f"filter returned None at frame {i}"


# Forward-reference for Optional used in the last test
from typing import Optional  # noqa: E402
