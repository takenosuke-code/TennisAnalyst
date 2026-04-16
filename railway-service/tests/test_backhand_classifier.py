"""Unit tests for the BackhandClassifier.

Tests cover true positives (one-handed and two-handed backhands for both
right-handed and left-handed players), true negatives (forehand, serve,
slice, standing still), two-handed vs one-handed detection, edge cases,
and clean-shot validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the railway-service root is on sys.path so the shot_classifiers
# package can be imported directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from shot_classifiers.backhand import BackhandClassifier


# ---------------------------------------------------------------------------
# MediaPipe 33-point landmark names (subset used for reference)
# ---------------------------------------------------------------------------
_LANDMARK_NAMES: dict[int, str] = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# Total number of landmarks in the MediaPipe Pose model.
_NUM_LANDMARKS = 33


# ---------------------------------------------------------------------------
# Frame generation helper
# ---------------------------------------------------------------------------

def make_frames(
    *,
    num_frames: int = 20,
    fps: float = 30.0,
    # Per-landmark overrides: dict mapping landmark index to
    # (start_x, end_x, start_y, end_y) tuples.  Values are linearly
    # interpolated across the frame sequence.
    landmark_overrides: dict[int, tuple[float, float, float, float]] | None = None,
    # Global defaults for landmarks not in overrides.
    default_x: float = 0.5,
    default_y: float = 0.5,
    default_z: float = 0.0,
    default_visibility: float = 0.9,
    # Optional joint_angles per frame (constant across all frames).
    joint_angles: dict[str, float] | None = None,
    # Landmarks to mark as invisible (visibility=0).
    invisible_landmarks: set[int] | None = None,
) -> list[dict]:
    """Generate a sequence of synthetic MediaPipe pose frames.

    Each frame contains 33 landmarks.  By default every landmark sits at
    ``(default_x, default_y)`` with full visibility.  Use
    ``landmark_overrides`` to linearly animate specific landmarks from
    ``(start_x, start_y)`` to ``(end_x, end_y)`` across the sequence.
    """
    overrides = landmark_overrides or {}
    invisible = invisible_landmarks or set()
    dt_ms = 1000.0 / fps
    frames: list[dict] = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)  # 0.0 -> 1.0
        landmarks: list[dict] = []

        for lm_id in range(_NUM_LANDMARKS):
            if lm_id in invisible:
                landmarks.append({
                    "id": lm_id,
                    "name": _LANDMARK_NAMES.get(lm_id, f"landmark_{lm_id}"),
                    "x": default_x,
                    "y": default_y,
                    "z": default_z,
                    "visibility": 0.0,
                })
                continue

            if lm_id in overrides:
                sx, ex, sy, ey = overrides[lm_id]
                x = sx + (ex - sx) * t
                y = sy + (ey - sy) * t
            else:
                x = default_x
                y = default_y

            landmarks.append({
                "id": lm_id,
                "name": _LANDMARK_NAMES.get(lm_id, f"landmark_{lm_id}"),
                "x": x,
                "y": y,
                "z": default_z,
                "visibility": default_visibility,
            })

        frames.append({
            "frame_index": i,
            "timestamp_ms": i * dt_ms,
            "landmarks": landmarks,
            "joint_angles": dict(joint_angles) if joint_angles else {},
        })

    return frames


def _right_handed_one_handed_backhand_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Landmark overrides for a canonical right-handed one-handed backhand.

    The lead wrist for a right-hander is the LEFT wrist (index 15).
    It moves from x=0.3 to x=0.7 (left-to-right), at hip height (y~0.6).

    The RIGHT wrist (index 16, racket hand) must have an x-range >= the
    left wrist so that ``detect_handedness`` returns "right".  We give it
    a slightly larger swing (0.20 -> 0.75) while keeping it far from the
    lead wrist in y so it is NOT detected as two-handed.

    Body landmarks are placed to simulate a behind-the-player view with
    realistic proportions.
    """
    return {
        # Nose: centred, head height
        0:  (0.50, 0.50, 0.25, 0.25),
        # Left shoulder
        11: (0.55, 0.55, 0.35, 0.35),
        # Right shoulder
        12: (0.45, 0.45, 0.35, 0.35),
        # Left elbow (follows lead wrist trajectory roughly)
        13: (0.35, 0.60, 0.45, 0.45),
        # Right elbow (racket arm, swings with right wrist)
        14: (0.25, 0.70, 0.42, 0.42),
        # Left wrist (LEAD) -- moves left to right = backhand direction
        15: (0.30, 0.70, 0.55, 0.55),
        # Right wrist (racket hand) -- larger range so handedness = "right",
        # offset in y so distance > two-handed threshold
        16: (0.20, 0.75, 0.42, 0.42),
        # Left hip
        23: (0.52, 0.52, 0.65, 0.65),
        # Right hip
        24: (0.48, 0.48, 0.65, 0.65),
    }


def _right_handed_two_handed_backhand_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a right-handed two-handed backhand.

    Both wrists move together from left to right, staying very close.
    The right wrist has a slightly wider x-range so ``detect_handedness``
    returns "right".  Both wrists share nearly the same (x, y) trajectory
    so they stay within the two-handed proximity threshold.
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.33, 0.68, 0.45, 0.45),
        14: (0.29, 0.72, 0.45, 0.45),
        # Left wrist (LEAD)
        15: (0.30, 0.70, 0.50, 0.50),
        # Right wrist -- slightly wider range (0.29->0.72 = 0.43 vs 0.40),
        # same y so wrists stay close together
        16: (0.29, 0.72, 0.50, 0.50),
        23: (0.52, 0.52, 0.65, 0.65),
        24: (0.48, 0.48, 0.65, 0.65),
    }


def _left_handed_one_handed_backhand_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a left-handed one-handed backhand.

    For a left-handed player the lead wrist is the RIGHT wrist (index 16).
    It moves from right to left (x decreases) -- the mirror of a right-handed
    backhand.

    For ``detect_handedness`` to return "left", the LEFT wrist (racket hand
    for a lefty) must have a *larger* x-range than the right wrist.  The
    left wrist swings with a wide arc while the right wrist (lead) makes the
    cross-body lateral motion.
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.75, 0.25, 0.42, 0.42),   # left elbow (racket arm, big swing)
        14: (0.65, 0.35, 0.45, 0.45),   # right elbow follows lead wrist
        # Left wrist (racket hand for lefty) -- large range so handedness = "left",
        # offset in y from lead wrist so NOT two-handed
        15: (0.80, 0.25, 0.42, 0.42),
        # Right wrist (LEAD) -- moves right-to-left = left-handed backhand
        16: (0.70, 0.30, 0.55, 0.55),
        23: (0.48, 0.48, 0.65, 0.65),
        24: (0.52, 0.52, 0.65, 0.65),
    }


def _left_handed_two_handed_backhand_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a left-handed two-handed backhand.

    Both wrists travel right-to-left together.  The LEFT wrist must have
    a strictly larger x-range than the RIGHT wrist so ``detect_handedness``
    returns "left".  Both wrists share nearly the same trajectory so they
    stay within the two-handed proximity threshold.
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.72, 0.28, 0.45, 0.45),
        14: (0.68, 0.32, 0.45, 0.45),
        # Left wrist -- wider range (0.72->0.28 = 0.44) so left_range > right_range
        15: (0.72, 0.28, 0.50, 0.50),
        # Right wrist (LEAD) -- slightly narrower range (0.70->0.30 = 0.40),
        # same y so wrists stay close together
        16: (0.70, 0.30, 0.50, 0.50),
        23: (0.48, 0.48, 0.65, 0.65),
        24: (0.52, 0.52, 0.65, 0.65),
    }


def _forehand_right_handed_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a right-handed forehand (should NOT be classified as backhand).

    For a right-handed forehand the dominant (right) wrist moves right-to-left.
    The left wrist has a *smaller* range so ``detect_handedness`` returns "right".
    The lead wrist for a backhand check would be the left wrist -- but it has
    a *negative* displacement (it moves right-to-left along with the body
    rotation), which should fail the lateral-direction check.
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.55, 0.45, 0.45, 0.45),
        14: (0.65, 0.35, 0.45, 0.45),
        # Left wrist -- small range, moves right-to-left (wrong direction for BH)
        15: (0.55, 0.45, 0.55, 0.55),
        # Right wrist -- large range right-to-left (forehand swing)
        16: (0.70, 0.30, 0.55, 0.55),
        23: (0.52, 0.52, 0.65, 0.65),
        24: (0.48, 0.48, 0.65, 0.65),
    }


def _serve_motion_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a serve motion (wrist goes above the head).

    Set up as a right-handed player (right wrist has larger range) whose
    lead wrist (LEFT, index 15) moves left-to-right (correct backhand
    direction) BUT the lead wrist's y-position goes well above the nose,
    triggering serve rejection.

    In MediaPipe image coords: y=0 is top of frame, so smaller y = higher.
    Nose is at y=0.25.  The lead wrist goes from y=0.35 down to y=0.10,
    which means min(wrist_y) = 0.10 < nose_y * 0.85 = 0.2125 => serve.
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.35, 0.60, 0.30, 0.10),
        14: (0.25, 0.70, 0.42, 0.42),
        # Left wrist (LEAD for RH) -- moves left-to-right but rises above head
        15: (0.30, 0.70, 0.35, 0.10),
        # Right wrist -- wider range so handedness = "right"
        16: (0.20, 0.75, 0.50, 0.50),
        23: (0.52, 0.52, 0.65, 0.65),
        24: (0.48, 0.48, 0.65, 0.65),
    }


def _slice_motion_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for a slice motion (pronounced high-to-low path).

    Set up as a right-handed player with correct lateral direction, but
    the lead wrist drops significantly from high to low.

    The lead wrist (LEFT, index 15) moves left-to-right AND drops:
    - x: 0.30 -> 0.70 (correct backhand direction, range=0.40)
    - y: 0.35 -> 0.75 (high-to-low drop of 0.40)

    Right wrist has range >= left wrist for handedness = "right".

    Torso height = |hip_y - shoulder_y| = |0.65 - 0.35| = 0.30
    drop / torso = 0.40 / 0.30 = 1.33 >> 0.45 threshold => slice rejection
    """
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.35, 0.55, 0.35, 0.60),
        14: (0.25, 0.70, 0.42, 0.42),
        # Left wrist (LEAD) -- correct lateral direction but big vertical drop
        15: (0.30, 0.70, 0.35, 0.75),
        # Right wrist -- wider range so handedness = "right"
        16: (0.20, 0.75, 0.50, 0.50),
        23: (0.52, 0.52, 0.65, 0.65),
        24: (0.48, 0.48, 0.65, 0.65),
    }


def _standing_still_overrides() -> dict[int, tuple[float, float, float, float]]:
    """Overrides for standing still (no significant motion)."""
    return {
        0:  (0.50, 0.50, 0.25, 0.25),
        11: (0.55, 0.55, 0.35, 0.35),
        12: (0.45, 0.45, 0.35, 0.35),
        13: (0.45, 0.45, 0.45, 0.45),
        14: (0.55, 0.55, 0.45, 0.45),
        15: (0.45, 0.45, 0.55, 0.55),
        16: (0.55, 0.55, 0.55, 0.55),
        23: (0.52, 0.52, 0.65, 0.65),
        24: (0.48, 0.48, 0.65, 0.65),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> BackhandClassifier:
    """Instantiate a fresh BackhandClassifier."""
    return BackhandClassifier()


# =========================================================================
# TRUE POSITIVE TESTS
# =========================================================================


class TestTruePositives:
    """Frames that SHOULD be classified as backhands with positive confidence."""

    def test_right_handed_one_handed_backhand(self, classifier: BackhandClassifier) -> None:
        """A canonical right-handed one-handed backhand should be detected.

        The left wrist (lead for RH) moves from x=0.3 to x=0.7 at hip
        height -- a clear left-to-right trajectory.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0, "Expected positive confidence for RH one-handed backhand"
        assert result.shot_type == "backhand"
        assert result.handedness == "right"
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "one_handed"

    def test_right_handed_two_handed_backhand(self, classifier: BackhandClassifier) -> None:
        """A right-handed two-handed backhand: both wrists travel together.

        Both left and right wrists start near x=0.3 and end near x=0.7,
        staying within the two-handed proximity threshold.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_two_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0, "Expected positive confidence for RH two-handed backhand"
        assert result.shot_type == "backhand"
        assert result.handedness == "right"
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "two_handed"

    def test_left_handed_one_handed_backhand(self, classifier: BackhandClassifier) -> None:
        """A left-handed one-handed backhand mirrors the right-handed version.

        The lead wrist (RIGHT wrist for LH) moves from x=0.7 to x=0.3
        (right-to-left).
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_left_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0, "Expected positive confidence for LH one-handed backhand"
        assert result.shot_type == "backhand"
        assert result.handedness == "left"
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "one_handed"

    def test_left_handed_two_handed_backhand(self, classifier: BackhandClassifier) -> None:
        """A left-handed two-handed backhand: both wrists move right-to-left."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_left_handed_two_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0, "Expected positive confidence for LH two-handed backhand"
        assert result.shot_type == "backhand"
        assert result.handedness == "left"
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "two_handed"

    def test_backhand_from_side_angle(self, classifier: BackhandClassifier) -> None:
        """A backhand viewed from a side camera angle.

        Side angle is characterised by narrow shoulder width relative to
        torso height (ratio < 0.4).  We achieve this by bringing the
        shoulders close in x while keeping a tall torso.
        """
        overrides = _right_handed_one_handed_backhand_overrides()
        # Narrow the shoulders to simulate a side view
        overrides[11] = (0.51, 0.51, 0.30, 0.30)  # left shoulder
        overrides[12] = (0.49, 0.49, 0.30, 0.30)  # right shoulder
        # Keep hips spread and lower to maintain tall torso
        overrides[23] = (0.52, 0.52, 0.70, 0.70)
        overrides[24] = (0.48, 0.48, 0.70, 0.70)

        frames = make_frames(num_frames=20, landmark_overrides=overrides)
        result = classifier.classify(frames)

        assert result.confidence > 0, "Backhand should be detected from side camera angle"
        assert result.camera_angle == "side"

    def test_backhand_from_front_angle(self, classifier: BackhandClassifier) -> None:
        """A backhand viewed from a front camera angle.

        Front angle: wide shoulders, high nose visibility.
        """
        overrides = _right_handed_one_handed_backhand_overrides()
        # Wide shoulders (ratio > 0.8)
        overrides[11] = (0.65, 0.65, 0.35, 0.35)
        overrides[12] = (0.35, 0.35, 0.35, 0.35)
        # Keep hips proportional
        overrides[23] = (0.55, 0.55, 0.60, 0.60)
        overrides[24] = (0.45, 0.45, 0.60, 0.60)
        # Nose well visible (default visibility is 0.9, which is > 0.6)
        overrides[0] = (0.50, 0.50, 0.25, 0.25)

        frames = make_frames(num_frames=20, landmark_overrides=overrides)
        result = classifier.classify(frames)

        assert result.confidence > 0, "Backhand should be detected from front camera angle"
        assert result.camera_angle == "front"


# =========================================================================
# TRUE NEGATIVE TESTS
# =========================================================================


class TestTrueNegatives:
    """Frames that must NOT be classified as backhands (confidence == 0)."""

    def test_forehand_rejected(self, classifier: BackhandClassifier) -> None:
        """A right-handed forehand (wrist moves right-to-left) must not register.

        For a right-handed player the lead wrist (LEFT) for backhand analysis
        should move left-to-right.  In a forehand, the left wrist does not
        exhibit this motion.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_forehand_right_handed_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Forehand should not be classified as backhand, got confidence={result.confidence}"
        )

    def test_serve_rejected(self, classifier: BackhandClassifier) -> None:
        """A serve motion with wrist above head height must be rejected.

        The serve puts the wrist well above the nose y-coordinate, which
        should trigger the serve-rejection heuristic.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_serve_motion_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Serve should not be classified as backhand, got confidence={result.confidence}"
        )

    def test_slice_rejected(self, classifier: BackhandClassifier) -> None:
        """A slice (high-to-low path) must be rejected.

        The wrist drops significantly from near shoulder height to below
        hip height -- the vertical displacement exceeds the slice threshold.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_slice_motion_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Slice should not be classified as backhand, got confidence={result.confidence}"
        )

    def test_standing_still_rejected(self, classifier: BackhandClassifier) -> None:
        """A player standing still produces zero lateral displacement.

        With no wrist movement, the minimum lateral displacement threshold
        should not be reached.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_standing_still_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Standing still should not be classified as backhand, got confidence={result.confidence}"
        )


# =========================================================================
# TWO-HANDED vs ONE-HANDED DETECTION
# =========================================================================


class TestTwoHandedDetection:
    """Verify the classifier correctly distinguishes grip type."""

    def test_two_handed_wrists_close(self, classifier: BackhandClassifier) -> None:
        """When both wrists stay within the proximity threshold, two-handed is reported."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_two_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "two_handed"

    def test_one_handed_wrists_far_apart(self, classifier: BackhandClassifier) -> None:
        """When the off-hand wrist is far from the lead wrist, one-handed is reported."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "one_handed"

    def test_two_handed_left_handed(self, classifier: BackhandClassifier) -> None:
        """Two-handed detection also works for left-handed players."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_left_handed_two_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0
        assert result.phase_timestamps is not None
        assert result.phase_timestamps.get("variant") == "two_handed"


# =========================================================================
# EDGE CASES
# =========================================================================


class TestEdgeCases:
    """Edge cases: short/long sequences, missing landmarks, low visibility."""

    def test_too_few_frames_rejected(self, classifier: BackhandClassifier) -> None:
        """Sequences shorter than the minimum frame count should be rejected.

        The classifier requires at least 8 frames (_MIN_FRAMES) to analyse.
        """
        frames = make_frames(
            num_frames=5,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Too-short sequences must be rejected"

    def test_exactly_min_frames(self, classifier: BackhandClassifier) -> None:
        """A sequence of exactly _MIN_FRAMES (8) should still be analysed."""
        frames = make_frames(
            num_frames=8,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        # Should not be auto-rejected for length.  Whether it classifies
        # positively depends on the actual motion -- our overrides produce
        # enough displacement to pass.
        assert result.confidence > 0, "Exactly 8 frames should be long enough to classify"

    def test_long_sequence_classifies(self, classifier: BackhandClassifier) -> None:
        """A longer sequence (60 frames, ~2 seconds) should still classify correctly."""
        frames = make_frames(
            num_frames=60,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0, "60-frame backhand should still be detected"
        assert result.shot_type == "backhand"

    def test_missing_wrist_landmarks_rejected(self, classifier: BackhandClassifier) -> None:
        """If the lead wrist landmarks are invisible, classification should fail.

        Setting visibility to 0 for both wrists means lm_xy returns None,
        so no wrist positions are collected.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
            invisible_landmarks={15, 16},  # both wrists invisible
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Missing wrist landmarks must cause rejection"

    def test_missing_hip_landmarks_rejected(self, classifier: BackhandClassifier) -> None:
        """If hip landmarks are invisible, the classifier cannot compute hip centres.

        This should cause rejection because the lateral-direction check
        requires hip_centers.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
            invisible_landmarks={23, 24},  # both hips invisible
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Missing hip landmarks must cause rejection"

    def test_low_visibility_landmarks(self, classifier: BackhandClassifier) -> None:
        """Landmarks with visibility at or below 0.3 are treated as absent.

        The base get_landmark() function returns None when visibility <= 0.3.
        """
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
            default_visibility=0.2,  # below the 0.3 threshold
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            "Low-visibility landmarks should be treated as missing"
        )


# =========================================================================
# CLEAN VALIDATION
# =========================================================================


class TestCleanValidation:
    """Tests for the validate_clean method."""

    def test_single_backhand_is_clean(self, classifier: BackhandClassifier) -> None:
        """A single smooth backhand within a normal duration should be clean."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0
        assert result.is_clean is True, "A single smooth backhand should be clean"

    def test_multi_shot_not_clean(self, classifier: BackhandClassifier) -> None:
        """A sequence with many direction reversals should not be clean.

        We simulate this by creating rapid oscillations in the lead wrist's
        x-position.  The validate_clean method rejects sequences whose
        reversal count exceeds 40% of the wrist position count.
        """
        import math as _math

        overrides = dict(_right_handed_one_handed_backhand_overrides())

        frames: list[dict] = []
        # Create rapid zigzag wrist motion: 10 cycles over 40 frames.
        # This produces ~20 reversals, well above the 40% * 40 = 16 threshold.
        num_frames = 40
        dt_ms = 1000.0 / 30.0
        for i in range(num_frames):
            t = i / (num_frames - 1)
            landmarks: list[dict] = []
            for lm_id in range(_NUM_LANDMARKS):
                if lm_id in overrides:
                    sx, ex, sy, ey = overrides[lm_id]
                    if lm_id == 15:  # lead wrist -- rapid zigzag
                        # 10 full oscillations via sin
                        x = 0.3 + 0.4 * abs(_math.sin(t * 10 * _math.pi))
                        y = sy + (ey - sy) * t
                    else:
                        x = sx + (ex - sx) * t
                        y = sy + (ey - sy) * t
                else:
                    x, y = 0.5, 0.5

                landmarks.append({
                    "id": lm_id,
                    "name": _LANDMARK_NAMES.get(lm_id, f"landmark_{lm_id}"),
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "visibility": 0.9,
                })

            frames.append({
                "frame_index": i,
                "timestamp_ms": i * dt_ms,
                "landmarks": landmarks,
                "joint_angles": {},
            })

        result = classifier.classify(frames)

        # The classifier may or may not give positive confidence (the overall
        # displacement could still be positive).  But it should NOT be clean.
        if result.confidence > 0:
            assert result.is_clean is False, (
                "Multi-directional zigzag motion should not be classified as a clean backhand"
            )

    def test_too_long_duration_not_clean(self, classifier: BackhandClassifier) -> None:
        """A sequence longer than _MAX_DURATION_MS (4000ms) should not be clean."""
        # 200 frames at 30fps = ~6.6 seconds > 4 seconds max
        frames = make_frames(
            num_frames=200,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        if result.confidence > 0:
            assert result.is_clean is False, (
                "A 6.6s sequence should not be classified as a clean single backhand"
            )

    def test_very_short_duration_not_clean(self, classifier: BackhandClassifier) -> None:
        """A sequence shorter than _MIN_DURATION_MS (200ms) should not be clean.

        8 frames at 30fps spans ~233ms which barely passes, so we use 100fps
        to compress the time to 70ms (7 intervals * 10ms).
        """
        frames = make_frames(
            num_frames=8,
            fps=100.0,  # 8 frames at 100fps => 70ms total << 200ms
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        # At 70ms total duration, this is below _MIN_DURATION_MS=200.
        if result.confidence > 0:
            assert result.is_clean is False, (
                "A 70ms sequence should be too short to be a clean backhand"
            )


# =========================================================================
# CLASSIFIER INTERFACE
# =========================================================================


class TestClassifierInterface:
    """Verify the classifier implements the BaseShotClassifier contract."""

    def test_shot_type_property(self, classifier: BackhandClassifier) -> None:
        """The shot_type property must return 'backhand'."""
        assert classifier.shot_type == "backhand"

    def test_result_fields_present(self, classifier: BackhandClassifier) -> None:
        """Every ClassificationResult must have all expected fields."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert hasattr(result, "shot_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_clean")
        assert hasattr(result, "camera_angle")
        assert hasattr(result, "handedness")
        assert hasattr(result, "phase_timestamps")

    def test_zero_confidence_on_empty_frames(self, classifier: BackhandClassifier) -> None:
        """An empty frame list should return confidence 0."""
        result = classifier.classify([])

        assert result.confidence == 0.0
        assert result.shot_type == "backhand"

    def test_confidence_bounded_zero_to_one(self, classifier: BackhandClassifier) -> None:
        """Confidence should always be in [0.0, 1.0]."""
        for overrides_fn in [
            _right_handed_one_handed_backhand_overrides,
            _right_handed_two_handed_backhand_overrides,
            _forehand_right_handed_overrides,
            _serve_motion_overrides,
            _standing_still_overrides,
        ]:
            frames = make_frames(num_frames=20, landmark_overrides=overrides_fn())
            result = classifier.classify(frames)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of bounds for {overrides_fn.__name__}"
            )

    def test_phase_timestamps_populated_on_positive(self, classifier: BackhandClassifier) -> None:
        """When classification is positive, phase timestamps should be present."""
        frames = make_frames(
            num_frames=20,
            landmark_overrides=_right_handed_one_handed_backhand_overrides(),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0
        assert result.phase_timestamps is not None
        assert "preparation_end" in result.phase_timestamps
        assert "contact" in result.phase_timestamps
        assert "follow_through_start" in result.phase_timestamps
        assert "variant" in result.phase_timestamps
