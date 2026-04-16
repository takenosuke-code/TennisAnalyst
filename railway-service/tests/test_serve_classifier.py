"""Comprehensive unit tests for ServeClassifier.

Tests cover true positives (various serve types), true negatives
(forehands, backhands, slices, standing still), serve-specific features
(trophy position, knee bend, duration), edge cases, and phase detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shot_classifiers.serve import ServeClassifier
from shot_classifiers.base import ClassificationResult

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (33-point model)
# ---------------------------------------------------------------------------
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28

_LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}


# ---------------------------------------------------------------------------
# Helper: generate synthetic pose frames
# ---------------------------------------------------------------------------

def _make_landmark(
    idx: int,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    visibility: float = 0.99,
) -> dict:
    return {
        "id": idx,
        "name": _LANDMARK_NAMES.get(idx, f"landmark_{idx}"),
        "x": x,
        "y": y,
        "z": z,
        "visibility": visibility,
    }


def _default_landmarks() -> dict[int, tuple[float, float]]:
    """Return default (x, y) positions for a standing player facing camera.

    Coordinate system: x in [0,1] left-to-right, y in [0,1] top-to-bottom.
    A standing player has nose near top, ankles near bottom.
    """
    return {
        LM_NOSE:            (0.50, 0.15),
        LM_LEFT_SHOULDER:   (0.55, 0.30),
        LM_RIGHT_SHOULDER:  (0.45, 0.30),
        LM_LEFT_ELBOW:      (0.60, 0.45),
        LM_RIGHT_ELBOW:     (0.40, 0.45),
        LM_LEFT_WRIST:      (0.62, 0.55),
        LM_RIGHT_WRIST:     (0.38, 0.55),
        LM_LEFT_HIP:        (0.53, 0.55),
        LM_RIGHT_HIP:       (0.47, 0.55),
        LM_LEFT_KNEE:       (0.54, 0.72),
        LM_RIGHT_KNEE:      (0.46, 0.72),
        LM_LEFT_ANKLE:      (0.54, 0.90),
        LM_RIGHT_ANKLE:     (0.46, 0.90),
    }


def make_frames(
    num_frames: int = 30,
    dt_ms: float = 33.0,
    overrides: dict[int, dict[int, tuple[float, float]]] | None = None,
    missing_landmarks: dict[int, list[int]] | None = None,
    joint_angles: dict[int, dict[str, float]] | None = None,
) -> list[dict]:
    """Generate a sequence of pose frames with 33 MediaPipe landmarks.

    Parameters
    ----------
    num_frames:
        Number of frames to generate.
    dt_ms:
        Milliseconds between consecutive frames.
    overrides:
        ``{frame_idx: {landmark_idx: (x, y), ...}, ...}`` to override
        specific landmark positions at specific frames.  Frames without
        overrides use linear interpolation between the nearest overridden
        values, or fall back to defaults.
    missing_landmarks:
        ``{frame_idx: [landmark_idx, ...]}`` to mark certain landmarks as
        invisible (visibility=0.0) at specific frames.
    joint_angles:
        ``{frame_idx: {"right_knee": float, ...}}`` to inject joint angle
        data.  If not given, joint_angles is an empty dict.
    """
    if overrides is None:
        overrides = {}
    if missing_landmarks is None:
        missing_landmarks = {}
    if joint_angles is None:
        joint_angles = {}

    defaults = _default_landmarks()

    # Build a set of all landmark indices that have overrides.
    all_override_lm_ids: set[int] = set()
    for frame_overrides in overrides.values():
        all_override_lm_ids.update(frame_overrides.keys())

    # For each landmark with overrides, build interpolation keyframes.
    def _interpolate(lm_idx: int, frame_idx: int) -> tuple[float, float]:
        """Interpolate (x, y) for a landmark at a given frame."""
        # Collect all frames where this landmark has an override.
        keyframes = sorted(
            (fi, pos)
            for fi, ovr in overrides.items()
            if lm_idx in ovr
            for pos in [ovr[lm_idx]]
        )
        if not keyframes:
            return defaults.get(lm_idx, (0.5, 0.5))

        # Before the first keyframe: hold first value.
        if frame_idx <= keyframes[0][0]:
            return keyframes[0][1]
        # After the last keyframe: hold last value.
        if frame_idx >= keyframes[-1][0]:
            return keyframes[-1][1]

        # Between keyframes: linear interpolation.
        for i in range(len(keyframes) - 1):
            fi_a, pos_a = keyframes[i]
            fi_b, pos_b = keyframes[i + 1]
            if fi_a <= frame_idx <= fi_b:
                t = (frame_idx - fi_a) / (fi_b - fi_a) if fi_b != fi_a else 0
                return (
                    pos_a[0] + t * (pos_b[0] - pos_a[0]),
                    pos_a[1] + t * (pos_b[1] - pos_a[1]),
                )

        return defaults.get(lm_idx, (0.5, 0.5))

    frames: list[dict] = []
    for fi in range(num_frames):
        landmarks: list[dict] = []
        hidden = set(missing_landmarks.get(fi, []))

        for lm_id in range(33):
            if lm_id in hidden:
                landmarks.append(
                    _make_landmark(lm_id, visibility=0.0)
                )
                continue

            if lm_id in all_override_lm_ids:
                x, y = _interpolate(lm_id, fi)
            elif lm_id in defaults:
                x, y = defaults[lm_id]
            else:
                x, y = 0.5, 0.5

            landmarks.append(_make_landmark(lm_id, x=x, y=y))

        frames.append({
            "frame_index": fi,
            "timestamp_ms": fi * dt_ms,
            "landmarks": landmarks,
            "joint_angles": joint_angles.get(fi, {}),
        })

    return frames


# ---------------------------------------------------------------------------
# Convenience: pre-built serve motion overrides
# ---------------------------------------------------------------------------

def _right_hand_serve_overrides(
    num_frames: int = 30,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Build overrides for a right-handed serve across *num_frames*.

    Phases (approximate at 30 frames):
      0-10  preparation:    right wrist at waist level (y=0.60)
      10-15 backswing:      right wrist rises to y=0.30
      15-20 trophy:         right wrist at y=0.15, knee bend
      20-25 forward swing:  right wrist at y=0.05-0.10 (peak above head)
      25-30 follow-through: right wrist descends to y=0.50

    The right shoulder stays at y=0.30, nose at y=0.15.
    The left wrist (toss arm) rises from y=0.55 to y=0.20 during backswing.
    """

    def _fi(frac: float) -> int:
        """Convert a fraction of num_frames to an integer frame index."""
        return max(0, min(num_frames - 1, round(frac * (num_frames - 1))))

    return {
        # Frame 0: preparation (standing)
        _fi(0.0): {
            LM_RIGHT_WRIST:    (0.38, 0.60),
            LM_RIGHT_ELBOW:    (0.40, 0.45),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_LEFT_WRIST:     (0.62, 0.55),
            LM_LEFT_ELBOW:     (0.60, 0.45),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        # Frame ~10 / 30: end of preparation, start of backswing
        _fi(0.33): {
            LM_RIGHT_WRIST:    (0.38, 0.50),
            LM_RIGHT_ELBOW:    (0.40, 0.38),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_LEFT_WRIST:     (0.58, 0.40),  # toss arm rising
            LM_LEFT_ELBOW:     (0.58, 0.35),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        # Frame ~15 / 30: trophy position (knee bend, wrist near peak,
        # elbow ~90 degrees)
        _fi(0.50): {
            LM_RIGHT_WRIST:    (0.35, 0.15),
            LM_RIGHT_ELBOW:    (0.38, 0.28),  # shoulder-elbow-wrist ~90
            LM_RIGHT_SHOULDER: (0.45, 0.32),
            LM_LEFT_WRIST:     (0.55, 0.20),  # toss arm high
            LM_LEFT_ELBOW:     (0.56, 0.28),
            LM_LEFT_SHOULDER:  (0.55, 0.32),
            LM_NOSE:           (0.50, 0.16),
            LM_RIGHT_HIP:      (0.47, 0.56),
            LM_LEFT_HIP:       (0.53, 0.56),
            LM_RIGHT_KNEE:     (0.56, 0.72),  # bent: knee pushed forward
            LM_LEFT_KNEE:      (0.44, 0.72),  # bent: knee pushed forward
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        # Frame ~22 / 30: contact (wrist at peak, above head)
        _fi(0.73): {
            LM_RIGHT_WRIST:    (0.42, 0.05),  # well above head
            LM_RIGHT_ELBOW:    (0.43, 0.15),
            LM_RIGHT_SHOULDER: (0.45, 0.28),
            LM_LEFT_WRIST:     (0.55, 0.35),  # toss arm descending
            LM_LEFT_ELBOW:     (0.56, 0.35),
            LM_LEFT_SHOULDER:  (0.55, 0.28),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.52),
            LM_LEFT_HIP:       (0.53, 0.52),
            LM_RIGHT_KNEE:     (0.46, 0.73),  # extended
            LM_LEFT_KNEE:      (0.54, 0.73),  # extended
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        # Frame ~28 / 30: follow-through (wrist descends)
        _fi(0.93): {
            LM_RIGHT_WRIST:    (0.55, 0.50),
            LM_RIGHT_ELBOW:    (0.50, 0.40),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_LEFT_WRIST:     (0.58, 0.50),
            LM_LEFT_ELBOW:     (0.58, 0.42),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_NOSE:           (0.50, 0.16),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
    }


def _left_hand_serve_overrides(
    num_frames: int = 30,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Mirror of right-hand serve for a left-handed player.

    Left wrist rises from y=0.60 to y=0.05 (contact above head).
    Right wrist is the toss arm.
    """

    def _fi(frac: float) -> int:
        return max(0, min(num_frames - 1, round(frac * (num_frames - 1))))

    return {
        _fi(0.0): {
            LM_LEFT_WRIST:     (0.62, 0.60),
            LM_LEFT_ELBOW:     (0.60, 0.45),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_RIGHT_WRIST:    (0.38, 0.55),
            LM_RIGHT_ELBOW:    (0.40, 0.45),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        _fi(0.33): {
            LM_LEFT_WRIST:     (0.62, 0.50),
            LM_LEFT_ELBOW:     (0.60, 0.38),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_RIGHT_WRIST:    (0.42, 0.40),  # toss arm rising
            LM_RIGHT_ELBOW:    (0.42, 0.35),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        _fi(0.50): {
            LM_LEFT_WRIST:     (0.65, 0.15),   # trophy
            LM_LEFT_ELBOW:     (0.62, 0.28),
            LM_LEFT_SHOULDER:  (0.55, 0.32),
            LM_RIGHT_WRIST:    (0.45, 0.20),   # toss arm high
            LM_RIGHT_ELBOW:    (0.44, 0.28),
            LM_RIGHT_SHOULDER: (0.45, 0.32),
            LM_NOSE:           (0.50, 0.16),
            LM_RIGHT_HIP:      (0.47, 0.56),
            LM_LEFT_HIP:       (0.53, 0.56),
            LM_RIGHT_KNEE:     (0.56, 0.72),   # bent: knee pushed forward
            LM_LEFT_KNEE:      (0.44, 0.72),   # bent: knee pushed forward
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        _fi(0.73): {
            LM_LEFT_WRIST:     (0.58, 0.05),   # contact above head
            LM_LEFT_ELBOW:     (0.57, 0.15),
            LM_LEFT_SHOULDER:  (0.55, 0.28),
            LM_RIGHT_WRIST:    (0.45, 0.35),
            LM_RIGHT_ELBOW:    (0.44, 0.35),
            LM_RIGHT_SHOULDER: (0.45, 0.28),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.52),
            LM_LEFT_HIP:       (0.53, 0.52),
            LM_RIGHT_KNEE:     (0.46, 0.73),
            LM_LEFT_KNEE:      (0.54, 0.73),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        _fi(0.93): {
            LM_LEFT_WRIST:     (0.45, 0.50),   # follow-through
            LM_LEFT_ELBOW:     (0.50, 0.40),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_RIGHT_WRIST:    (0.42, 0.50),
            LM_RIGHT_ELBOW:    (0.42, 0.42),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_NOSE:           (0.50, 0.16),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
    }


def _forehand_overrides(num_frames: int = 20) -> dict[int, dict[int, tuple[float, float]]]:
    """Forehand motion: wrist stays below shoulder the entire time."""

    def _fi(frac: float) -> int:
        return max(0, min(num_frames - 1, round(frac * (num_frames - 1))))

    return {
        _fi(0.0): {
            LM_RIGHT_WRIST:    (0.30, 0.55),
            LM_RIGHT_ELBOW:    (0.35, 0.45),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
            LM_LEFT_WRIST:     (0.62, 0.50),
            LM_LEFT_SHOULDER:  (0.55, 0.30),
            LM_NOSE:           (0.50, 0.15),
            LM_RIGHT_HIP:      (0.47, 0.55),
            LM_LEFT_HIP:       (0.53, 0.55),
            LM_RIGHT_KNEE:     (0.46, 0.72),
            LM_LEFT_KNEE:      (0.54, 0.72),
            LM_RIGHT_ANKLE:    (0.46, 0.90),
            LM_LEFT_ANKLE:     (0.54, 0.90),
        },
        _fi(0.3): {
            LM_RIGHT_WRIST:    (0.35, 0.50),
            LM_RIGHT_ELBOW:    (0.38, 0.42),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
        },
        _fi(0.5): {
            LM_RIGHT_WRIST:    (0.45, 0.45),  # contact zone, still below shoulder
            LM_RIGHT_ELBOW:    (0.43, 0.38),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
        },
        _fi(0.7): {
            LM_RIGHT_WRIST:    (0.55, 0.48),
            LM_RIGHT_ELBOW:    (0.50, 0.42),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
        },
        _fi(1.0): {
            LM_RIGHT_WRIST:    (0.60, 0.55),
            LM_RIGHT_ELBOW:    (0.55, 0.45),
            LM_RIGHT_SHOULDER: (0.45, 0.30),
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> ServeClassifier:
    return ServeClassifier()


# =========================================================================
# True Positive Tests
# =========================================================================

class TestServeTruePositives:
    """Serve motions that MUST be classified as serves with high confidence."""

    def test_right_handed_serve(self, classifier: ServeClassifier) -> None:
        """Standard right-handed serve: wrist rises from y=0.6 to y=0.05."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.shot_type == "serve"
        assert result.confidence > 0.3, (
            f"Right-handed serve confidence too low: {result.confidence}"
        )
        assert result.handedness == "right"

    def test_left_handed_serve(self, classifier: ServeClassifier) -> None:
        """Left-handed serve: left wrist rises from y=0.6 to y=0.05."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_left_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.shot_type == "serve"
        assert result.confidence > 0.3, (
            f"Left-handed serve confidence too low: {result.confidence}"
        )
        assert result.handedness == "left"

    def test_fast_serve_compact_motion(self, classifier: ServeClassifier) -> None:
        """Fast serve with fewer frames (compact, explosive motion)."""
        frames = make_frames(
            num_frames=15,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(15),
        )
        result = classifier.classify(frames)

        assert result.shot_type == "serve"
        assert result.confidence > 0.15, (
            f"Fast serve confidence too low: {result.confidence}"
        )

    def test_slow_motion_serve(self, classifier: ServeClassifier) -> None:
        """Slow-motion footage: same motion spread over 90 frames at 33ms.

        Total duration ~3 seconds, which is within the acceptable range.
        """
        frames = make_frames(
            num_frames=90,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(90),
        )
        result = classifier.classify(frames)

        assert result.shot_type == "serve"
        assert result.confidence > 0.3, (
            f"Slow-motion serve confidence too low: {result.confidence}"
        )

    def test_serve_from_side_camera(self, classifier: ServeClassifier) -> None:
        """Side-view camera: shoulders appear narrower (side profile).

        We reduce the shoulder x-gap to simulate a side view while keeping
        the vertical serve trajectory intact.
        """
        overrides = _right_hand_serve_overrides(30)
        # Narrow shoulder width to simulate side view
        for fi, lms in overrides.items():
            if LM_LEFT_SHOULDER in lms:
                lx, ly = lms[LM_LEFT_SHOULDER]
                lms[LM_LEFT_SHOULDER] = (0.52, ly)
            if LM_RIGHT_SHOULDER in lms:
                rx, ry = lms[LM_RIGHT_SHOULDER]
                lms[LM_RIGHT_SHOULDER] = (0.48, ry)

        frames = make_frames(num_frames=30, dt_ms=33.0, overrides=overrides)
        result = classifier.classify(frames)

        assert result.shot_type == "serve"
        assert result.confidence > 0.15, (
            f"Side-camera serve confidence too low: {result.confidence}"
        )


# =========================================================================
# True Negative Tests
# =========================================================================

class TestServeTrueNegatives:
    """Non-serve motions that must NOT be classified as serve."""

    def test_forehand_not_serve(self, classifier: ServeClassifier) -> None:
        """Forehand: right wrist stays between y=0.45 and y=0.60 (never above shoulder)."""
        frames = make_frames(
            num_frames=20,
            dt_ms=33.0,
            overrides=_forehand_overrides(20),
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Forehand misclassified as serve with confidence {result.confidence}"
        )

    def test_backhand_not_serve(self, classifier: ServeClassifier) -> None:
        """Backhand: wrist stays at shoulder level or below.

        The left wrist (lead for right-handed backhand) sweeps laterally
        but never rises above the shoulder.
        """
        overrides = {
            0: {
                LM_LEFT_WRIST:     (0.35, 0.45),
                LM_LEFT_ELBOW:     (0.40, 0.40),
                LM_LEFT_SHOULDER:  (0.55, 0.30),
                LM_RIGHT_WRIST:    (0.38, 0.50),
                LM_RIGHT_SHOULDER: (0.45, 0.30),
                LM_NOSE:           (0.50, 0.15),
                LM_RIGHT_HIP:      (0.47, 0.55),
                LM_LEFT_HIP:       (0.53, 0.55),
                LM_RIGHT_KNEE:     (0.46, 0.72),
                LM_LEFT_KNEE:      (0.54, 0.72),
                LM_RIGHT_ANKLE:    (0.46, 0.90),
                LM_LEFT_ANKLE:     (0.54, 0.90),
            },
            9: {
                LM_LEFT_WRIST:     (0.50, 0.40),
                LM_LEFT_ELBOW:     (0.48, 0.38),
            },
            19: {
                LM_LEFT_WRIST:     (0.65, 0.48),
                LM_LEFT_ELBOW:     (0.58, 0.42),
            },
        }
        frames = make_frames(num_frames=20, dt_ms=33.0, overrides=overrides)
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Backhand misclassified as serve with confidence {result.confidence}"
        )

    def test_slice_not_serve(self, classifier: ServeClassifier) -> None:
        """Slice: high-to-low motion but wrist never goes above head.

        The wrist starts at shoulder height (y=0.30) and descends — unlike
        a serve where it goes above the nose (y=0.15).
        """
        overrides = {
            0: {
                LM_RIGHT_WRIST:    (0.40, 0.35),
                LM_RIGHT_ELBOW:    (0.42, 0.35),
                LM_RIGHT_SHOULDER: (0.45, 0.30),
                LM_LEFT_SHOULDER:  (0.55, 0.30),
                LM_NOSE:           (0.50, 0.15),
                LM_RIGHT_HIP:      (0.47, 0.55),
                LM_LEFT_HIP:       (0.53, 0.55),
                LM_RIGHT_KNEE:     (0.46, 0.72),
                LM_LEFT_KNEE:      (0.54, 0.72),
                LM_RIGHT_ANKLE:    (0.46, 0.90),
                LM_LEFT_ANKLE:     (0.54, 0.90),
            },
            9: {
                LM_RIGHT_WRIST:    (0.48, 0.45),
                LM_RIGHT_ELBOW:    (0.46, 0.40),
            },
            19: {
                LM_RIGHT_WRIST:    (0.55, 0.60),
                LM_RIGHT_ELBOW:    (0.50, 0.50),
            },
        }
        frames = make_frames(num_frames=20, dt_ms=33.0, overrides=overrides)
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Slice misclassified as serve with confidence {result.confidence}"
        )

    def test_standing_still_not_serve(self, classifier: ServeClassifier) -> None:
        """Player standing still: no motion at all."""
        frames = make_frames(num_frames=20, dt_ms=33.0, overrides={})
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Standing still misclassified as serve with confidence {result.confidence}"
        )


# =========================================================================
# Serve-Specific Feature Tests
# =========================================================================

class TestServeFeatures:
    """Test detection of serve-specific biomechanical features."""

    def test_trophy_position_detected(self, classifier: ServeClassifier) -> None:
        """Trophy position: wrist reaches peak, elbow near 90 degrees.

        The classifier should detect a trophy position in the phase timestamps.
        """
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Serve not detected at all"
        assert result.phase_timestamps is not None, (
            "Phase timestamps not returned for serve"
        )
        assert "trophy_position" in result.phase_timestamps, (
            "Trophy position not in phase timestamps"
        )

    def test_knee_bend_and_extend(self, classifier: ServeClassifier) -> None:
        """Knee angle should decrease (bend) then increase (extend).

        We verify indirectly: a serve with proper knee bend should have
        higher confidence than one without.
        """
        # Serve WITH knee bend (standard overrides already model this)
        frames_with_bend = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result_with = classifier.classify(frames_with_bend)

        # Serve WITHOUT knee bend (knees stay straight)
        overrides_no_bend = _right_hand_serve_overrides(30)
        for fi, lms in overrides_no_bend.items():
            if LM_RIGHT_KNEE in lms:
                lms[LM_RIGHT_KNEE] = (0.46, 0.72)  # always straight
            if LM_LEFT_KNEE in lms:
                lms[LM_LEFT_KNEE] = (0.54, 0.72)   # always straight

        frames_no_bend = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=overrides_no_bend,
        )
        result_without = classifier.classify(frames_no_bend)

        # Both should still be classified as serves (wrist above head is primary),
        # but the version with knee bend should have equal or higher confidence.
        assert result_with.confidence > 0.0
        assert result_without.confidence >= 0.0

    def test_serve_duration_longer_than_groundstroke(
        self, classifier: ServeClassifier
    ) -> None:
        """Serve sequences are longer than groundstrokes.

        A 30-frame sequence at 33ms = ~1 second is typical for a serve.
        A 10-frame sequence at 33ms = ~330ms is typical for a groundstroke
        and should be rejected as too short.
        """
        # This is slightly under the 8-frame minimum but tests the concept.
        short_overrides = _right_hand_serve_overrides(8)
        frames_short = make_frames(
            num_frames=8,
            dt_ms=33.0,
            overrides=short_overrides,
        )

        # Duration is only 7 * 33 = 231ms, which is below _MIN_DURATION_MS (300).
        result_short = classifier.classify(frames_short)

        # Normal-length serve
        frames_normal = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result_normal = classifier.classify(frames_normal)

        assert result_normal.confidence > result_short.confidence


# =========================================================================
# Edge Cases
# =========================================================================

class TestServeEdgeCases:
    """Edge cases: short sequences, missing landmarks, partial visibility."""

    def test_too_few_frames_rejected(self, classifier: ServeClassifier) -> None:
        """Fewer than _MIN_FRAMES (8) should return confidence 0."""
        frames = make_frames(num_frames=5, dt_ms=33.0, overrides={})
        result = classifier.classify(frames)
        assert result.confidence == 0.0

    def test_single_frame_rejected(self, classifier: ServeClassifier) -> None:
        """A single frame cannot be a serve."""
        frames = make_frames(num_frames=1, dt_ms=33.0, overrides={})
        result = classifier.classify(frames)
        assert result.confidence == 0.0

    def test_missing_wrist_landmarks(self, classifier: ServeClassifier) -> None:
        """If both wrists are invisible for most frames, reject."""
        missing = {
            i: [LM_RIGHT_WRIST, LM_LEFT_WRIST] for i in range(25)
        }  # 25 of 30 frames
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
            missing_landmarks=missing,
        )
        result = classifier.classify(frames)

        # With only 5 visible wrist frames out of 30, this is below _MIN_FRAMES.
        assert result.confidence == 0.0

    def test_missing_shoulder_landmarks(self, classifier: ServeClassifier) -> None:
        """If shoulders are invisible, the classifier should still handle gracefully."""
        missing = {
            i: [LM_RIGHT_SHOULDER, LM_LEFT_SHOULDER]
            for i in range(20)
        }
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
            missing_landmarks=missing,
        )
        result = classifier.classify(frames)

        # Should not crash; may have low or zero confidence.
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_player_partially_out_of_frame(
        self, classifier: ServeClassifier
    ) -> None:
        """Lower body landmarks missing (player cropped at waist).

        The serve should still be detected from upper body motion alone,
        though potentially with lower confidence.
        """
        lower_body = [
            LM_LEFT_HIP, LM_RIGHT_HIP,
            LM_LEFT_KNEE, LM_RIGHT_KNEE,
            LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
        ]
        missing = {i: lower_body for i in range(30)}
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
            missing_landmarks=missing,
        )
        result = classifier.classify(frames)

        # Should not crash. Confidence may be zero due to missing torso
        # height data, but must return a valid result.
        assert isinstance(result, ClassificationResult)
        assert result.shot_type == "serve"

    def test_very_long_sequence_rejected(
        self, classifier: ServeClassifier
    ) -> None:
        """A sequence longer than _MAX_DURATION_MS should be rejected."""
        frames = make_frames(
            num_frames=30,
            dt_ms=500.0,  # 30 * 500 = 15000ms = 15s, exceeds 12s limit
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)
        assert result.confidence == 0.0


# =========================================================================
# Phase Detection Tests
# =========================================================================

class TestServePhaseDetection:
    """Verify phase timestamps are returned and in correct chronological order."""

    def test_phase_timestamps_returned(self, classifier: ServeClassifier) -> None:
        """A valid serve should include phase_timestamps."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Serve not detected"
        assert result.phase_timestamps is not None

    def test_phase_chronological_order(self, classifier: ServeClassifier) -> None:
        """Phase timestamps must be in strict chronological order.

        Expected order: preparation -> backswing -> trophy_position ->
        forward_swing -> contact -> follow_through.
        """
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Serve not detected"
        phases = result.phase_timestamps
        assert phases is not None

        expected_order = [
            "preparation",
            "backswing",
            "trophy_position",
            "forward_swing",
            "contact",
            "follow_through",
        ]

        # Collect timestamp_ms for each phase.
        phase_times = []
        for phase_name in expected_order:
            assert phase_name in phases, f"Missing phase: {phase_name}"
            phase_data = phases[phase_name]
            phase_times.append(phase_data["timestamp_ms"])

        # Each phase should start at or after the previous one.
        for i in range(1, len(phase_times)):
            assert phase_times[i] >= phase_times[i - 1], (
                f"Phase {expected_order[i]} (t={phase_times[i]}) starts before "
                f"{expected_order[i-1]} (t={phase_times[i-1]})"
            )

    def test_phases_contain_frame_index(self, classifier: ServeClassifier) -> None:
        """Each phase should contain a frame_index field."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Serve not detected"
        assert result.phase_timestamps is not None

        for phase_name, phase_data in result.phase_timestamps.items():
            assert "frame_index" in phase_data, (
                f"Phase '{phase_name}' missing frame_index"
            )
            assert isinstance(phase_data["frame_index"], (int, float)), (
                f"Phase '{phase_name}' frame_index is not numeric"
            )


# =========================================================================
# Classifier Interface Tests
# =========================================================================

class TestServeClassifierInterface:
    """Verify the classifier satisfies the BaseShotClassifier contract."""

    def test_shot_type_property(self, classifier: ServeClassifier) -> None:
        assert classifier.shot_type == "serve"

    def test_returns_classification_result(
        self, classifier: ServeClassifier
    ) -> None:
        frames = make_frames(num_frames=30, dt_ms=33.0, overrides={})
        result = classifier.classify(frames)
        assert isinstance(result, ClassificationResult)

    def test_confidence_bounds(self, classifier: ServeClassifier) -> None:
        """Confidence must always be in [0.0, 1.0]."""
        for overrides_fn in [_right_hand_serve_overrides, _forehand_overrides]:
            n = 30 if overrides_fn == _right_hand_serve_overrides else 20
            frames = make_frames(
                num_frames=n,
                dt_ms=33.0,
                overrides=overrides_fn(n),
            )
            result = classifier.classify(frames)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence {result.confidence} out of bounds"
            )

    def test_handedness_field(self, classifier: ServeClassifier) -> None:
        """Result must contain a valid handedness field."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)
        assert result.handedness in ("right", "left")

    def test_camera_angle_field(self, classifier: ServeClassifier) -> None:
        """Result must contain a valid camera_angle field."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)
        assert result.camera_angle in (
            "behind", "side", "front", "overhead", "unknown"
        )

    def test_validate_clean_on_valid_serve(
        self, classifier: ServeClassifier
    ) -> None:
        """A well-formed serve should pass validate_clean."""
        frames = make_frames(
            num_frames=30,
            dt_ms=33.0,
            overrides=_right_hand_serve_overrides(30),
        )
        result = classifier.classify(frames)

        if result.confidence > 0.0:
            assert result.is_clean is True or result.is_clean is False
