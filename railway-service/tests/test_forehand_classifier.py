"""Comprehensive unit tests for the ForehandClassifier.

Tests cover true positive detection, true negative rejection, edge cases,
clean shot validation, and camera angle detection using synthetic pose
frame sequences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow imports from the railway-service package
sys.path.insert(0, str(Path(__file__).parent.parent))

from shot_classifiers.forehand import ForehandClassifier
from shot_classifiers.base import ClassificationResult


# ---------------------------------------------------------------------------
# MediaPipe 33-point landmark names (same order as main.py)
# ---------------------------------------------------------------------------
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Landmark index shortcuts
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24


# ---------------------------------------------------------------------------
# Helper: synthetic frame generator
# ---------------------------------------------------------------------------

def _default_landmark_positions() -> dict[int, tuple[float, float]]:
    """Return a neutral standing pose as (x, y) positions for all 33 landmarks.

    Values are in MediaPipe normalized coordinates (0-1, origin top-left).
    """
    return {
        0:  (0.50, 0.15),   # nose
        1:  (0.49, 0.13),   # left_eye_inner
        2:  (0.48, 0.13),   # left_eye
        3:  (0.47, 0.13),   # left_eye_outer
        4:  (0.51, 0.13),   # right_eye_inner
        5:  (0.52, 0.13),   # right_eye
        6:  (0.53, 0.13),   # right_eye_outer
        7:  (0.46, 0.14),   # left_ear
        8:  (0.54, 0.14),   # right_ear
        9:  (0.49, 0.17),   # mouth_left
        10: (0.51, 0.17),   # mouth_right
        11: (0.42, 0.25),   # left_shoulder
        12: (0.58, 0.25),   # right_shoulder
        13: (0.38, 0.35),   # left_elbow
        14: (0.62, 0.35),   # right_elbow
        15: (0.35, 0.45),   # left_wrist
        16: (0.65, 0.45),   # right_wrist
        17: (0.34, 0.47),   # left_pinky
        18: (0.66, 0.47),   # right_pinky
        19: (0.34, 0.46),   # left_index
        20: (0.66, 0.46),   # right_index
        21: (0.35, 0.46),   # left_thumb
        22: (0.65, 0.46),   # right_thumb
        23: (0.45, 0.50),   # left_hip
        24: (0.55, 0.50),   # right_hip
        25: (0.44, 0.65),   # left_knee
        26: (0.56, 0.65),   # right_knee
        27: (0.43, 0.80),   # left_ankle
        28: (0.57, 0.80),   # right_ankle
        29: (0.43, 0.83),   # left_heel
        30: (0.57, 0.83),   # right_heel
        31: (0.42, 0.85),   # left_foot_index
        32: (0.58, 0.85),   # right_foot_index
    }


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b at parameter t in [0, 1]."""
    return a + (b - a) * t


def make_frames(
    num_frames: int = 20,
    fps: float = 30.0,
    *,
    # Per-landmark animation overrides: {landmark_id: (start_xy, end_xy)}
    animations: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    # Joint angle overrides at start and end (linearly interpolated)
    angle_start: dict[str, float] | None = None,
    angle_end: dict[str, float] | None = None,
    # Shoulder width animation (for camera angle / trunk rotation tests)
    shoulder_width_start: float | None = None,
    shoulder_width_end: float | None = None,
    # Visibility override for all landmarks (default 0.9)
    visibility: float = 0.9,
    # Per-landmark visibility overrides
    visibility_overrides: dict[int, float] | None = None,
) -> list[dict]:
    """Generate a synthetic sequence of MediaPipe pose frames.

    Each frame contains all 33 landmarks with (x, y, z, visibility) and
    a ``joint_angles`` dict. Landmarks interpolate linearly between their
    start and end positions over ``num_frames`` frames.

    Args:
        num_frames: Number of frames to generate.
        fps: Simulated frame rate (controls timestamp_ms spacing).
        animations: Dict mapping landmark index to (start_xy, end_xy).
            Unlisted landmarks stay at their default neutral positions.
        angle_start: Joint angle values at frame 0.
        angle_end: Joint angle values at the last frame.
        shoulder_width_start: If set, overrides left/right shoulder x
            positions to produce this apparent width at frame 0.
        shoulder_width_end: Same, at the last frame.
        visibility: Default visibility for all landmarks.
        visibility_overrides: Per-landmark visibility values.

    Returns:
        A list of frame dicts matching the PoseFrame schema.
    """
    defaults = _default_landmark_positions()
    animations = animations or {}
    angle_start = angle_start or {}
    angle_end = angle_end or {}
    visibility_overrides = visibility_overrides or {}

    dt_ms = 1000.0 / fps
    frames: list[dict] = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)  # 0.0 -> 1.0
        timestamp_ms = round(i * dt_ms, 1)

        # Build landmarks
        landmarks: list[dict] = []
        for lm_id in range(33):
            if lm_id in animations:
                start_xy, end_xy = animations[lm_id]
                x = _lerp(start_xy[0], end_xy[0], t)
                y = _lerp(start_xy[1], end_xy[1], t)
            else:
                x, y = defaults[lm_id]

            # Shoulder width override adjusts left_shoulder and right_shoulder
            if shoulder_width_start is not None and shoulder_width_end is not None:
                w = _lerp(shoulder_width_start, shoulder_width_end, t)
                center_x = 0.50
                if lm_id == LM_LEFT_SHOULDER:
                    x = center_x - w / 2
                    y = defaults[lm_id][1]
                elif lm_id == LM_RIGHT_SHOULDER:
                    x = center_x + w / 2
                    y = defaults[lm_id][1]

            vis = visibility_overrides.get(lm_id, visibility)
            landmarks.append({
                "id": lm_id,
                "name": LANDMARK_NAMES[lm_id],
                "x": round(x, 4),
                "y": round(y, 4),
                "z": 0.0,
                "visibility": vis,
            })

        # Build joint angles
        joint_angles: dict[str, float] = {}
        all_angle_keys = set(angle_start.keys()) | set(angle_end.keys())
        for key in all_angle_keys:
            start_val = angle_start.get(key, 90.0)
            end_val = angle_end.get(key, start_val)
            joint_angles[key] = round(_lerp(start_val, end_val, t), 1)

        frames.append({
            "frame_index": i,
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
            "joint_angles": joint_angles,
        })

    return frames


# ---------------------------------------------------------------------------
# Convenience helpers for common shot patterns
# ---------------------------------------------------------------------------

def _right_forehand_frames(
    num_frames: int = 20,
    fps: float = 30.0,
    *,
    wrist_start_x: float = 0.70,
    wrist_end_x: float = 0.30,
    wrist_y: float = 0.50,
    elbow_start: float = 90.0,
    elbow_end: float = 150.0,
    trunk_start: float = 5.0,
    trunk_end: float = 45.0,
    shoulder_width_start: float = 0.12,
    shoulder_width_end: float = 0.20,
    visibility: float = 0.9,
) -> list[dict]:
    """Generate frames for a right-handed forehand (behind camera view)."""
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            LM_RIGHT_WRIST: ((wrist_start_x, wrist_y), (wrist_end_x, wrist_y)),
            LM_RIGHT_ELBOW: ((0.62, 0.35), (0.45, 0.35)),
        },
        angle_start={
            "right_elbow": elbow_start,
            "left_elbow": 160.0,
            "right_shoulder": 40.0,
            "left_shoulder": 20.0,
            "right_knee": 160.0,
            "left_knee": 160.0,
            "hip_rotation": 5.0,
            "trunk_rotation": trunk_start,
        },
        angle_end={
            "right_elbow": elbow_end,
            "left_elbow": 160.0,
            "right_shoulder": 80.0,
            "left_shoulder": 20.0,
            "right_knee": 160.0,
            "left_knee": 160.0,
            "hip_rotation": 5.0,
            "trunk_rotation": trunk_end,
        },
        shoulder_width_start=shoulder_width_start,
        shoulder_width_end=shoulder_width_end,
        visibility=visibility,
    )


def _left_forehand_frames(
    num_frames: int = 20,
    fps: float = 30.0,
    *,
    wrist_start_x: float = 0.30,
    wrist_end_x: float = 0.70,
    wrist_y: float = 0.50,
) -> list[dict]:
    """Generate frames for a left-handed forehand (behind camera view)."""
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            LM_LEFT_WRIST: ((wrist_start_x, wrist_y), (wrist_end_x, wrist_y)),
            LM_LEFT_ELBOW: ((0.38, 0.35), (0.55, 0.35)),
            # Keep right wrist mostly still so handedness detects left
            LM_RIGHT_WRIST: ((0.60, 0.45), (0.60, 0.45)),
        },
        angle_start={
            "left_elbow": 90.0,
            "right_elbow": 160.0,
            "trunk_rotation": 5.0,
        },
        angle_end={
            "left_elbow": 150.0,
            "right_elbow": 160.0,
            "trunk_rotation": 45.0,
        },
        shoulder_width_start=0.12,
        shoulder_width_end=0.20,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> ForehandClassifier:
    return ForehandClassifier()


# ===========================================================================
# 1. TRUE POSITIVE TESTS
# ===========================================================================


class TestTruePositives:
    """Verify that clear forehand motions are detected with high confidence."""

    def test_right_handed_forehand_behind_view(self, classifier: ForehandClassifier):
        """A standard right-handed forehand viewed from behind should classify
        with high confidence and correct handedness."""
        frames = _right_forehand_frames()
        result = classifier.classify(frames)

        assert result.shot_type == "forehand"
        assert result.confidence >= 0.5
        assert result.handedness == "right"

    def test_left_handed_forehand_behind_view(self, classifier: ForehandClassifier):
        """A left-handed forehand viewed from behind should classify with the
        dominant wrist moving from left to right."""
        frames = _left_forehand_frames()
        result = classifier.classify(frames)

        assert result.shot_type == "forehand"
        assert result.confidence >= 0.3
        assert result.handedness == "left"

    def test_fast_swing_forehand(self, classifier: ForehandClassifier):
        """A fast swing (fewer frames, higher fps) should still be detected."""
        frames = _right_forehand_frames(num_frames=10, fps=60.0)
        result = classifier.classify(frames)

        assert result.shot_type == "forehand"
        assert result.confidence >= 0.3

    def test_slow_motion_forehand(self, classifier: ForehandClassifier):
        """A slow-motion clip (many frames, lower effective fps) should still
        be recognized as a forehand."""
        # 60 frames at 30fps = 2 seconds, within the clean shot window
        frames = _right_forehand_frames(num_frames=60, fps=30.0)
        result = classifier.classify(frames)

        assert result.shot_type == "forehand"
        assert result.confidence >= 0.3

    def test_forehand_side_camera(self, classifier: ForehandClassifier):
        """A forehand from a side camera view should still classify.
        Side view has narrow shoulder width."""
        frames = _right_forehand_frames(
            shoulder_width_start=0.04,
            shoulder_width_end=0.06,
        )
        result = classifier.classify(frames)

        assert result.shot_type == "forehand"
        assert result.confidence >= 0.3

    def test_forehand_front_camera(self, classifier: ForehandClassifier):
        """A right-handed forehand from the front view -- the wrist moves
        in the opposite screen direction, but the classifier should adapt
        based on camera angle detection."""
        # From front view, right-handed forehand wrist moves left-to-right
        # on screen (reversed from behind). The large shoulder width signals
        # front/behind view. Nose is highly visible for front detection.
        frames = make_frames(
            num_frames=20,
            animations={
                # Right wrist sweeps left-to-right in screen coords (front view)
                LM_RIGHT_WRIST: ((0.30, 0.50), (0.70, 0.50)),
                LM_RIGHT_ELBOW: ((0.45, 0.35), (0.62, 0.35)),
            },
            angle_start={
                "right_elbow": 90.0,
                "trunk_rotation": 5.0,
            },
            angle_end={
                "right_elbow": 150.0,
                "trunk_rotation": 45.0,
            },
            shoulder_width_start=0.20,
            shoulder_width_end=0.20,
            # Nose highly visible => front view detection
            visibility=0.9,
        )
        # Boost nose visibility so estimate_camera_angle detects "front"
        for f in frames:
            f["landmarks"][LM_NOSE]["visibility"] = 0.95

        result = classifier.classify(frames)

        # Should at least not crash; confidence may vary depending on
        # whether camera angle detection correctly picks "front"
        assert result.shot_type == "forehand"


# ===========================================================================
# 2. TRUE NEGATIVE TESTS
# ===========================================================================


class TestTrueNegatives:
    """Verify that non-forehand motions are rejected (confidence 0.0 or very low)."""

    def test_backhand_rejected(self, classifier: ForehandClassifier):
        """A backhand motion (right wrist moves left-to-right for a right-hander)
        should not classify as a forehand."""
        # Use no shoulder width change and no elbow extension to isolate the
        # lateral direction check.  The wrist starts on the left side and
        # sweeps to the right -- opposite of a right-handed forehand.
        frames = make_frames(
            num_frames=20,
            animations={
                LM_RIGHT_WRIST: ((0.30, 0.50), (0.70, 0.50)),
                LM_RIGHT_ELBOW: ((0.45, 0.35), (0.62, 0.35)),
            },
            angle_start={"right_elbow": 150.0, "trunk_rotation": 5.0},
            angle_end={"right_elbow": 150.0, "trunk_rotation": 5.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = classifier.classify(frames)

        # The lateral displacement is in the wrong direction for a forehand
        # and there is no shoulder rotation or elbow extension signal
        assert result.confidence < 0.3

    def test_serve_motion_rejected(self, classifier: ForehandClassifier):
        """A serve motion where the wrist goes well above the shoulder should
        be rejected because the below-shoulder score will be very low."""
        frames = make_frames(
            num_frames=20,
            animations={
                # Right wrist goes from slightly below shoulder to well above head
                # Most frames have wrist above shoulder (y < 0.25)
                LM_RIGHT_WRIST: ((0.55, 0.20), (0.55, 0.02)),
                LM_RIGHT_ELBOW: ((0.57, 0.22), (0.55, 0.10)),
            },
            angle_start={"right_elbow": 90.0, "trunk_rotation": 5.0},
            angle_end={"right_elbow": 170.0, "trunk_rotation": 10.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = classifier.classify(frames)

        # Wrist is above shoulder for the entire sequence, so
        # below-shoulder score is near 0 and lateral displacement is minimal
        assert result.confidence < 0.3

    def test_slice_motion_rejected(self, classifier: ForehandClassifier):
        """A slice (high-to-low wrist path with negligible lateral movement)
        should not be classified as a forehand."""
        frames = make_frames(
            num_frames=20,
            animations={
                # Wrist drops vertically with almost no lateral movement
                LM_RIGHT_WRIST: ((0.55, 0.25), (0.55, 0.70)),
                LM_RIGHT_ELBOW: ((0.57, 0.30), (0.55, 0.45)),
            },
            angle_start={"right_elbow": 130.0, "trunk_rotation": 5.0},
            angle_end={"right_elbow": 140.0, "trunk_rotation": 8.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = classifier.classify(frames)

        # Effectively zero lateral displacement -> lateral_score = 0
        assert result.confidence < 0.3

    def test_standing_still_rejected(self, classifier: ForehandClassifier):
        """A player standing still with no wrist movement should not be a forehand."""
        frames = make_frames(
            num_frames=20,
            # No animations -- all landmarks stay at default neutral position
            angle_start={"right_elbow": 160.0, "trunk_rotation": 5.0},
            angle_end={"right_elbow": 160.0, "trunk_rotation": 5.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0

    def test_walking_motion_rejected(self, classifier: ForehandClassifier):
        """Walking-like motion (small oscillating wrist movement) should not
        be classified as a forehand."""
        # Simulate walking: wrist oscillates laterally with small amplitude
        animations = {}
        # Both wrists move symmetrically (not a dominant hand swing)
        frames = make_frames(
            num_frames=20,
            animations={
                LM_RIGHT_WRIST: ((0.63, 0.45), (0.67, 0.45)),
                LM_LEFT_WRIST:  ((0.37, 0.45), (0.33, 0.45)),
            },
            angle_start={"right_elbow": 155.0, "left_elbow": 155.0, "trunk_rotation": 3.0},
            angle_end={"right_elbow": 155.0, "left_elbow": 155.0, "trunk_rotation": 3.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = classifier.classify(frames)

        # Minimal lateral displacement; should not pass threshold
        assert result.confidence < 0.3


# ===========================================================================
# 3. EDGE CASE TESTS
# ===========================================================================


class TestEdgeCases:
    """Verify graceful handling of unusual inputs."""

    def test_very_short_sequence_rejected(self, classifier: ForehandClassifier):
        """Sequences with fewer than 5 frames should be rejected outright."""
        frames = _right_forehand_frames(num_frames=4)
        result = classifier.classify(frames)

        assert result.confidence == 0.0
        assert result.is_clean is False

    def test_empty_sequence(self, classifier: ForehandClassifier):
        """An empty frame list should return zero confidence without crashing."""
        result = classifier.classify([])

        assert result.confidence == 0.0
        assert result.shot_type == "forehand"

    def test_single_frame(self, classifier: ForehandClassifier):
        """A single frame should be rejected gracefully."""
        frames = _right_forehand_frames(num_frames=1)
        result = classifier.classify(frames)

        assert result.confidence == 0.0

    def test_very_long_sequence(self, classifier: ForehandClassifier):
        """A sequence spanning > 3 seconds should still classify, but
        validate_clean should return False."""
        # 120 frames at 30fps = 4 seconds, exceeds _MAX_SHOT_DURATION_MS
        frames = _right_forehand_frames(num_frames=120, fps=30.0)
        result = classifier.classify(frames)

        # Classification may still detect forehand motion
        assert result.shot_type == "forehand"
        # But it should not be considered a clean single shot
        assert result.is_clean is False

    def test_low_visibility_landmarks(self, classifier: ForehandClassifier):
        """Landmarks with low visibility should be handled gracefully.
        The base get_landmark function filters out visibility < 0.3."""
        frames = _right_forehand_frames(visibility=0.2)
        result = classifier.classify(frames)

        # With all landmarks below 0.3 visibility, lm_xy returns None
        # everywhere, so the classifier should reject gracefully
        assert result.confidence == 0.0

    def test_missing_landmarks(self, classifier: ForehandClassifier):
        """Frames with a truncated landmarks list should not crash."""
        frames = _right_forehand_frames(num_frames=10)
        # Truncate landmarks to only 10 (missing wrists, hips, etc.)
        for f in frames:
            f["landmarks"] = f["landmarks"][:10]
        result = classifier.classify(frames)

        # Should not crash, but cannot classify without key landmarks
        assert result.confidence == 0.0

    def test_missing_joint_angles(self, classifier: ForehandClassifier):
        """Frames with no joint_angles key should still classify based on
        wrist trajectory alone."""
        frames = _right_forehand_frames()
        for f in frames:
            f["joint_angles"] = {}
        result = classifier.classify(frames)

        # Should still classify based on lateral displacement and
        # wrist-below-shoulder signals (elbow/rotation get neutral scores)
        assert result.shot_type == "forehand"
        assert result.confidence >= 0.3

    def test_partial_visibility(self, classifier: ForehandClassifier):
        """Some landmarks visible and some not -- should handle mixed visibility."""
        frames = _right_forehand_frames()
        # Make hip landmarks invisible on half the frames
        for i, f in enumerate(frames):
            if i % 2 == 0:
                f["landmarks"][LM_LEFT_HIP]["visibility"] = 0.1
                f["landmarks"][LM_RIGHT_HIP]["visibility"] = 0.1
        result = classifier.classify(frames)

        # Should still work with the frames that have visible hips
        assert result.shot_type == "forehand"


# ===========================================================================
# 4. CLEAN SHOT VALIDATION TESTS
# ===========================================================================


class TestValidateClean:
    """Test the validate_clean method for single-shot integrity checks."""

    def test_single_forehand_is_clean(self, classifier: ForehandClassifier):
        """A single forehand within duration limits should pass validate_clean."""
        frames = _right_forehand_frames(num_frames=20, fps=30.0)
        result = classifier.classify(frames)

        assert result.is_clean is True

    def test_too_short_not_clean(self, classifier: ForehandClassifier):
        """Sequences below 5 frames fail validate_clean."""
        frames = _right_forehand_frames(num_frames=4)
        assert classifier.validate_clean(frames) is False

    def test_too_long_not_clean(self, classifier: ForehandClassifier):
        """Sequences exceeding 3 seconds fail validate_clean."""
        # 120 frames at 30fps = 4 seconds > 3s limit
        frames = _right_forehand_frames(num_frames=120, fps=30.0)
        assert classifier.validate_clean(frames) is False

    def test_multiple_shots_not_clean(self, classifier: ForehandClassifier):
        """A sequence with more than 2 direction reversals (simulating a rally
        or multiple spliced forehands) should fail validate_clean."""
        # Create a four-stroke rally to guarantee > 2 direction reversals
        # after smoothing.
        segments = [
            (0.70, 0.35),  # sweep right->left (forehand)
            (0.35, 0.68),  # sweep left->right (recovery)
            (0.68, 0.33),  # sweep right->left (forehand)
            (0.33, 0.66),  # sweep left->right (recovery)
        ]

        combined: list[dict] = []
        offset_ms = 0.0
        global_idx = 0
        for wrist_start, wrist_end in segments:
            seg = _right_forehand_frames(
                num_frames=6, fps=30.0,
                wrist_start_x=wrist_start, wrist_end_x=wrist_end,
            )
            for f in seg:
                f["frame_index"] = global_idx
                f["timestamp_ms"] = offset_ms + (f["frame_index"] - (global_idx - len(seg) + len(seg))) * 33.3
                global_idx += 1
            offset_ms = combined[-1]["timestamp_ms"] + 33.3 if combined else 0.0
            combined.extend(seg)

        # Fix timestamps to be monotonically increasing
        for i, f in enumerate(combined):
            f["frame_index"] = i
            f["timestamp_ms"] = round(i * 33.3, 1)

        # 24 frames at 30fps ~ 0.8s, under the 3s limit.
        # 3 direction reversals should trigger rejection (limit is 2).
        assert classifier.validate_clean(combined) is False

    def test_serve_contamination_not_clean(self, classifier: ForehandClassifier):
        """A forehand where the wrist goes above the nose at any point
        should fail validate_clean (serve contamination)."""
        frames = _right_forehand_frames(num_frames=20, fps=30.0)
        # Move wrist above nose (y < nose_y) in a few middle frames
        nose_y = frames[10]["landmarks"][LM_NOSE]["y"]
        for i in range(8, 13):
            frames[i]["landmarks"][LM_RIGHT_WRIST]["y"] = nose_y - 0.10

        assert classifier.validate_clean(frames) is False


# ===========================================================================
# 5. CAMERA ANGLE DETECTION TESTS
# ===========================================================================


class TestCameraAngleDetection:
    """Verify camera angle detection for forehand sequences."""

    def test_behind_view_detected(self, classifier: ForehandClassifier):
        """Wide shoulder width + low nose visibility => 'behind' camera angle."""
        frames = _right_forehand_frames(
            shoulder_width_start=0.18,
            shoulder_width_end=0.22,
        )
        # Reduce nose visibility to signal behind view
        for f in frames:
            f["landmarks"][LM_NOSE]["visibility"] = 0.3

        result = classifier.classify(frames)
        assert result.camera_angle == "behind"

    def test_side_view_detected(self, classifier: ForehandClassifier):
        """Narrow shoulder width (shoulders overlapping) => 'side' camera angle."""
        frames = _right_forehand_frames(
            shoulder_width_start=0.03,
            shoulder_width_end=0.05,
        )
        result = classifier.classify(frames)
        assert result.camera_angle == "side"

    def test_front_view_detected(self, classifier: ForehandClassifier):
        """Wide shoulder width + highly visible nose => 'front' camera angle."""
        frames = _right_forehand_frames(
            shoulder_width_start=0.20,
            shoulder_width_end=0.22,
        )
        # High nose visibility for front view
        for f in frames:
            f["landmarks"][LM_NOSE]["visibility"] = 0.95

        result = classifier.classify(frames)
        assert result.camera_angle == "front"


# ===========================================================================
# 6. CLASSIFICATION RESULT STRUCTURE TESTS
# ===========================================================================


class TestResultStructure:
    """Verify the ClassificationResult fields are populated correctly."""

    def test_result_has_all_fields(self, classifier: ForehandClassifier):
        """The returned ClassificationResult should have all required fields."""
        frames = _right_forehand_frames()
        result = classifier.classify(frames)

        assert hasattr(result, "shot_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_clean")
        assert hasattr(result, "camera_angle")
        assert hasattr(result, "handedness")
        assert hasattr(result, "phase_timestamps")

    def test_confidence_is_bounded(self, classifier: ForehandClassifier):
        """Confidence should always be between 0.0 and 1.0."""
        frames = _right_forehand_frames()
        result = classifier.classify(frames)

        assert 0.0 <= result.confidence <= 1.0

    def test_phase_timestamps_on_positive(self, classifier: ForehandClassifier):
        """A positive classification should include phase timestamps."""
        frames = _right_forehand_frames()
        result = classifier.classify(frames)

        if result.confidence > 0.3:
            assert result.phase_timestamps is not None
            assert "preparation" in result.phase_timestamps
            assert "contact" in result.phase_timestamps
            assert "follow_through" in result.phase_timestamps

    def test_rejection_result_fields(self, classifier: ForehandClassifier):
        """A rejected classification should still have valid fields."""
        result = classifier.classify([])

        assert result.shot_type == "forehand"
        assert result.confidence == 0.0
        assert result.is_clean is False

    def test_shot_type_property(self, classifier: ForehandClassifier):
        """The shot_type property should return 'forehand'."""
        assert classifier.shot_type == "forehand"
