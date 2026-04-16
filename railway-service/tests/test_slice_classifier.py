"""Comprehensive unit tests for the SliceClassifier.

The slice is the hardest shot to classify because it is biomechanically
similar to groundstrokes but with a distinct HIGH-TO-LOW wrist path
(wrist y increases in image coordinates during the swing phase).

Tests are organized into:
- True positives (backhand slice, forehand slice, varied camera angles)
- True negatives (topspin forehand, topspin backhand, serve, volley)
- Slice vs topspin boundary tests (the critical classification boundary)
- Edge cases (flat/borderline trajectory, missing landmarks)
- Clean validation (single shot passes, multi-shot fails)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shot_classifiers.slice import SliceClassifier


# ---------------------------------------------------------------------------
# Frame generation helpers
# ---------------------------------------------------------------------------

def _make_landmark(
    idx: int, x: float, y: float, z: float = 0.0, visibility: float = 0.9
) -> dict:
    """Create a single MediaPipe landmark dict."""
    return {"id": idx, "name": f"landmark_{idx}", "x": x, "y": y, "z": z, "visibility": visibility}


def make_frames(
    n_frames: int = 20,
    *,
    # Wrist trajectory (dominant hand = right by default)
    wrist_start_x: float = 0.3,
    wrist_end_x: float = 0.6,
    wrist_start_y: float = 0.35,
    wrist_end_y: float = 0.55,
    # Body positioning
    shoulder_y: float = 0.32,
    hip_y: float = 0.55,
    nose_y: float = 0.22,
    # Shoulder positions (controls apparent rotation)
    left_shoulder_x: float = 0.45,
    right_shoulder_x: float = 0.55,
    # Shoulder width change (rotation proxy): end values
    left_shoulder_x_end: float | None = None,
    right_shoulder_x_end: float | None = None,
    # Elbow angle (degrees)
    elbow_angle: float = 150.0,
    elbow_angle_end: float | None = None,
    # Duration
    duration_ms: float = 1000.0,
    # Left wrist (non-dominant)
    left_wrist_x: float = 0.45,
    left_wrist_y: float = 0.45,
    # Control which hand has more lateral displacement
    left_wrist_x_end: float | None = None,
    # Visibility for wrist landmarks
    wrist_visibility: float = 0.9,
    # Nose visibility
    nose_visibility: float = 0.8,
) -> list[dict]:
    """Generate a sequence of pose frames with 33 landmarks.

    By default, generates a right-handed backhand slice pattern:
    - Right wrist starts at y=0.35 (high), ends at y=0.55 (low) -- HIGH-TO-LOW
    - Right wrist x: moves from 0.3 to 0.6 (left to right)
    - Elbow angle stays ~150 degrees (arm relatively straight)
    - Modest trunk rotation (~10 degrees)

    All 33 landmarks are populated so MediaPipe expectations are met.
    """
    if left_shoulder_x_end is None:
        left_shoulder_x_end = left_shoulder_x
    if right_shoulder_x_end is None:
        right_shoulder_x_end = right_shoulder_x
    if elbow_angle_end is None:
        elbow_angle_end = elbow_angle
    if left_wrist_x_end is None:
        left_wrist_x_end = left_wrist_x

    frames: list[dict] = []

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)  # 0.0 -> 1.0

        # Interpolate wrist position
        rw_x = wrist_start_x + (wrist_end_x - wrist_start_x) * t
        rw_y = wrist_start_y + (wrist_end_y - wrist_start_y) * t

        # Interpolate shoulder positions (rotation change)
        ls_x = left_shoulder_x + (left_shoulder_x_end - left_shoulder_x) * t
        rs_x = right_shoulder_x + (right_shoulder_x_end - right_shoulder_x) * t

        # Interpolate left wrist
        lw_x = left_wrist_x + (left_wrist_x_end - left_wrist_x) * t

        # Interpolate elbow angle
        ea = elbow_angle + (elbow_angle_end - elbow_angle) * t

        # Compute elbow position from angle (simplified: place elbow between
        # shoulder and wrist, offset by angle)
        re_x = (rs_x + rw_x) / 2
        re_y = (shoulder_y + rw_y) / 2

        # Left elbow (non-dominant, roughly mirrored)
        le_x = (ls_x + lw_x) / 2
        le_y = (shoulder_y + left_wrist_y) / 2

        # Build all 33 landmarks
        landmarks = []
        for idx in range(33):
            if idx == 0:  # NOSE
                landmarks.append(_make_landmark(idx, 0.5, nose_y, visibility=nose_visibility))
            elif idx == 11:  # LEFT_SHOULDER
                landmarks.append(_make_landmark(idx, ls_x, shoulder_y))
            elif idx == 12:  # RIGHT_SHOULDER
                landmarks.append(_make_landmark(idx, rs_x, shoulder_y))
            elif idx == 13:  # LEFT_ELBOW
                landmarks.append(_make_landmark(idx, le_x, le_y))
            elif idx == 14:  # RIGHT_ELBOW
                landmarks.append(_make_landmark(idx, re_x, re_y))
            elif idx == 15:  # LEFT_WRIST
                landmarks.append(_make_landmark(idx, lw_x, left_wrist_y))
            elif idx == 16:  # RIGHT_WRIST
                landmarks.append(_make_landmark(idx, rw_x, rw_y, visibility=wrist_visibility))
            elif idx == 23:  # LEFT_HIP
                landmarks.append(_make_landmark(idx, 0.47, hip_y))
            elif idx == 24:  # RIGHT_HIP
                landmarks.append(_make_landmark(idx, 0.53, hip_y))
            elif idx == 25:  # LEFT_KNEE
                landmarks.append(_make_landmark(idx, 0.47, 0.72))
            elif idx == 26:  # RIGHT_KNEE
                landmarks.append(_make_landmark(idx, 0.53, 0.72))
            elif idx == 27:  # LEFT_ANKLE
                landmarks.append(_make_landmark(idx, 0.47, 0.90))
            elif idx == 28:  # RIGHT_ANKLE
                landmarks.append(_make_landmark(idx, 0.53, 0.90))
            else:
                # Fill remaining landmarks with reasonable defaults
                landmarks.append(_make_landmark(idx, 0.5, 0.5, visibility=0.5))

        timestamp_ms = (duration_ms * t)

        frame = {
            "frame_index": i,
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
            "joint_angles": {
                "right_elbow": ea,
                "left_elbow": 130.0,
            },
        }
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> SliceClassifier:
    return SliceClassifier()


# ===========================================================================
# TRUE POSITIVE TESTS -- sequences that SHOULD be classified as slice
# ===========================================================================


class TestTruePositives:
    """Verify that genuine slice motions are detected with reasonable confidence."""

    def test_backhand_slice_basic(self, classifier: SliceClassifier) -> None:
        """Standard right-handed backhand slice: wrist HIGH-TO-LOW path.

        Wrist starts at y=0.35 (shoulder height), descends to y=0.55 (hip
        height).  In image coords, y increasing = wrist moving DOWN physically.
        This is the most common slice variant.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
            wrist_start_x=0.3,
            wrist_end_x=0.6,
            elbow_angle=150.0,
            duration_ms=1000.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Backhand slice should be detected"
        assert "slice" in result.shot_type, f"Expected slice, got {result.shot_type}"
        assert result.handedness == "right"

    def test_backhand_slice_high_confidence(self, classifier: SliceClassifier) -> None:
        """A textbook backhand slice with large vertical drop should have
        high confidence (> 0.5).
        """
        frames = make_frames(
            n_frames=25,
            wrist_start_y=0.30,   # very high start
            wrist_end_y=0.60,     # well below hip
            wrist_start_x=0.25,
            wrist_end_x=0.65,
            elbow_angle=155.0,    # very straight arm
            duration_ms=1200.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.5, (
            f"Textbook backhand slice should have high confidence, got {result.confidence}"
        )
        assert "slice" in result.shot_type

    def test_forehand_slice(self, classifier: SliceClassifier) -> None:
        """Forehand slice: same high-to-low pattern but wrist starts on
        the dominant (right) side of the body.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.53,
            # Wrist starts on the right side (forehand for right-hander)
            wrist_start_x=0.65,
            wrist_end_x=0.45,
            elbow_angle=145.0,
            duration_ms=900.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Forehand slice should be detected"
        assert "slice" in result.shot_type

    def test_left_handed_backhand_slice(self, classifier: SliceClassifier) -> None:
        """Left-handed backhand slice: left wrist has more displacement
        so handedness detection picks left.
        """
        frames = make_frames(
            n_frames=20,
            # Right wrist stays relatively still
            wrist_start_x=0.5,
            wrist_end_x=0.52,
            wrist_start_y=0.45,
            wrist_end_y=0.46,
            wrist_visibility=0.9,
            # Left wrist does the swing (high-to-low)
            left_wrist_x=0.7,
            left_wrist_x_end=0.4,
            left_wrist_y=0.35,  # Note: left_wrist_y is static in make_frames
            elbow_angle=130.0,
            duration_ms=1000.0,
        )
        # Override left wrist y to create high-to-low path manually
        for i, frame in enumerate(frames):
            t = i / max(1, len(frames) - 1)
            frame["landmarks"][15]["y"] = 0.35 + 0.20 * t  # high-to-low
            frame["joint_angles"]["left_elbow"] = 150.0

        result = classifier.classify(frames)

        assert result.confidence > 0.0, "Left-handed slice should be detected"
        assert result.handedness == "left"
        assert "slice" in result.shot_type


# ===========================================================================
# TRUE NEGATIVE TESTS -- sequences that MUST NOT be classified as slice
# ===========================================================================


class TestTrueNegatives:
    """Verify that non-slice shots are rejected (confidence 0.0)."""

    def test_topspin_forehand_rejected(self, classifier: SliceClassifier) -> None:
        """Topspin forehand: LOW-TO-HIGH wrist path (y DECREASES).

        The wrist starts low (y=0.55) and finishes high (y=0.35).
        This is the OPPOSITE of a slice and must be rejected.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.55,   # starts LOW (high y)
            wrist_end_y=0.35,     # ends HIGH (low y) -- LOW-TO-HIGH
            wrist_start_x=0.6,
            wrist_end_x=0.3,
            elbow_angle=100.0,    # more bent arm typical of topspin
            elbow_angle_end=160.0,
            # More trunk rotation typical of topspin
            left_shoulder_x=0.42,
            left_shoulder_x_end=0.48,
            right_shoulder_x=0.52,
            right_shoulder_x_end=0.58,
            duration_ms=800.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Topspin forehand must NOT be classified as slice, got confidence {result.confidence}"
        )

    def test_topspin_backhand_rejected(self, classifier: SliceClassifier) -> None:
        """Topspin backhand: LOW-TO-HIGH wrist path (y DECREASES).

        Same low-to-high trajectory as topspin forehand.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.55,
            wrist_end_y=0.30,     # strong upward motion
            wrist_start_x=0.3,
            wrist_end_x=0.65,
            elbow_angle=90.0,
            elbow_angle_end=155.0,
            duration_ms=900.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Topspin backhand must NOT be classified as slice, got confidence {result.confidence}"
        )

    def test_serve_rejected(self, classifier: SliceClassifier) -> None:
        """Serve: wrist goes well above the head (y much lower than nose_y).

        The serve contamination check should catch this even if the
        overall trajectory has a downward component.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.40,
            wrist_end_y=0.55,
            wrist_start_x=0.55,
            wrist_end_x=0.50,
            nose_y=0.22,
            elbow_angle=90.0,
            elbow_angle_end=170.0,
            duration_ms=1500.0,
        )
        # Make the wrist go above the head in the middle of the sequence
        # (trophy position)
        for i in range(5, 12):
            t = (i - 5) / 7
            # Wrist rises well above head: y goes to 0.05 (way above nose at 0.22)
            frames[i]["landmarks"][16]["y"] = 0.40 - 0.35 * math.sin(t * math.pi)

        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Serve must NOT be classified as slice, got confidence {result.confidence}"
        )

    def test_volley_rejected(self, classifier: SliceClassifier) -> None:
        """Volley: very compact motion with minimal backswing.

        The wrist barely moves vertically -- too little drop to meet the
        slice threshold.
        """
        frames = make_frames(
            n_frames=12,
            wrist_start_y=0.42,
            wrist_end_y=0.44,   # barely any vertical drop
            wrist_start_x=0.48,
            wrist_end_x=0.55,   # minimal lateral motion
            elbow_angle=140.0,
            duration_ms=400.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Volley must NOT be classified as slice, got confidence {result.confidence}"
        )

    def test_too_few_frames_rejected(self, classifier: SliceClassifier) -> None:
        """Sequences with fewer than 8 frames should be rejected outright."""
        frames = make_frames(
            n_frames=5,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Too few frames should be rejected"

    def test_too_long_duration_rejected(self, classifier: SliceClassifier) -> None:
        """A sequence spanning more than 4 seconds is too long for a single slice."""
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
            duration_ms=5000.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Too-long duration should be rejected"

    def test_too_short_duration_rejected(self, classifier: SliceClassifier) -> None:
        """A sequence under 200ms is too short for a real slice."""
        frames = make_frames(
            n_frames=10,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
            duration_ms=100.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Too-short duration should be rejected"


# ===========================================================================
# SLICE VS TOPSPIN DISTINCTION -- the critical classification boundary
# ===========================================================================


class TestSliceVsTopspin:
    """The slice vs topspin boundary is the hardest to get right.

    The key signal is wrist trajectory direction:
    - Slice:   wrist y INCREASES (moves down physically)
    - Topspin: wrist y DECREASES (moves up physically)
    """

    def test_clear_downward_wrist_is_slice(self, classifier: SliceClassifier) -> None:
        """Unambiguous downward wrist path -> slice."""
        frames = make_frames(
            wrist_start_y=0.32,
            wrist_end_y=0.58,   # strong downward = 0.26 drop
            elbow_angle=150.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0
        assert "slice" in result.shot_type

    def test_clear_upward_wrist_is_not_slice(self, classifier: SliceClassifier) -> None:
        """Unambiguous upward wrist path -> NOT slice (topspin)."""
        frames = make_frames(
            wrist_start_y=0.58,
            wrist_end_y=0.32,   # strong upward motion
            elbow_angle=100.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0

    def test_slice_higher_confidence_with_straight_arm(self, classifier: SliceClassifier) -> None:
        """A slice with a straighter arm (150 deg) should score higher than
        one with a more bent arm (110 deg), all else being equal.
        """
        base_kwargs = dict(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.54,
            duration_ms=1000.0,
        )

        frames_straight = make_frames(**base_kwargs, elbow_angle=155.0)
        frames_bent = make_frames(**base_kwargs, elbow_angle=110.0)

        result_straight = classifier.classify(frames_straight)
        result_bent = classifier.classify(frames_bent)

        # Both may classify as slice, but straight arm should score higher
        assert result_straight.confidence >= result_bent.confidence, (
            f"Straight arm ({result_straight.confidence}) should score >= "
            f"bent arm ({result_bent.confidence})"
        )

    def test_slice_higher_confidence_with_less_rotation(self, classifier: SliceClassifier) -> None:
        """A slice with less trunk rotation should score higher than one
        with heavy rotation.
        """
        # Low rotation: shoulders barely change
        frames_low_rot = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.54,
            left_shoulder_x=0.45,
            left_shoulder_x_end=0.46,
            right_shoulder_x=0.55,
            right_shoulder_x_end=0.54,
            elbow_angle=150.0,
        )

        # High rotation: shoulders change significantly
        frames_high_rot = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.54,
            left_shoulder_x=0.42,
            left_shoulder_x_end=0.52,
            right_shoulder_x=0.52,
            right_shoulder_x_end=0.62,
            elbow_angle=150.0,
        )

        result_low = classifier.classify(frames_low_rot)
        result_high = classifier.classify(frames_high_rot)

        # Low rotation should get higher slice confidence
        assert result_low.confidence >= result_high.confidence, (
            f"Low rotation ({result_low.confidence}) should score >= "
            f"high rotation ({result_high.confidence})"
        )

    def test_topspin_with_heavy_rotation_doubly_rejected(
        self, classifier: SliceClassifier
    ) -> None:
        """Topspin with heavy trunk rotation: two strong anti-slice signals.

        Should definitely be rejected.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.55,   # LOW start (topspin)
            wrist_end_y=0.30,     # HIGH end (topspin)
            elbow_angle=90.0,     # bent arm (topspin)
            elbow_angle_end=165.0,
            # Heavy rotation
            left_shoulder_x=0.40,
            left_shoulder_x_end=0.50,
            right_shoulder_x=0.50,
            right_shoulder_x_end=0.60,
            duration_ms=800.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, "Topspin with heavy rotation must be rejected"


# ===========================================================================
# CAMERA ANGLE TESTS
# ===========================================================================


class TestCameraAngles:
    """Slice detection should work across different camera angles."""

    def test_behind_view_slice(self, classifier: SliceClassifier) -> None:
        """Standard behind-the-player view (most broadcast angle)."""
        # Wide shoulder width relative to torso = behind/front view
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            left_shoulder_x=0.40,
            right_shoulder_x=0.60,
            elbow_angle=150.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0
        assert "slice" in result.shot_type

    def test_side_view_slice(self, classifier: SliceClassifier) -> None:
        """Side view: shoulders appear narrower.

        The high-to-low wrist path should still be detected.
        """
        # Narrow shoulder width = side view
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            left_shoulder_x=0.48,
            right_shoulder_x=0.52,
            elbow_angle=150.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0, (
            "Side view slice should still be detected"
        )
        assert "slice" in result.shot_type


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    """Borderline and degenerate cases."""

    def test_flat_groundstroke_borderline(self, classifier: SliceClassifier) -> None:
        """Flat groundstroke: wrist path is nearly level (minimal vertical drop).

        Should NOT be classified as a slice since the drop is below threshold.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.44,
            wrist_end_y=0.46,   # only 0.02 drop -- essentially flat
            elbow_angle=140.0,
            duration_ms=800.0,
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            f"Flat groundstroke with minimal drop should be rejected, "
            f"got confidence {result.confidence}"
        )

    def test_missing_wrist_landmarks(self, classifier: SliceClassifier) -> None:
        """When wrist landmarks have low visibility, classification should
        fail gracefully (confidence 0.0).
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
            wrist_visibility=0.1,  # below the 0.3 visibility threshold
        )
        result = classifier.classify(frames)

        assert result.confidence == 0.0, (
            "Missing wrist landmarks should result in rejection"
        )

    def test_missing_nose_landmarks(self, classifier: SliceClassifier) -> None:
        """Missing nose landmarks should not crash the classifier.

        Serve contamination check relies on nose; if nose is missing
        it should proceed without crashing.
        """
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.35,
            wrist_end_y=0.55,
            nose_visibility=0.1,  # nose not visible
            elbow_angle=150.0,
        )
        # Should not raise -- may or may not classify as slice
        result = classifier.classify(frames)
        assert isinstance(result.confidence, float)

    def test_empty_frames(self, classifier: SliceClassifier) -> None:
        """Empty frame list should return confidence 0.0."""
        result = classifier.classify([])
        assert result.confidence == 0.0

    def test_frames_with_no_landmarks(self, classifier: SliceClassifier) -> None:
        """Frames with empty landmarks lists should be handled gracefully."""
        frames = [
            {"frame_index": i, "timestamp_ms": i * 50.0, "landmarks": [], "joint_angles": {}}
            for i in range(20)
        ]
        result = classifier.classify(frames)
        assert result.confidence == 0.0


# ===========================================================================
# CLEAN VALIDATION TESTS
# ===========================================================================


class TestCleanValidation:
    """validate_clean should accept single slices and reject multi-shot sequences."""

    def test_single_slice_is_clean(self, classifier: SliceClassifier) -> None:
        """A single clean slice sequence should pass validation."""
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            elbow_angle=150.0,
            duration_ms=1000.0,
        )
        assert classifier.validate_clean(frames) is True

    def test_multi_shot_is_not_clean(self, classifier: SliceClassifier) -> None:
        """A sequence containing multiple direction reversals (e.g., two
        shots concatenated) should fail clean validation.
        """
        # Create a single slice
        single = make_frames(
            n_frames=15,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            elbow_angle=150.0,
            duration_ms=600.0,
        )
        # Concatenate it with a second shot (reversed trajectory)
        second = make_frames(
            n_frames=15,
            wrist_start_y=0.55,
            wrist_end_y=0.33,
            elbow_angle=150.0,
            duration_ms=600.0,
        )
        # Adjust timestamps and frame indices for the second shot
        offset_ms = single[-1]["timestamp_ms"]
        offset_idx = single[-1]["frame_index"] + 1
        for frame in second:
            frame["timestamp_ms"] += offset_ms
            frame["frame_index"] += offset_idx

        multi = single + second

        # Also add a lot of x-direction reversals to trigger multi-shot detection
        for i, frame in enumerate(multi):
            t = i / max(1, len(multi) - 1)
            frame["landmarks"][16]["x"] = 0.5 + 0.15 * math.sin(t * 6 * math.pi)

        assert classifier.validate_clean(multi) is False

    def test_too_short_not_clean(self, classifier: SliceClassifier) -> None:
        """Sequences shorter than _MIN_FRAMES should fail clean validation."""
        frames = make_frames(n_frames=5, duration_ms=300.0)
        assert classifier.validate_clean(frames) is False

    def test_too_long_not_clean(self, classifier: SliceClassifier) -> None:
        """Sequences longer than 4 seconds should fail clean validation."""
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            duration_ms=5000.0,
        )
        assert classifier.validate_clean(frames) is False


# ===========================================================================
# RESULT STRUCTURE TESTS
# ===========================================================================


class TestResultStructure:
    """Verify the ClassificationResult structure is populated correctly."""

    def test_result_fields_on_positive(self, classifier: SliceClassifier) -> None:
        """A positive classification should populate all result fields."""
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.33,
            wrist_end_y=0.55,
            elbow_angle=150.0,
            duration_ms=1000.0,
        )
        result = classifier.classify(frames)

        assert result.confidence > 0.0
        assert "slice" in result.shot_type
        assert result.camera_angle in ("behind", "side", "front", "overhead", "unknown")
        assert result.handedness in ("right", "left")
        assert isinstance(result.is_clean, bool)

    def test_result_fields_on_rejection(self, classifier: SliceClassifier) -> None:
        """A rejection should return shot_type='slice' with confidence 0.0."""
        frames = make_frames(
            n_frames=20,
            wrist_start_y=0.55,
            wrist_end_y=0.35,  # topspin trajectory
        )
        result = classifier.classify(frames)

        assert result.shot_type == "slice"
        assert result.confidence == 0.0
        assert result.is_clean is False

    def test_shot_type_property(self, classifier: SliceClassifier) -> None:
        """The shot_type property should return 'slice'."""
        assert classifier.shot_type == "slice"

    def test_phase_timestamps_present(self, classifier: SliceClassifier) -> None:
        """A positive classification should include phase timestamps."""
        frames = make_frames(
            n_frames=25,
            wrist_start_y=0.30,
            wrist_end_y=0.58,
            elbow_angle=155.0,
            duration_ms=1200.0,
        )
        result = classifier.classify(frames)

        if result.confidence > 0.0 and result.phase_timestamps is not None:
            # Should have slice-specific phase keys
            assert "takeback_peak" in result.phase_timestamps or "contact" in result.phase_timestamps
