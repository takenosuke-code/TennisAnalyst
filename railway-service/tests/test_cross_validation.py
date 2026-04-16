"""Cross-validation tests for the shot classification system.

Verifies that all 4 shot classifiers (forehand, backhand, serve, slice) work
TOGETHER correctly -- no shot type gets misclassified by multiple classifiers,
and the classify_shot() aggregator picks the correct winner.

Sections:
1. Mutual exclusivity -- each clear shot triggers only its classifier
2. classify_shot() aggregator -- returns highest confidence, handles edge cases
3. Cross-contamination -- clear shots do NOT trigger wrong classifiers
4. Camera angle consistency -- same shot, different angle, same classification
5. Handedness consistency -- right-handed and left-handed forehands both classify
6. Known issues -- slice vs volley / flat groundstroke documented as xfail
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shot_classifiers import (
    CLASSIFIERS,
    BackhandClassifier,
    ClassificationResult,
    ForehandClassifier,
    ServeClassifier,
    SliceClassifier,
    classify_shot,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
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


# ---------------------------------------------------------------------------
# Default neutral standing pose
# ---------------------------------------------------------------------------
def _default_positions() -> dict[int, tuple[float, float]]:
    return {
        0:  (0.50, 0.15),   # nose
        1:  (0.49, 0.13),   2:  (0.48, 0.13),   3:  (0.47, 0.13),
        4:  (0.51, 0.13),   5:  (0.52, 0.13),   6:  (0.53, 0.13),
        7:  (0.46, 0.14),   8:  (0.54, 0.14),
        9:  (0.49, 0.17),   10: (0.51, 0.17),
        11: (0.42, 0.25),   # left_shoulder
        12: (0.58, 0.25),   # right_shoulder
        13: (0.38, 0.35),   # left_elbow
        14: (0.62, 0.35),   # right_elbow
        15: (0.35, 0.45),   # left_wrist
        16: (0.65, 0.45),   # right_wrist
        17: (0.34, 0.47),   18: (0.66, 0.47),
        19: (0.34, 0.46),   20: (0.66, 0.46),
        21: (0.35, 0.46),   22: (0.65, 0.46),
        23: (0.45, 0.50),   # left_hip
        24: (0.55, 0.50),   # right_hip
        25: (0.44, 0.65),   # left_knee
        26: (0.56, 0.65),   # right_knee
        27: (0.43, 0.80),   # left_ankle
        28: (0.57, 0.80),   # right_ankle
        29: (0.43, 0.83),   30: (0.57, 0.83),
        31: (0.42, 0.85),   32: (0.58, 0.85),
    }


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Shared frame generator (mirrors pattern from individual test files)
# ---------------------------------------------------------------------------
def make_frames(
    num_frames: int = 20,
    fps: float = 30.0,
    *,
    animations: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    angle_start: dict[str, float] | None = None,
    angle_end: dict[str, float] | None = None,
    shoulder_width_start: float | None = None,
    shoulder_width_end: float | None = None,
    visibility: float = 0.9,
    visibility_overrides: dict[int, float] | None = None,
    nose_visibility: float | None = None,
) -> list[dict]:
    """Generate a synthetic sequence of MediaPipe pose frames.

    Linearly interpolates landmark positions, shoulder widths, and joint
    angles between start and end values over ``num_frames`` frames.
    """
    defaults = _default_positions()
    animations = animations or {}
    angle_start = angle_start or {}
    angle_end = angle_end or {}
    visibility_overrides = visibility_overrides or {}

    dt_ms = 1000.0 / fps
    frames: list[dict] = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        timestamp_ms = round(i * dt_ms, 1)

        landmarks: list[dict] = []
        for lm_id in range(33):
            if lm_id in animations:
                start_xy, end_xy = animations[lm_id]
                x = _lerp(start_xy[0], end_xy[0], t)
                y = _lerp(start_xy[1], end_xy[1], t)
            else:
                x, y = defaults[lm_id]

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
            if nose_visibility is not None and lm_id == LM_NOSE:
                vis = nose_visibility

            landmarks.append({
                "id": lm_id,
                "name": LANDMARK_NAMES[lm_id],
                "x": round(x, 4),
                "y": round(y, 4),
                "z": 0.0,
                "visibility": vis,
            })

        joint_angles: dict[str, float] = {}
        all_keys = set(angle_start.keys()) | set(angle_end.keys())
        for key in all_keys:
            s = angle_start.get(key, 90.0)
            e = angle_end.get(key, s)
            joint_angles[key] = round(_lerp(s, e, t), 1)

        frames.append({
            "frame_index": i,
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
            "joint_angles": joint_angles,
        })

    return frames


# ---------------------------------------------------------------------------
# Shot-specific frame generators
# ---------------------------------------------------------------------------

def _forehand_frames(
    num_frames: int = 20,
    fps: float = 30.0,
    *,
    shoulder_width_start: float = 0.12,
    shoulder_width_end: float = 0.20,
    nose_visibility: float = 0.3,
) -> list[dict]:
    """Right-handed forehand: right wrist sweeps R->L, stays below shoulder."""
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            LM_RIGHT_WRIST: ((0.70, 0.50), (0.30, 0.50)),
            LM_RIGHT_ELBOW: ((0.62, 0.35), (0.45, 0.35)),
        },
        angle_start={"right_elbow": 90.0, "left_elbow": 160.0, "trunk_rotation": 5.0},
        angle_end={"right_elbow": 150.0, "left_elbow": 160.0, "trunk_rotation": 45.0},
        shoulder_width_start=shoulder_width_start,
        shoulder_width_end=shoulder_width_end,
        nose_visibility=nose_visibility,
    )


def _backhand_frames(num_frames: int = 20, fps: float = 30.0) -> list[dict]:
    """Right-handed backhand: lead (left) wrist sweeps L->R, stays below shoulder.

    For a right-handed player the BackhandClassifier tracks the LEFT wrist
    as the lead wrist.  We animate the left wrist moving left-to-right
    (positive x displacement) which is the backhand swing direction.

    The right (dominant) wrist must have MORE lateral range than the left
    wrist so that detect_handedness() returns "right".  In a real backhand
    both hands grip the racket and swing together, so the dominant wrist
    also has substantial movement.
    """
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            # Left wrist (lead for right-handed backhand) moves L->R
            LM_LEFT_WRIST: ((0.30, 0.40), (0.60, 0.40)),
            LM_LEFT_ELBOW: ((0.34, 0.33), (0.52, 0.33)),
            # Right wrist (dominant) also swings L->R with >= range so
            # detect_handedness returns "right"
            LM_RIGHT_WRIST: ((0.28, 0.42), (0.65, 0.42)),
            LM_RIGHT_ELBOW: ((0.32, 0.35), (0.58, 0.35)),
        },
        angle_start={
            "left_elbow": 110.0,
            "right_elbow": 110.0,
            "trunk_rotation": 5.0,
        },
        angle_end={
            "left_elbow": 155.0,
            "right_elbow": 155.0,
            "trunk_rotation": 40.0,
        },
        shoulder_width_start=0.12,
        shoulder_width_end=0.20,
        nose_visibility=0.3,
    )


def _serve_frames(num_frames: int = 20, fps: float = 30.0) -> list[dict]:
    """Right-handed serve: right wrist rises well above head, knee bend present.

    The wrist starts near the waist and rises to well above the nose
    (y goes from ~0.45 down to ~0.02 in image coords).
    """
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            # Right wrist: starts at waist, rises well above head
            LM_RIGHT_WRIST: ((0.55, 0.45), (0.52, 0.02)),
            # Right elbow also rises
            LM_RIGHT_ELBOW: ((0.57, 0.35), (0.54, 0.12)),
            # Left wrist rises too (ball toss arm)
            LM_LEFT_WRIST: ((0.40, 0.45), (0.42, 0.10)),
            # Knee bend: right knee bends then extends
            LM_RIGHT_KNEE: ((0.56, 0.65), (0.56, 0.62)),
            LM_RIGHT_ANKLE: ((0.57, 0.80), (0.57, 0.78)),
        },
        angle_start={
            "right_elbow": 90.0,
            "left_elbow": 130.0,
            "right_knee": 170.0,
        },
        angle_end={
            "right_elbow": 170.0,
            "left_elbow": 130.0,
            "right_knee": 170.0,
        },
        shoulder_width_start=0.16,
        shoulder_width_end=0.16,
        nose_visibility=0.3,
    )


def _slice_frames(num_frames: int = 20, fps: float = 30.0) -> list[dict]:
    """Right-handed slice: wrist moves high-to-low (y increases in image coords).

    The wrist starts near shoulder height (y=0.30) and descends to below
    hip level (y=0.60), with a relatively straight arm and compact rotation.
    """
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        animations={
            # Right wrist: high-to-low path
            LM_RIGHT_WRIST: ((0.55, 0.30), (0.50, 0.60)),
            LM_RIGHT_ELBOW: ((0.57, 0.28), (0.52, 0.45)),
        },
        angle_start={"right_elbow": 150.0, "left_elbow": 130.0},
        angle_end={"right_elbow": 155.0, "left_elbow": 130.0},
        shoulder_width_start=0.16,
        shoulder_width_end=0.16,
        nose_visibility=0.3,
    )


def _standing_frames(num_frames: int = 20, fps: float = 30.0) -> list[dict]:
    """Player standing still: no significant motion in any landmark."""
    return make_frames(
        num_frames=num_frames,
        fps=fps,
        # No animations -- all landmarks stay at default neutral positions
        angle_start={"right_elbow": 160.0, "left_elbow": 160.0},
        angle_end={"right_elbow": 160.0, "left_elbow": 160.0},
        shoulder_width_start=0.16,
        shoulder_width_end=0.16,
        nose_visibility=0.3,
    )


# ---------------------------------------------------------------------------
# Helper to run all classifiers on a frame sequence
# ---------------------------------------------------------------------------

def _run_all_classifiers(frames: list[dict]) -> dict[str, ClassificationResult]:
    """Run every classifier on the same frames, return {name: result}."""
    return {name: clf.classify(frames) for name, clf in CLASSIFIERS.items()}


# ===========================================================================
# 1. MUTUAL EXCLUSIVITY TESTS
# ===========================================================================


class TestMutualExclusivity:
    """For each shot type, the correct classifier must have the highest
    confidence and no other classifier should claim it with high confidence."""

    def test_forehand_exclusivity(self) -> None:
        """A clear forehand should only trigger the forehand classifier."""
        frames = _forehand_frames()
        results = _run_all_classifiers(frames)

        forehand_conf = results["forehand"].confidence
        assert forehand_conf > 0.3, (
            f"Forehand classifier should detect forehand with confidence > 0.3, "
            f"got {forehand_conf}"
        )

        for name, result in results.items():
            if name == "forehand":
                continue
            assert result.confidence < forehand_conf, (
                f"Classifier '{name}' (conf={result.confidence}) should not "
                f"exceed forehand confidence ({forehand_conf})"
            )

    def test_backhand_exclusivity(self) -> None:
        """A clear backhand should only trigger the backhand classifier."""
        frames = _backhand_frames()
        results = _run_all_classifiers(frames)

        backhand_conf = results["backhand"].confidence
        assert backhand_conf > 0.3, (
            f"Backhand classifier should detect backhand with confidence > 0.3, "
            f"got {backhand_conf}"
        )

        for name, result in results.items():
            if name == "backhand":
                continue
            assert result.confidence < backhand_conf, (
                f"Classifier '{name}' (conf={result.confidence}) should not "
                f"exceed backhand confidence ({backhand_conf})"
            )

    def test_serve_exclusivity(self) -> None:
        """A clear serve should only trigger the serve classifier."""
        frames = _serve_frames()
        results = _run_all_classifiers(frames)

        serve_conf = results["serve"].confidence
        assert serve_conf > 0.3, (
            f"Serve classifier should detect serve with confidence > 0.3, "
            f"got {serve_conf}"
        )

        for name, result in results.items():
            if name == "serve":
                continue
            assert result.confidence < serve_conf, (
                f"Classifier '{name}' (conf={result.confidence}) should not "
                f"exceed serve confidence ({serve_conf})"
            )

    def test_slice_exclusivity(self) -> None:
        """A clear slice should only trigger the slice classifier."""
        frames = _slice_frames()
        results = _run_all_classifiers(frames)

        slice_conf = results["slice"].confidence
        assert slice_conf > 0.3, (
            f"Slice classifier should detect slice with confidence > 0.3, "
            f"got {slice_conf}"
        )

        for name, result in results.items():
            if name == "slice":
                continue
            assert result.confidence < slice_conf, (
                f"Classifier '{name}' (conf={result.confidence}) should not "
                f"exceed slice confidence ({slice_conf})"
            )

    def test_no_two_classifiers_both_high_on_forehand(self) -> None:
        """No two classifiers should both report > 0.5 on a forehand."""
        frames = _forehand_frames()
        results = _run_all_classifiers(frames)
        high_conf = {n: r.confidence for n, r in results.items() if r.confidence > 0.5}
        assert len(high_conf) <= 1, (
            f"Multiple classifiers with high confidence on forehand: {high_conf}"
        )


# ===========================================================================
# 2. classify_shot() AGGREGATOR TESTS
# ===========================================================================


class TestClassifyShotAggregator:
    """Test that classify_shot() returns the correct winner."""

    def test_aggregator_picks_forehand(self) -> None:
        """classify_shot() should return 'forehand' for a clear forehand."""
        frames = _forehand_frames()
        result = classify_shot(frames)
        assert result.shot_type == "forehand", (
            f"Expected 'forehand', got '{result.shot_type}'"
        )
        assert result.confidence > 0.0

    def test_aggregator_picks_backhand(self) -> None:
        """classify_shot() should return 'backhand' for a clear backhand."""
        frames = _backhand_frames()
        result = classify_shot(frames)
        assert result.shot_type == "backhand", (
            f"Expected 'backhand', got '{result.shot_type}'"
        )
        assert result.confidence > 0.0

    def test_aggregator_picks_serve(self) -> None:
        """classify_shot() should return 'serve' for a clear serve."""
        frames = _serve_frames()
        result = classify_shot(frames)
        assert result.shot_type == "serve", (
            f"Expected 'serve', got '{result.shot_type}'"
        )
        assert result.confidence > 0.0

    def test_aggregator_picks_slice(self) -> None:
        """classify_shot() should return a slice type for a clear slice."""
        frames = _slice_frames()
        result = classify_shot(frames)
        assert "slice" in result.shot_type, (
            f"Expected shot_type containing 'slice', got '{result.shot_type}'"
        )
        assert result.confidence > 0.0

    def test_aggregator_returns_unknown_for_standing(self) -> None:
        """classify_shot() should return 'unknown' with confidence 0 for
        a stationary player."""
        frames = _standing_frames()
        result = classify_shot(frames)
        assert result.shot_type == "unknown"
        assert result.confidence == 0.0

    def test_aggregator_returns_unknown_for_empty_frames(self) -> None:
        """classify_shot() should return 'unknown' with confidence 0 for
        an empty frame list."""
        result = classify_shot([])
        assert result.shot_type == "unknown"
        assert result.confidence == 0.0

    def test_aggregator_returns_highest_confidence(self) -> None:
        """When multiple classifiers fire, classify_shot() must return
        the result with the highest confidence."""
        frames = _forehand_frames()
        all_results = _run_all_classifiers(frames)
        aggregated = classify_shot(frames)

        # Find the actual highest confidence among all classifiers
        positive_results = {
            n: r for n, r in all_results.items() if r.confidence > 0
        }
        if positive_results:
            best_name = max(positive_results, key=lambda n: positive_results[n].confidence)
            best_conf = positive_results[best_name].confidence
            assert aggregated.confidence == best_conf, (
                f"Aggregator confidence {aggregated.confidence} does not match "
                f"best classifier '{best_name}' confidence {best_conf}"
            )

    def test_aggregator_result_is_valid_classification_result(self) -> None:
        """The aggregator should return a proper ClassificationResult."""
        result = classify_shot(_forehand_frames())
        assert isinstance(result, ClassificationResult)
        assert hasattr(result, "shot_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_clean")
        assert hasattr(result, "camera_angle")


# ===========================================================================
# 3. CROSS-CONTAMINATION TESTS
# ===========================================================================


class TestCrossContamination:
    """Verify that clear shots do NOT trigger the wrong classifiers."""

    def test_forehand_does_not_trigger_backhand(self) -> None:
        """A clear forehand should NOT be detected as a backhand."""
        frames = _forehand_frames()
        result = BackhandClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Backhand classifier should reject forehand, got conf={result.confidence}"
        )

    def test_forehand_does_not_trigger_serve(self) -> None:
        """A clear forehand should NOT be detected as a serve."""
        frames = _forehand_frames()
        result = ServeClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Serve classifier should reject forehand, got conf={result.confidence}"
        )

    def test_forehand_does_not_trigger_slice(self) -> None:
        """A clear forehand (level wrist path) should NOT be detected as a slice."""
        frames = _forehand_frames()
        result = SliceClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Slice classifier should reject forehand, got conf={result.confidence}"
        )

    def test_serve_does_not_trigger_forehand(self) -> None:
        """A clear serve should NOT be detected as a forehand.

        The forehand classifier checks wrist-below-shoulder and lateral
        displacement.  A serve has the wrist above head and minimal lateral
        sweep, so it should fail both checks.
        """
        frames = _serve_frames()
        result = ForehandClassifier().classify(frames)
        assert result.confidence < 0.3, (
            f"Forehand classifier should reject serve, got conf={result.confidence}"
        )

    def test_serve_does_not_trigger_backhand(self) -> None:
        """A clear serve should NOT be detected as a backhand."""
        frames = _serve_frames()
        result = BackhandClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Backhand classifier should reject serve, got conf={result.confidence}"
        )

    def test_serve_does_not_trigger_slice(self) -> None:
        """A clear serve should NOT be detected as a slice.

        The slice classifier has a serve contamination check that rejects
        sequences where the wrist goes well above the head.
        """
        frames = _serve_frames()
        result = SliceClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Slice classifier should reject serve, got conf={result.confidence}"
        )

    def test_slice_does_not_trigger_forehand(self) -> None:
        """A clear slice (high-to-low vertical path) should NOT be detected
        as a forehand.  The forehand classifier needs lateral displacement
        which a primarily vertical slice lacks.
        """
        frames = _slice_frames()
        result = ForehandClassifier().classify(frames)
        assert result.confidence < 0.3, (
            f"Forehand classifier should reject slice, got conf={result.confidence}"
        )

    def test_slice_does_not_trigger_serve(self) -> None:
        """A clear slice (wrist stays below shoulder) should NOT be detected
        as a serve (wrist must go above head)."""
        frames = _slice_frames()
        result = ServeClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Serve classifier should reject slice, got conf={result.confidence}"
        )

    # NOTE: A slice and a backhand can have borderline overlap because a
    # backhand slice has lateral motion similar to a backhand.  The backhand
    # classifier's _looks_like_slice() rejection should catch the high-to-low
    # trajectory, but this is documented as a known edge case.
    def test_slice_backhand_overlap_documented(self) -> None:
        """A backhand slice may trigger the backhand classifier at low
        confidence because of shared lateral direction.  The backhand
        classifier's slice rejection check should prevent high confidence.

        This test documents the borderline behavior rather than asserting
        zero confidence.
        """
        frames = _slice_frames()
        bh_result = BackhandClassifier().classify(frames)
        sl_result = SliceClassifier().classify(frames)

        # The slice classifier should win if both fire
        if bh_result.confidence > 0:
            assert sl_result.confidence > bh_result.confidence, (
                f"When both fire, slice ({sl_result.confidence}) should beat "
                f"backhand ({bh_result.confidence}) on a slice input"
            )

    def test_standing_triggers_none(self) -> None:
        """A player standing still should trigger NONE of the classifiers."""
        frames = _standing_frames()
        results = _run_all_classifiers(frames)

        for name, result in results.items():
            assert result.confidence == 0.0, (
                f"Classifier '{name}' should return 0 confidence for standing, "
                f"got {result.confidence}"
            )


# ===========================================================================
# 4. CAMERA ANGLE CONSISTENCY TESTS
# ===========================================================================


class TestCameraAngleConsistency:
    """The same shot from different camera angles should produce the same
    shot type classification from the aggregator."""

    def test_forehand_behind_vs_side(self) -> None:
        """A forehand from behind view and side view should both classify
        as forehand."""
        behind = _forehand_frames(
            shoulder_width_start=0.18,
            shoulder_width_end=0.22,
            nose_visibility=0.3,
        )
        side = _forehand_frames(
            shoulder_width_start=0.03,
            shoulder_width_end=0.05,
            nose_visibility=0.3,
        )

        result_behind = classify_shot(behind)
        result_side = classify_shot(side)

        assert result_behind.shot_type == "forehand"
        assert result_side.shot_type == "forehand"

    def test_forehand_behind_vs_front(self) -> None:
        """A forehand from behind view and front view should both classify
        as forehand.  The front view has reversed lateral direction but
        the classifier adapts via camera angle detection.
        """
        behind = _forehand_frames(
            shoulder_width_start=0.18,
            shoulder_width_end=0.22,
            nose_visibility=0.3,
        )

        # Front view: right-handed forehand appears as L->R on screen,
        # wide shoulders, highly visible nose.
        front = make_frames(
            num_frames=20,
            animations={
                LM_RIGHT_WRIST: ((0.30, 0.50), (0.70, 0.50)),
                LM_RIGHT_ELBOW: ((0.45, 0.35), (0.62, 0.35)),
            },
            angle_start={"right_elbow": 90.0, "trunk_rotation": 5.0},
            angle_end={"right_elbow": 150.0, "trunk_rotation": 45.0},
            shoulder_width_start=0.24,
            shoulder_width_end=0.24,
            nose_visibility=0.95,
        )

        result_behind = classify_shot(behind)
        result_front = classify_shot(front)

        assert result_behind.shot_type == "forehand"
        assert result_front.shot_type == "forehand"

    def test_serve_behind_vs_side(self) -> None:
        """A serve from behind view and side view should both classify
        as serve.  The primary signal (wrist above shoulder) is visible
        from both angles."""
        behind = _serve_frames()

        # Side view: narrow shoulders
        side = make_frames(
            num_frames=20,
            animations={
                LM_RIGHT_WRIST: ((0.52, 0.45), (0.51, 0.02)),
                LM_RIGHT_ELBOW: ((0.53, 0.35), (0.52, 0.12)),
                LM_LEFT_WRIST: ((0.48, 0.45), (0.49, 0.10)),
                LM_RIGHT_KNEE: ((0.52, 0.65), (0.52, 0.62)),
                LM_RIGHT_ANKLE: ((0.52, 0.80), (0.52, 0.78)),
            },
            angle_start={"right_elbow": 90.0, "right_knee": 170.0},
            angle_end={"right_elbow": 170.0, "right_knee": 170.0},
            shoulder_width_start=0.04,
            shoulder_width_end=0.04,
            nose_visibility=0.3,
        )

        result_behind = classify_shot(behind)
        result_side = classify_shot(side)

        assert result_behind.shot_type == "serve"
        assert result_side.shot_type == "serve"


# ===========================================================================
# 5. HANDEDNESS CONSISTENCY TESTS
# ===========================================================================


class TestHandednessConsistency:
    """A right-handed and left-handed forehand should both be classified
    as 'forehand' by the aggregator."""

    def test_right_handed_forehand(self) -> None:
        """A right-handed forehand (right wrist R->L) should classify as
        forehand with handedness='right'."""
        frames = _forehand_frames()
        result = classify_shot(frames)

        assert result.shot_type == "forehand"
        assert result.handedness == "right"

    def test_left_handed_forehand(self) -> None:
        """A left-handed forehand (left wrist L->R) should classify as
        forehand with handedness='left'."""
        frames = make_frames(
            num_frames=20,
            animations={
                LM_LEFT_WRIST: ((0.30, 0.50), (0.70, 0.50)),
                LM_LEFT_ELBOW: ((0.38, 0.35), (0.55, 0.35)),
                # Keep right wrist still so handedness detects left
                LM_RIGHT_WRIST: ((0.60, 0.45), (0.60, 0.45)),
            },
            angle_start={"left_elbow": 90.0, "right_elbow": 160.0, "trunk_rotation": 5.0},
            angle_end={"left_elbow": 150.0, "right_elbow": 160.0, "trunk_rotation": 45.0},
            shoulder_width_start=0.12,
            shoulder_width_end=0.20,
            nose_visibility=0.3,
        )
        result = classify_shot(frames)

        assert result.shot_type == "forehand"
        assert result.handedness == "left"

    def test_both_hands_same_shot_type(self) -> None:
        """Both right-handed and left-handed forehands should produce
        the same shot_type='forehand' from the aggregator."""
        right = _forehand_frames()
        left = make_frames(
            num_frames=20,
            animations={
                LM_LEFT_WRIST: ((0.30, 0.50), (0.70, 0.50)),
                LM_LEFT_ELBOW: ((0.38, 0.35), (0.55, 0.35)),
                LM_RIGHT_WRIST: ((0.60, 0.45), (0.60, 0.45)),
            },
            angle_start={"left_elbow": 90.0, "right_elbow": 160.0, "trunk_rotation": 5.0},
            angle_end={"left_elbow": 150.0, "right_elbow": 160.0, "trunk_rotation": 45.0},
            shoulder_width_start=0.12,
            shoulder_width_end=0.20,
            nose_visibility=0.3,
        )

        result_right = classify_shot(right)
        result_left = classify_shot(left)

        assert result_right.shot_type == result_left.shot_type == "forehand"


# ===========================================================================
# 6. KNOWN ISSUES (documented as xfail)
# ===========================================================================


class TestKnownIssues:
    """Document known classification difficulties.

    The slice classifier currently has issues distinguishing slices from
    volleys and flat groundstrokes in certain edge cases.  These are
    documented here as expected failures.
    """

    @pytest.mark.xfail(
        reason="Slice classifier may misclassify a compact volley with slight "
               "downward wrist path as a slice.  The volley has too little "
               "vertical drop to meet the threshold in most cases, but edge "
               "cases near the boundary can leak through.",
        strict=False,
    )
    def test_slice_vs_volley_boundary(self) -> None:
        """A volley with a slight downward wrist path (y increase of ~0.04)
        is near the slice threshold boundary.  The slice classifier should
        reject it, but may not when the torso is short (making the normalized
        drop appear larger).
        """
        # Short torso makes normalized delta larger
        frames = make_frames(
            num_frames=12,
            animations={
                LM_RIGHT_WRIST: ((0.52, 0.38), (0.56, 0.42)),
                LM_RIGHT_ELBOW: ((0.54, 0.34), (0.55, 0.38)),
                # Short torso: shoulders and hips close together
                LM_LEFT_SHOULDER: ((0.46, 0.32), (0.46, 0.32)),
                LM_RIGHT_SHOULDER: ((0.54, 0.32), (0.54, 0.32)),
                LM_LEFT_HIP: ((0.47, 0.40), (0.47, 0.40)),
                LM_RIGHT_HIP: ((0.53, 0.40), (0.53, 0.40)),
            },
            angle_start={"right_elbow": 145.0},
            angle_end={"right_elbow": 148.0},
        )
        result = SliceClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Volley should not be classified as slice, got conf={result.confidence}"
        )

    @pytest.mark.xfail(
        reason="Flat groundstrokes with modest downward angle near the "
               "MIN_HIGH_TO_LOW_RATIO threshold can be misclassified as "
               "slice when arm extension is high and rotation is low.",
        strict=False,
    )
    def test_slice_vs_flat_groundstroke_boundary(self) -> None:
        """A flat groundstroke with a slight downward angle (not truly a
        slice) near the classification threshold.  Ideally rejected, but
        the current classifier can be fooled when the elbow is very straight
        and trunk rotation is minimal.
        """
        frames = make_frames(
            num_frames=20,
            animations={
                # Slight downward path -- just above the typical rejection threshold
                LM_RIGHT_WRIST: ((0.55, 0.38), (0.45, 0.44)),
                LM_RIGHT_ELBOW: ((0.57, 0.33), (0.48, 0.38)),
            },
            angle_start={"right_elbow": 155.0},
            angle_end={"right_elbow": 158.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = SliceClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Flat groundstroke should not be classified as slice, "
            f"got conf={result.confidence}"
        )

    @pytest.mark.xfail(
        reason="Slice classifier does not have a dedicated volley rejection "
               "heuristic beyond the minimum high-to-low threshold.  A volley "
               "with moderate downward motion and a straight arm can score "
               "above the confidence threshold.",
        strict=False,
    )
    def test_slice_vs_approach_volley(self) -> None:
        """An approach volley with a moderate downward punch motion can
        resemble a slice.  The compact swing and straight arm match slice
        signals even though it is a different shot type.
        """
        frames = make_frames(
            num_frames=15,
            animations={
                LM_RIGHT_WRIST: ((0.50, 0.32), (0.55, 0.50)),
                LM_RIGHT_ELBOW: ((0.52, 0.30), (0.54, 0.40)),
            },
            angle_start={"right_elbow": 155.0},
            angle_end={"right_elbow": 158.0},
            shoulder_width_start=0.16,
            shoulder_width_end=0.16,
        )
        result = SliceClassifier().classify(frames)
        assert result.confidence == 0.0, (
            f"Approach volley should not be classified as slice, "
            f"got conf={result.confidence}"
        )


# ===========================================================================
# 7. ADDITIONAL EDGE CASES
# ===========================================================================


class TestAdditionalEdgeCases:
    """Additional cross-system edge cases."""

    def test_all_classifiers_return_zero_yields_unknown(self) -> None:
        """When every classifier returns confidence 0, classify_shot()
        should return 'unknown'."""
        frames = _standing_frames()
        result = classify_shot(frames)
        assert result.shot_type == "unknown"
        assert result.confidence == 0.0
        assert result.is_clean is False

    def test_confidence_is_always_bounded(self) -> None:
        """Every classifier's confidence should be in [0.0, 1.0] for all
        shot types."""
        all_generators = [
            _forehand_frames,
            _backhand_frames,
            _serve_frames,
            _slice_frames,
            _standing_frames,
        ]
        for gen in all_generators:
            frames = gen()
            results = _run_all_classifiers(frames)
            for name, result in results.items():
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Classifier '{name}' returned confidence {result.confidence} "
                    f"outside [0, 1] for {gen.__name__}"
                )

    def test_aggregator_result_always_has_camera_angle(self) -> None:
        """The aggregated result should always have a camera_angle field."""
        for gen in [_forehand_frames, _backhand_frames, _serve_frames,
                    _slice_frames, _standing_frames]:
            result = classify_shot(gen())
            assert result.camera_angle in (
                "behind", "side", "front", "overhead", "unknown"
            ), f"Invalid camera_angle '{result.camera_angle}' for {gen.__name__}"
