"""Slice shot classifier.

Detects forehand and backhand slice shots by identifying the defining
high-to-low swing path, open racket face (straighter arm), and compact
body rotation that distinguish slices from topspin groundstrokes, serves,
and volleys.

Only slice shots pass this classifier -- all other shot types are rejected
with confidence 0.0.

In image coordinates y=0 is the top of the frame and y increases downward.
A physically high-to-low swing (the slice signature) means the wrist starts
at a LOW y value (high position) and moves to a HIGH y value (low position),
so the wrist y-coordinate INCREASES during the swing phase.
"""

from __future__ import annotations

import math

from .base import (
    BaseShotClassifier,
    ClassificationResult,
    LM_LEFT_ELBOW,
    LM_LEFT_HIP,
    LM_LEFT_SHOULDER,
    LM_LEFT_WRIST,
    LM_NOSE,
    LM_RIGHT_ELBOW,
    LM_RIGHT_HIP,
    LM_RIGHT_SHOULDER,
    LM_RIGHT_WRIST,
    detect_handedness,
    distance,
    estimate_camera_angle,
    lm_xy,
    midpoint,
)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Minimum number of frames for a valid slice sequence.
_MIN_FRAMES = 8

# Duration bounds (ms) for a single slice shot.
_MIN_DURATION_MS = 200
_MAX_DURATION_MS = 4000

# The wrist y must increase by at least this fraction of the torso length
# during the swing to count as a high-to-low path.
_MIN_HIGH_TO_LOW_RATIO = 0.10

# If the wrist rises above the nose y by more than this fraction of the
# torso length, the sequence is likely a serve, not a slice.
_SERVE_CONTAMINATION_RATIO = 0.15

# Maximum trunk rotation change (radians) expected in a slice.  Slices use
# less rotation than full topspin groundstrokes.
_MAX_TRUNK_ROTATION_RAD = 0.55

# Elbow angle above which the arm is considered "straight enough" for a
# slice (degrees).  Slices keep the arm more extended than topspin shots.
_STRAIGHT_ARM_ANGLE_DEG = 120.0

# High-to-low delta tiers for confidence scoring.
_DELTA_STRONG = 0.20
_DELTA_MODERATE = 0.12

# Confidence component weights (must sum to 1.0).
_W_PRIMARY = 0.55   # high-to-low wrist path
_W_TRUNK = 0.20     # low trunk rotation
_W_ELBOW = 0.15     # straight arm
_W_COMPACT = 0.10   # compact motion

# Below this composite confidence the shot is rejected as "not a slice".
_MIN_CONFIDENCE_THRESHOLD = 0.30

# Minimum total wrist travel distance (normalized by torso length) to
# reject volleys and other ultra-compact motions that are not slices.
_MIN_WRIST_TRAVEL_RATIO = 0.5

# Maximum wrist y-trajectory direction changes before multi-shot rejection.
# Combined x + y direction changes are counted; a single clean slice should
# have very few reversals across both axes.
_MAX_DIRECTION_CHANGES = 3


class SliceClassifier(BaseShotClassifier):
    """Classifier that detects forehand and backhand slice shots.

    The detection algorithm:

    1. Determine handedness (which hand is dominant).
    2. Track the dominant wrist y-coordinate across the frame sequence.
    3. Verify a high-to-low swing path:
       - The wrist starts at low y (physically high) and moves to high y
         (physically low) during the swing phase.
       - This is the opposite of topspin groundstrokes (low-to-high) and
         distinguishes slices from every other shot type.
    4. Compute a weighted confidence from four signals:
       - High-to-low wrist delta, normalized by torso length (55%)
       - Low trunk rotation change (20%)
       - Straighter arm / larger elbow angle (15%)
       - Compact swing arc (10%)
    5. Reject serve contamination (wrist above head), topspin (upward
       trajectory), and multi-shot sequences.
    6. Determine forehand vs backhand side from wrist position relative
       to body midline.

    ``validate_clean`` rejects sequences with upward wrist trajectory,
    serve contamination, multi-shot direction changes, and sequences
    that are too short or too long.
    """

    # -- BaseShotClassifier interface ----------------------------------------

    @property
    def shot_type(self) -> str:
        """Return the shot type this classifier detects."""
        return "slice"

    def classify(self, frames: list[dict]) -> ClassificationResult:
        """Classify a sequence of pose frames as a slice or not.

        Returns a ``ClassificationResult`` with ``shot_type`` set to
        ``'forehand_slice'`` or ``'backhand_slice'``.  If the sequence
        is not a slice, ``confidence`` will be 0.0.
        """
        camera_angle = estimate_camera_angle(frames)
        handedness = detect_handedness(frames)

        if len(frames) < _MIN_FRAMES:
            return self._reject(camera_angle, handedness)

        duration_ms = (
            frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        )
        if duration_ms < _MIN_DURATION_MS or duration_ms > _MAX_DURATION_MS:
            return self._reject(camera_angle, handedness)

        wrist_idx, elbow_idx, shoulder_idx = self._arm_indices(handedness)

        # -- Extract wrist trajectory and reference measurements ----------
        wrist_ys = self._extract_wrist_y(frames, wrist_idx)
        if len(wrist_ys) < _MIN_FRAMES:
            return self._reject(camera_angle, handedness)

        torso_len = self._median_torso_length(frames)
        if torso_len < 0.01:
            return self._reject(camera_angle, handedness)

        # -- Serve contamination check ------------------------------------
        if self._has_serve_contamination(frames, wrist_idx, torso_len):
            return self._reject(camera_angle, handedness)

        # -- Primary signal: high-to-low wrist delta ----------------------
        high_to_low_delta = self._compute_high_to_low_delta(wrist_ys, torso_len)
        if high_to_low_delta < _MIN_HIGH_TO_LOW_RATIO:
            return self._reject(camera_angle, handedness)

        # -- Volley rejection: minimum wrist travel distance ---------------
        wrist_travel = self._compute_wrist_travel(frames, wrist_idx, torso_len)
        if wrist_travel < _MIN_WRIST_TRAVEL_RATIO:
            return self._reject(camera_angle, handedness)

        # -- Secondary signals --------------------------------------------
        trunk_rotation = self._compute_trunk_rotation(frames)
        avg_elbow_angle = self._compute_avg_elbow_angle(
            frames, elbow_idx, shoulder_idx, wrist_idx
        )
        compactness = self._compute_compactness(frames, wrist_idx, torso_len)

        # -- Confidence scoring -------------------------------------------
        confidence = self._score_confidence(
            high_to_low_delta, trunk_rotation, avg_elbow_angle, compactness
        )

        if confidence < _MIN_CONFIDENCE_THRESHOLD:
            return self._reject(camera_angle, handedness)

        is_clean = self.validate_clean(frames)
        side = self._determine_side(frames, handedness, wrist_idx)
        phase_timestamps = self._detect_phases(wrist_ys, frames)

        return ClassificationResult(
            shot_type=f"{side}_slice",
            confidence=round(confidence, 3),
            is_clean=is_clean,
            camera_angle=camera_angle,
            handedness=handedness,
            phase_timestamps=phase_timestamps,
        )

    def validate_clean(self, frames: list[dict]) -> bool:
        """Check whether the sequence contains a single clean slice shot.

        Rejects sequences that:
        - Have fewer than ``_MIN_FRAMES`` frames.
        - Span outside the ``_MIN_DURATION_MS`` / ``_MAX_DURATION_MS`` window.
        - Show an upward wrist trajectory (topspin, not slice).
        - Show the wrist above the head (serve contamination).
        - Contain too many y-direction changes (multi-shot contamination).
        """
        if len(frames) < _MIN_FRAMES:
            return False

        duration_ms = (
            frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        )
        if duration_ms < _MIN_DURATION_MS or duration_ms > _MAX_DURATION_MS:
            return False

        handedness = detect_handedness(frames)
        wrist_idx, _, _ = self._arm_indices(handedness)

        wrist_ys = self._extract_wrist_y(frames, wrist_idx)
        if len(wrist_ys) < _MIN_FRAMES:
            return False

        torso_len = self._median_torso_length(frames)
        if torso_len < 0.01:
            return False

        # Reject upward trajectory (topspin, not slice)
        if self._compute_high_to_low_delta(wrist_ys, torso_len) < _MIN_HIGH_TO_LOW_RATIO:
            return False

        # Reject serve contamination
        if self._has_serve_contamination(frames, wrist_idx, torso_len):
            return False

        # Reject multi-shot sequences (check both x and y direction changes)
        wrist_xs = self._extract_wrist_x(frames, wrist_idx)
        total_changes = (
            self._count_direction_changes(wrist_ys)
            + self._count_direction_changes(wrist_xs)
        )
        if total_changes > _MAX_DIRECTION_CHANGES:
            return False

        return True

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _arm_indices(handedness: str) -> tuple[int, int, int]:
        """Return (wrist, elbow, shoulder) landmark indices for the dominant arm."""
        if handedness == "left":
            return LM_LEFT_WRIST, LM_LEFT_ELBOW, LM_LEFT_SHOULDER
        return LM_RIGHT_WRIST, LM_RIGHT_ELBOW, LM_RIGHT_SHOULDER

    @staticmethod
    def _reject(camera_angle: str, handedness: str) -> ClassificationResult:
        """Return a zero-confidence rejection result."""
        return ClassificationResult(
            shot_type="slice",
            confidence=0.0,
            is_clean=False,
            camera_angle=camera_angle,
            handedness=handedness,
        )

    # -- Trajectory extraction -----------------------------------------------

    @staticmethod
    def _extract_wrist_y(frames: list[dict], wrist_idx: int) -> list[float]:
        """Extract the y-coordinates of the dominant wrist across all frames.

        Skips frames where the wrist is not visible (visibility too low).
        """
        ys: list[float] = []
        for frame in frames:
            pos = lm_xy(frame, wrist_idx)
            if pos is not None:
                ys.append(pos[1])
        return ys

    @staticmethod
    def _extract_wrist_x(frames: list[dict], wrist_idx: int) -> list[float]:
        """Extract the x-coordinates of the dominant wrist across all frames.

        Skips frames where the wrist is not visible (visibility too low).
        """
        xs: list[float] = []
        for frame in frames:
            pos = lm_xy(frame, wrist_idx)
            if pos is not None:
                xs.append(pos[0])
        return xs

    # -- Torso reference measurement -----------------------------------------

    @staticmethod
    def _median_torso_length(frames: list[dict]) -> float:
        """Compute the median shoulder-midpoint-to-hip-midpoint distance.

        Provides a body-relative scale so thresholds work regardless of the
        player's distance from the camera.
        """
        lengths: list[float] = []
        sample_step = max(1, len(frames) // 10)
        for frame in frames[::sample_step]:
            ls = lm_xy(frame, LM_LEFT_SHOULDER)
            rs = lm_xy(frame, LM_RIGHT_SHOULDER)
            lh = lm_xy(frame, LM_LEFT_HIP)
            rh = lm_xy(frame, LM_RIGHT_HIP)
            if ls and rs and lh and rh:
                shoulder_mid = midpoint(ls, rs)
                hip_mid = midpoint(lh, rh)
                lengths.append(distance(shoulder_mid, hip_mid))

        if not lengths:
            return 0.0

        lengths.sort()
        return lengths[len(lengths) // 2]

    # -- Serve contamination -------------------------------------------------

    @staticmethod
    def _has_serve_contamination(
        frames: list[dict], wrist_idx: int, torso_len: float
    ) -> bool:
        """Return True if the wrist goes well above the head.

        In image coords, "above" means smaller y.  If the wrist y is
        significantly less than the nose y the wrist is above the head —
        a serve signal, not a slice.
        """
        threshold = torso_len * _SERVE_CONTAMINATION_RATIO
        for frame in frames:
            wrist = lm_xy(frame, wrist_idx)
            nose = lm_xy(frame, LM_NOSE)
            if wrist and nose:
                if (nose[1] - wrist[1]) > threshold:
                    return True
        return False

    # -- Primary signal: high-to-low swing delta -----------------------------

    @staticmethod
    def _compute_high_to_low_delta(wrist_ys: list[float], torso_len: float) -> float:
        """Measure the high-to-low wrist path, normalized by torso length.

        Finds the swing phase by locating the minimum wrist y (highest
        physical position) in the first 75% of the sequence, then the
        maximum wrist y (lowest physical position) after that point.
        The delta between them, divided by torso length, quantifies how
        much the wrist descended.

        Returns a ratio >= 0.  Larger values indicate a stronger
        high-to-low path (more slice-like).
        """
        if len(wrist_ys) < 3 or torso_len < 0.01:
            return 0.0

        # Preparation phase: find the frame where the wrist is highest
        # (minimum y) in the first 75% of the sequence.
        search_end = max(2, int(len(wrist_ys) * 0.75))
        min_y = wrist_ys[0]
        min_y_idx = 0
        for i in range(1, search_end):
            if wrist_ys[i] < min_y:
                min_y = wrist_ys[i]
                min_y_idx = i

        # Swing/contact phase: find the frame where the wrist is lowest
        # (maximum y) after the preparation peak.
        max_y_after = min_y
        for i in range(min_y_idx, len(wrist_ys)):
            if wrist_ys[i] > max_y_after:
                max_y_after = wrist_ys[i]

        delta = max_y_after - min_y  # positive = wrist moved down physically
        return delta / torso_len

    # -- Secondary signal: trunk rotation ------------------------------------

    @staticmethod
    def _compute_trunk_rotation(frames: list[dict]) -> float:
        """Estimate total trunk rotation change across the sequence.

        Uses the angle between the shoulder line and hip line projected
        onto the x-axis.  Slices produce noticeably less rotation than
        full topspin groundstrokes.

        Returns the absolute rotation range in radians.
        """
        angles: list[float] = []
        sample_step = max(1, len(frames) // 15)
        for frame in frames[::sample_step]:
            ls = lm_xy(frame, LM_LEFT_SHOULDER)
            rs = lm_xy(frame, LM_RIGHT_SHOULDER)
            lh = lm_xy(frame, LM_LEFT_HIP)
            rh = lm_xy(frame, LM_RIGHT_HIP)
            if not (ls and rs and lh and rh):
                continue

            shoulder_angle = math.atan2(ls[1] - rs[1], ls[0] - rs[0])
            hip_angle = math.atan2(lh[1] - rh[1], lh[0] - rh[0])
            angles.append(shoulder_angle - hip_angle)

        if len(angles) < 2:
            return 0.0

        return max(angles) - min(angles)

    # -- Secondary signal: elbow extension -----------------------------------

    @staticmethod
    def _compute_avg_elbow_angle(
        frames: list[dict],
        elbow_idx: int,
        shoulder_idx: int,
        wrist_idx: int,
    ) -> float:
        """Compute the average elbow angle across the sequence.

        Prefers the pre-computed ``joint_angles`` dict.  Falls back to
        computing the angle from landmark positions.  Returns degrees.
        """
        angles: list[float] = []
        angle_key = "left_elbow" if elbow_idx == LM_LEFT_ELBOW else "right_elbow"

        for frame in frames:
            # Prefer pre-computed joint angle
            ja = frame.get("joint_angles", {})
            if angle_key in ja and ja[angle_key] is not None:
                angles.append(float(ja[angle_key]))
                continue

            # Fallback: compute from landmarks
            s = lm_xy(frame, shoulder_idx)
            e = lm_xy(frame, elbow_idx)
            w = lm_xy(frame, wrist_idx)
            if s and e and w:
                angles.append(_angle_at_vertex(s, e, w))

        if not angles:
            return 90.0  # neutral fallback

        return sum(angles) / len(angles)

    # -- Volley rejection: wrist travel distance ------------------------------

    @staticmethod
    def _compute_wrist_travel(
        frames: list[dict], wrist_idx: int, torso_len: float
    ) -> float:
        """Compute total wrist travel distance normalized by torso length.

        Volleys have very short wrist travel (< 0.5 torso lengths) compared
        to slices (typically 1.5 - 3.0 torso lengths).
        """
        if torso_len < 0.01:
            return 0.0

        positions: list[tuple[float, float]] = []
        for frame in frames:
            pos = lm_xy(frame, wrist_idx)
            if pos is not None:
                positions.append(pos)

        if len(positions) < 2:
            return 0.0

        total_travel = sum(
            distance(positions[i - 1], positions[i])
            for i in range(1, len(positions))
        )
        return total_travel / torso_len

    # -- Secondary signal: compactness ---------------------------------------

    @staticmethod
    def _compute_compactness(
        frames: list[dict], wrist_idx: int, torso_len: float
    ) -> float:
        """Measure how compact the swing arc is.

        Returns a value in [0, 1] where 1 is maximally compact.  Uses
        total wrist travel distance normalized by torso length.  Shorter
        travel = more compact = more slice-like.

        Typical ranges (travel / torso_len):
        - Volley: < 1.5  (very compact, but we reject volleys elsewhere)
        - Slice:  1.5 - 3.0
        - Full groundstroke: 3.0 - 5.0+
        """
        if torso_len < 0.01:
            return 0.5

        positions: list[tuple[float, float]] = []
        for frame in frames:
            pos = lm_xy(frame, wrist_idx)
            if pos is not None:
                positions.append(pos)

        if len(positions) < 2:
            return 0.5

        total_travel = sum(
            distance(positions[i - 1], positions[i])
            for i in range(1, len(positions))
        )
        normalized = total_travel / torso_len

        # Map to [0, 1]: <=2 is very compact (1.0), >=4 is not compact (0.0)
        if normalized <= 2.0:
            return 1.0
        if normalized >= 4.0:
            return 0.0
        return 1.0 - (normalized - 2.0) / 2.0

    # -- Multi-shot detection ------------------------------------------------

    @staticmethod
    def _count_direction_changes(ys: list[float]) -> int:
        """Count the number of times the wrist y-trajectory reverses direction.

        A single slice should have at most a few direction changes (up
        during takeback, down during swing, possibly a small follow-through
        correction).  Many changes indicate multiple shots.
        """
        if len(ys) < 3:
            return 0

        # Smooth with a 3-point moving average to reduce noise
        smoothed: list[float] = []
        for i in range(len(ys)):
            start = max(0, i - 1)
            end = min(len(ys), i + 2)
            smoothed.append(sum(ys[start:end]) / (end - start))

        changes = 0
        prev_direction: float = 0.0
        for i in range(1, len(smoothed)):
            diff = smoothed[i] - smoothed[i - 1]
            if abs(diff) < 0.005:
                continue  # ignore noise
            direction = 1.0 if diff > 0 else -1.0
            if prev_direction != 0.0 and direction != prev_direction:
                changes += 1
            prev_direction = direction

        return changes

    # -- Confidence scoring --------------------------------------------------

    @staticmethod
    def _score_confidence(
        high_to_low_delta: float,
        trunk_rotation: float,
        avg_elbow_angle: float,
        compactness: float,
    ) -> float:
        """Combine biomechanical signals into a single confidence score.

        Primary signal (55%): magnitude of the high-to-low wrist path.
        Secondary signals: low trunk rotation (20%), straight arm (15%),
        compact swing (10%).
        """
        # -- Primary: high-to-low delta ----------------------------------
        if high_to_low_delta >= _DELTA_STRONG:
            primary = 1.0
        elif high_to_low_delta >= _DELTA_MODERATE:
            primary = 0.6 + 0.4 * (
                (high_to_low_delta - _DELTA_MODERATE)
                / (_DELTA_STRONG - _DELTA_MODERATE)
            )
        else:
            # Between _MIN_HIGH_TO_LOW_RATIO and _DELTA_MODERATE
            primary = 0.3 + 0.3 * (
                (high_to_low_delta - _MIN_HIGH_TO_LOW_RATIO)
                / max(0.001, _DELTA_MODERATE - _MIN_HIGH_TO_LOW_RATIO)
            )

        # -- Secondary: trunk rotation (less = more slice-like) ----------
        if trunk_rotation <= 0.15:
            trunk_score = 1.0
        elif trunk_rotation >= _MAX_TRUNK_ROTATION_RAD:
            trunk_score = 0.0
        else:
            trunk_score = 1.0 - (trunk_rotation - 0.15) / (
                _MAX_TRUNK_ROTATION_RAD - 0.15
            )

        # -- Secondary: elbow angle (straighter = more slice-like) -------
        if avg_elbow_angle >= 160.0:
            elbow_score = 1.0
        elif avg_elbow_angle >= _STRAIGHT_ARM_ANGLE_DEG:
            elbow_score = 0.5 + 0.5 * (
                (avg_elbow_angle - _STRAIGHT_ARM_ANGLE_DEG)
                / (160.0 - _STRAIGHT_ARM_ANGLE_DEG)
            )
        elif avg_elbow_angle >= 90.0:
            elbow_score = 0.2 + 0.3 * (
                (avg_elbow_angle - 90.0)
                / (_STRAIGHT_ARM_ANGLE_DEG - 90.0)
            )
        else:
            elbow_score = 0.1

        # -- Secondary: compactness (already in [0, 1]) ------------------
        compact_score = compactness

        confidence = (
            _W_PRIMARY * primary
            + _W_TRUNK * trunk_score
            + _W_ELBOW * elbow_score
            + _W_COMPACT * compact_score
        )
        return max(0.0, min(1.0, confidence))

    # -- Side detection (forehand vs backhand) --------------------------------

    @staticmethod
    def _determine_side(
        frames: list[dict], handedness: str, wrist_idx: int
    ) -> str:
        """Determine whether the slice is on the forehand or backhand side.

        For a right-hander, a backhand slice has the dominant wrist on the
        LEFT side of the body midline; a forehand slice has it on the RIGHT.
        The opposite applies for left-handers.

        Backhand slice is the far more common variant in tennis.
        """
        cross_body_count = 0
        same_side_count = 0

        for frame in frames:
            wrist = lm_xy(frame, wrist_idx)
            ls = lm_xy(frame, LM_LEFT_SHOULDER)
            rs = lm_xy(frame, LM_RIGHT_SHOULDER)
            if not (wrist and ls and rs):
                continue

            body_center_x = (ls[0] + rs[0]) / 2.0

            if handedness == "right":
                # Right-hander: wrist left of center = backhand side
                if wrist[0] < body_center_x:
                    cross_body_count += 1
                else:
                    same_side_count += 1
            else:
                # Left-hander: wrist right of center = backhand side
                if wrist[0] > body_center_x:
                    cross_body_count += 1
                else:
                    same_side_count += 1

        if cross_body_count > same_side_count:
            return "backhand"
        return "forehand"

    # -- Phase detection -----------------------------------------------------

    @staticmethod
    def _detect_phases(
        wrist_ys: list[float], frames: list[dict]
    ) -> dict | None:
        """Identify key phase timestamps in the slice motion.

        Phases:
        - preparation: start of sequence (initial positioning)
        - takeback_peak: highest wrist position (minimum y in image coords)
        - contact: estimated ball-strike frame (peak downward wrist speed)
        - follow_through: end of sequence

        Returns None if detection fails.
        """
        if not wrist_ys or not frames:
            return None

        # Takeback peak: minimum y in the first 75% of the wrist trajectory
        search_end = max(2, int(len(wrist_ys) * 0.75))
        min_y = wrist_ys[0]
        min_y_idx = 0
        for i in range(1, search_end):
            if wrist_ys[i] < min_y:
                min_y = wrist_ys[i]
                min_y_idx = i

        # Contact: the frame with the largest single-frame downward wrist
        # movement (largest positive y-change) after the takeback peak.
        contact_idx = min_y_idx
        max_descent_rate = 0.0
        for i in range(min_y_idx + 1, len(wrist_ys)):
            descent = wrist_ys[i] - wrist_ys[i - 1]
            if descent > max_descent_rate:
                max_descent_rate = descent
                contact_idx = i

        # Map wrist_ys indices to frame indices.  wrist_ys was built by
        # iterating frames and skipping invisible ones, so the indices
        # closely track (but may be offset by skipped frames).  Clamp to
        # valid frame range.
        def _frame_idx(ys_idx: int) -> int:
            return frames[min(ys_idx, len(frames) - 1)].get(
                "frame_index", ys_idx
            )

        return {
            "preparation": {"frame": frames[0].get("frame_index", 0)},
            "takeback_peak": {"frame": _frame_idx(min_y_idx)},
            "contact": {"frame": _frame_idx(contact_idx)},
            "follow_through": {
                "frame": frames[-1].get("frame_index", len(frames) - 1)
            },
        }


# ---------------------------------------------------------------------------
# Module-level geometry helper
# ---------------------------------------------------------------------------

def _angle_at_vertex(
    a: tuple[float, float],
    vertex: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Compute the angle (in degrees) at the vertex point B in triangle ABC.

    Uses the dot-product formula: angle = arccos( (BA . BC) / (|BA| |BC|) ).
    """
    va = (a[0] - vertex[0], a[1] - vertex[1])
    vc = (c[0] - vertex[0], c[1] - vertex[1])

    dot = va[0] * vc[0] + va[1] * vc[1]
    mag_a = math.sqrt(va[0] ** 2 + va[1] ** 2)
    mag_c = math.sqrt(vc[0] ** 2 + vc[1] ** 2)

    if mag_a < 1e-9 or mag_c < 1e-9:
        return 180.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))
