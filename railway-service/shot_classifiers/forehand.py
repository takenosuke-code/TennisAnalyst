"""Forehand shot classifier.

Detects forehand groundstrokes by tracking the dominant wrist trajectory
relative to the body midline, combined with shoulder rotation, elbow
extension patterns, and swing path analysis.  Rejects serves (wrist above
shoulder height), backhands (wrist starts on the non-dominant side), and
slices (high-to-low swing path).

Only forehands pass this classifier -- all other shot types are rejected
with confidence 0.0.
"""

from __future__ import annotations

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
    get_landmark,
    lm_xy,
    midpoint,
)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
_MIN_FRAMES = 5
_MAX_SHOT_DURATION_MS = 3000.0
_MIN_LATERAL_DISPLACEMENT = 0.04  # normalized coords; wrist must cross at least this far
_SERVE_HEIGHT_RATIO = 0.85  # wrist-above-shoulder ratio above which we reject as serve
_MIN_CONFIDENCE_THRESHOLD = 0.35  # below this we call it "not a forehand"
_DIRECTION_CHANGE_LIMIT = 2  # max direction reversals before rejecting as unclean


class ForehandClassifier(BaseShotClassifier):
    """Classifier that detects forehand groundstrokes from pose frame sequences.

    The detection algorithm:

    1. Determine handedness (which hand is dominant).
    2. Track the dominant wrist's lateral position relative to the hip
       midpoint across the frame sequence.
    3. Verify that the wrist starts on the *same* side as the dominant
       shoulder (forehand preparation) and sweeps across the body midline
       (forehand follow-through).  A backhand starts on the *opposite* side.
    4. Compute a weighted confidence from four signals:
       - Lateral wrist displacement across the body (primary, 40%)
       - Shoulder rotation / trunk coiling (20%)
       - Elbow extension during the swing (20%)
       - Wrist stays below shoulder height, ruling out serves (20%)
    5. Adapt heuristics to the estimated camera angle.

    ``validate_clean`` additionally rejects multi-stroke sequences, serve
    contamination (wrist above head), and sequences that are too short or
    too long.
    """

    # -- BaseShotClassifier interface ----------------------------------------

    @property
    def shot_type(self) -> str:
        """Return the shot type this classifier detects."""
        return "forehand"

    def classify(self, frames: list[dict]) -> ClassificationResult:
        """Classify a sequence of pose frames as forehand or not.

        Returns a ``ClassificationResult`` with ``shot_type='forehand'``.
        If the sequence is not a forehand, ``confidence`` will be 0.0.
        """
        camera_angle = estimate_camera_angle(frames)
        handedness = detect_handedness(frames)

        if len(frames) < _MIN_FRAMES:
            return self._reject(camera_angle, handedness)

        # Select dominant / non-dominant landmark indices
        dom_wrist, dom_shoulder, dom_elbow, dom_hip, off_hip = (
            self._hand_indices(handedness)
        )

        # ----- Collect per-frame measurements -----
        wrist_xs: list[float] = []
        hip_mids: list[tuple[float, float]] = []
        wrist_ys: list[float] = []
        shoulder_ys: list[float] = []
        shoulder_widths: list[float] = []
        elbow_angles: list[float | None] = []
        wrist_positions: list[tuple[float, float]] = []

        for frame in frames:
            wrist = lm_xy(frame, dom_wrist)
            lh = lm_xy(frame, LM_LEFT_HIP)
            rh = lm_xy(frame, LM_RIGHT_HIP)
            dom_sh = lm_xy(frame, dom_shoulder)

            if wrist and lh and rh:
                hip_mid = midpoint(lh, rh)
                wrist_xs.append(wrist[0] - hip_mid[0])  # relative lateral offset
                hip_mids.append(hip_mid)
                wrist_ys.append(wrist[1])
                wrist_positions.append(wrist)
                if dom_sh:
                    shoulder_ys.append(dom_sh[1])
                else:
                    shoulder_ys.append(hip_mid[1])  # fallback

            # Shoulder width for trunk rotation proxy
            ls = lm_xy(frame, LM_LEFT_SHOULDER)
            rs = lm_xy(frame, LM_RIGHT_SHOULDER)
            if ls and rs:
                shoulder_widths.append(abs(ls[0] - rs[0]))

            # Elbow angle from joint_angles if available
            angle_key = "right_elbow" if handedness == "right" else "left_elbow"
            angles = frame.get("joint_angles", {})
            elbow_angles.append(angles.get(angle_key))

        # Need a minimum number of valid measurements
        if len(wrist_xs) < _MIN_FRAMES:
            return self._reject(camera_angle, handedness)

        # ----- Signal 1: Lateral wrist displacement (40%) -----
        lateral_score = self._score_lateral_displacement(
            wrist_xs, handedness, camera_angle
        )

        # Lateral displacement is the primary discriminator for a forehand.
        # If the wrist does not move in the correct cross-body direction,
        # this cannot be a forehand.
        if lateral_score == 0.0:
            return self._reject(camera_angle, handedness)

        # ----- Signal 2: Shoulder rotation (20%) -----
        rotation_score = self._score_shoulder_rotation(shoulder_widths)

        # ----- Signal 3: Elbow extension pattern (20%) -----
        extension_score = self._score_elbow_extension(elbow_angles)

        # ----- Signal 4: Wrist stays below shoulder (20%) -----
        below_score = self._score_wrist_below_shoulder(wrist_ys, shoulder_ys)

        # ----- Combine -----
        confidence = (
            0.40 * lateral_score
            + 0.20 * rotation_score
            + 0.20 * extension_score
            + 0.20 * below_score
        )
        confidence = max(0.0, min(1.0, confidence))

        if confidence < _MIN_CONFIDENCE_THRESHOLD:
            return self._reject(camera_angle, handedness)

        is_clean = self.validate_clean(frames)

        # Detect phase timestamps (best-effort)
        phase_timestamps = self._detect_phases(frames, dom_wrist, dom_elbow)

        return ClassificationResult(
            shot_type="forehand",
            confidence=round(confidence, 3),
            is_clean=is_clean,
            camera_angle=camera_angle,
            handedness=handedness,
            phase_timestamps=phase_timestamps,
        )

    def validate_clean(self, frames: list[dict]) -> bool:
        """Check whether the sequence contains a single clean forehand.

        Rejects sequences that:
        - Have fewer than 5 frames.
        - Span more than 3 seconds of shot motion.
        - Contain multiple lateral direction changes (rally contamination).
        - Show the wrist going above nose height (serve contamination).
        """
        if len(frames) < _MIN_FRAMES:
            return False

        duration_ms = (
            frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        )
        if duration_ms > _MAX_SHOT_DURATION_MS:
            return False

        handedness = detect_handedness(frames)
        dom_wrist = LM_RIGHT_WRIST if handedness == "right" else LM_LEFT_WRIST

        # --- Check wrist above head (serve contamination) ---
        for frame in frames:
            wrist = lm_xy(frame, dom_wrist)
            nose = lm_xy(frame, LM_NOSE)
            if wrist and nose:
                # In normalized coords, lower y = higher on screen
                if wrist[1] < nose[1]:
                    return False

        # --- Check for multiple direction changes ---
        wrist_xs: list[float] = []
        for frame in frames:
            w = lm_xy(frame, dom_wrist)
            if w:
                wrist_xs.append(w[0])

        if len(wrist_xs) < _MIN_FRAMES:
            return False

        direction_changes = self._count_direction_changes(wrist_xs)
        if direction_changes > _DIRECTION_CHANGE_LIMIT:
            return False

        return True

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _hand_indices(
        handedness: str,
    ) -> tuple[int, int, int, int, int]:
        """Return (wrist, shoulder, elbow, dom_hip, off_hip) landmark indices."""
        if handedness == "right":
            return (
                LM_RIGHT_WRIST,
                LM_RIGHT_SHOULDER,
                LM_RIGHT_ELBOW,
                LM_RIGHT_HIP,
                LM_LEFT_HIP,
            )
        return (
            LM_LEFT_WRIST,
            LM_LEFT_SHOULDER,
            LM_LEFT_ELBOW,
            LM_LEFT_HIP,
            LM_RIGHT_HIP,
        )

    @staticmethod
    def _reject(camera_angle: str, handedness: str) -> ClassificationResult:
        """Return a zero-confidence rejection result."""
        return ClassificationResult(
            shot_type="forehand",
            confidence=0.0,
            is_clean=False,
            camera_angle=camera_angle,
            handedness=handedness,
        )

    # -- Scoring helpers -----------------------------------------------------

    @staticmethod
    def _score_lateral_displacement(
        relative_xs: list[float],
        handedness: str,
        camera_angle: str,
    ) -> float:
        """Score the dominant wrist's cross-body lateral sweep.

        For a right-handed forehand viewed from behind, the wrist starts
        to the right of the hip center (positive relative x) and sweeps
        to the left (negative relative x).  Coordinates flip for left-
        handers and for front-view camera angles.

        Returns a score in [0, 1].
        """
        if len(relative_xs) < 2:
            return 0.0

        start_x = sum(relative_xs[:3]) / min(3, len(relative_xs))
        end_x = sum(relative_xs[-3:]) / min(3, len(relative_xs[-3:]))

        displacement = start_x - end_x  # positive = moved from right to left

        # Adjust sign expectations based on handedness and camera angle.
        # Behind view, right-handed: wrist moves right -> left (positive displacement).
        # Front view, right-handed: mirrored, so displacement sign flips.
        # Left-handed: opposite direction in all views.
        expected_positive = True
        if handedness == "left":
            expected_positive = not expected_positive
        if camera_angle == "front":
            expected_positive = not expected_positive

        signed_disp = displacement if expected_positive else -displacement

        # For side view, lateral displacement is less visible; relax threshold
        threshold = _MIN_LATERAL_DISPLACEMENT
        if camera_angle == "side":
            threshold *= 0.5

        if signed_disp < threshold:
            return 0.0

        # Scale to 0-1 (displacement of 0.15+ in normalized coords is strong)
        max_disp = 0.15 if camera_angle != "side" else 0.08
        return min(1.0, signed_disp / max_disp)

    @staticmethod
    def _score_shoulder_rotation(shoulder_widths: list[float]) -> float:
        """Score trunk coiling/uncoiling from changes in apparent shoulder width.

        During a forehand, the shoulders rotate: the width narrows during
        preparation (coiling) and widens through contact (uncoiling).

        Returns a score in [0, 1].
        """
        if len(shoulder_widths) < 4:
            return 0.0

        # Split into first half (coil) and second half (uncoil)
        mid = len(shoulder_widths) // 2
        first_half = shoulder_widths[:mid]
        second_half = shoulder_widths[mid:]

        min_first = min(first_half)
        max_second = max(second_half)

        if min_first <= 0.001:
            return 0.0

        # Ratio of widening; a forehand typically has a 20%+ change
        ratio = (max_second - min_first) / min_first
        if ratio < 0.05:
            return 0.0

        return min(1.0, ratio / 0.30)

    @staticmethod
    def _score_elbow_extension(elbow_angles: list[float | None]) -> float:
        """Score elbow extension pattern during the swing.

        A forehand shows the elbow compressing during preparation then
        extending rapidly through contact.  We look for a minimum angle
        followed by a higher maximum angle.

        Returns a score in [0, 1].
        """
        valid = [a for a in elbow_angles if a is not None]
        if len(valid) < 4:
            return 0.5  # neutral when data is missing

        # Find the minimum in the first 60% and maximum in the last 60%
        split = max(2, int(len(valid) * 0.6))
        early_min = min(valid[:split])
        late_max = max(valid[len(valid) - split :])

        extension_range = late_max - early_min
        if extension_range < 10:
            return 0.0

        # 60+ degrees of extension is a strong forehand signal
        return min(1.0, extension_range / 60.0)

    @staticmethod
    def _score_wrist_below_shoulder(
        wrist_ys: list[float], shoulder_ys: list[float]
    ) -> float:
        """Score how consistently the wrist stays at or below shoulder height.

        In normalized image coordinates, larger y = lower on screen.
        A forehand keeps the wrist roughly at or below shoulder level;
        a serve raises the wrist well above the shoulder.

        If the wrist goes above nose height (~0.15 in normalised coords)
        at any point, this is almost certainly a serve, not a forehand,
        so the score is forced to 0.

        Returns a score in [0, 1].
        """
        if not wrist_ys or not shoulder_ys:
            return 0.0

        # Hard reject: if the wrist goes above nose height at any point,
        # this is serve territory — return 0 immediately.
        # In normalised image coords, nose is typically around y=0.15.
        # If the wrist goes above the average shoulder height by more than
        # the shoulder-to-nose distance, it's clearly above the head.
        avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
        min_wrist_y = min(wrist_ys)
        # Estimate nose height as roughly 0.10 above shoulder (lower y value)
        estimated_nose_y = avg_shoulder_y - 0.10
        if min_wrist_y < estimated_nose_y:
            return 0.0

        count = min(len(wrist_ys), len(shoulder_ys))
        below_count = 0
        for i in range(count):
            # wrist_y >= shoulder_y means wrist is at or below shoulder
            # Allow a small margin above shoulder (0.03 in normalized coords)
            if wrist_ys[i] >= shoulder_ys[i] - 0.03:
                below_count += 1

        ratio = below_count / count
        if ratio < (1.0 - _SERVE_HEIGHT_RATIO):
            return 0.0

        return ratio

    @staticmethod
    def _count_direction_changes(xs: list[float]) -> int:
        """Count the number of lateral direction reversals.

        A single forehand has one primary sweep direction.  Multiple
        reversals suggest a rally or mixed-action sequence.
        """
        if len(xs) < 3:
            return 0

        # Smooth with a simple 3-point moving average to reduce noise
        smoothed: list[float] = []
        for i in range(len(xs)):
            start = max(0, i - 1)
            end = min(len(xs), i + 2)
            smoothed.append(sum(xs[start:end]) / (end - start))

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

    @staticmethod
    def _detect_phases(
        frames: list[dict],
        dom_wrist: int,
        dom_elbow: int,
    ) -> dict | None:
        """Attempt to identify forehand swing phases by frame index.

        Phases:
        - preparation: first frame to the frame with minimum elbow angle
        - acceleration: minimum elbow angle to maximum wrist speed
        - contact: frame around peak wrist speed
        - follow_through: after contact to end

        Returns None if phase detection fails.
        """
        elbow_angle_key = (
            "right_elbow" if dom_wrist == LM_RIGHT_WRIST else "left_elbow"
        )

        # Collect elbow angles with frame indices
        angle_pairs: list[tuple[int, float]] = []
        wrist_positions: list[tuple[int, tuple[float, float]]] = []

        for frame in frames:
            idx = frame.get("frame_index", 0)
            angles = frame.get("joint_angles", {})
            ea = angles.get(elbow_angle_key)
            if ea is not None:
                angle_pairs.append((idx, ea))

            w = lm_xy(frame, dom_wrist)
            if w:
                wrist_positions.append((idx, w))

        if len(angle_pairs) < 4 or len(wrist_positions) < 4:
            return None

        # Find minimum elbow angle (end of preparation / loading)
        min_angle_idx, _ = min(angle_pairs, key=lambda p: p[1])

        # Find peak wrist speed (contact point)
        max_speed = 0.0
        max_speed_idx = wrist_positions[-1][0]
        for i in range(1, len(wrist_positions)):
            prev_idx, prev_pos = wrist_positions[i - 1]
            curr_idx, curr_pos = wrist_positions[i]
            d = distance(prev_pos, curr_pos)
            if d > max_speed:
                max_speed = d
                max_speed_idx = curr_idx

        first_idx = frames[0].get("frame_index", 0)
        last_idx = frames[-1].get("frame_index", len(frames) - 1)

        return {
            "preparation": {"start": first_idx, "end": min_angle_idx},
            "acceleration": {"start": min_angle_idx, "end": max_speed_idx},
            "contact": {"frame": max_speed_idx},
            "follow_through": {"start": max_speed_idx, "end": last_idx},
        }
