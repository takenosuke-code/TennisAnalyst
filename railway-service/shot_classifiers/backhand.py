"""Backhand shot classifier (one-handed and two-handed).

Detects backhand groundstrokes by analysing wrist trajectory relative to the
body midline, wrist proximity (one-handed vs two-handed), shoulder rotation,
and elbow flexion.  Specifically rejects forehands (opposite lateral swing
direction), slices (high-to-low swing path with open wrist), and serves
(wrist above head height).
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
    velocity,
)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Minimum number of frames needed to analyse a shot.
_MIN_FRAMES = 8

# Maximum duration (ms) for a single clean backhand.
_MAX_DURATION_MS = 4000

# Minimum duration (ms) – anything shorter is likely noise.
_MIN_DURATION_MS = 200

# If the distance between both wrists (normalised to torso height) stays
# below this value for the majority of the swing, the shot is two-handed.
_TWO_HAND_WRIST_PROXIMITY = 0.35

# Fraction of swing frames where the wrists must be close for a two-handed
# classification.
_TWO_HAND_FRAME_RATIO = 0.55

# Minimum lateral displacement of the lead wrist (normalised) to register as
# a full swing.
_MIN_LATERAL_DISPLACEMENT = 0.04

# If the wrist's maximum y-position goes above the nose, it looks more like
# a serve.  This ratio is max_wrist_y / nose_y (in image coords where y
# increases downward, so *lower* y = higher on screen).
_SERVE_WRIST_HEIGHT_RATIO = 0.85

# Vertical drop ratio threshold for slice detection.  If the wrist drops
# more than this fraction of the torso height over the swing, it is likely
# a slice rather than a topspin/flat backhand.
_SLICE_VERTICAL_DROP_RATIO = 0.45

# Elbow angle range that is typical for a backhand at contact (degrees).
_BACKHAND_ELBOW_MIN = 100.0
_BACKHAND_ELBOW_MAX = 180.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BackhandClassifier(BaseShotClassifier):
    """Detect backhand groundstrokes (one-handed and two-handed)."""

    @property
    def shot_type(self) -> str:
        return "backhand"

    # --------------------------------------------------------------------- #
    #  Public API                                                             #
    # --------------------------------------------------------------------- #

    def classify(self, frames: list[dict]) -> ClassificationResult:
        """Classify a frame sequence as a backhand (or not).

        Returns a ``ClassificationResult`` with ``confidence == 0.0`` when the
        frames do not represent a backhand.
        """
        camera_angle = estimate_camera_angle(frames)
        handedness = detect_handedness(frames)

        zero = ClassificationResult(
            shot_type=self.shot_type,
            confidence=0.0,
            is_clean=False,
            camera_angle=camera_angle,
            handedness=handedness,
        )

        if len(frames) < _MIN_FRAMES:
            return zero

        # ---- gather per-frame landmark positions ---- #
        lead_wrist_id, off_wrist_id = self._wrist_ids(handedness)
        lead_elbow_key, off_elbow_key = self._elbow_keys(handedness)

        lead_positions: list[tuple[float, float]] = []
        off_positions: list[tuple[float, float]] = []
        hip_centers: list[tuple[float, float]] = []
        nose_ys: list[float] = []
        shoulder_ys: list[float] = []
        torso_heights: list[float] = []
        lead_elbow_angles: list[float] = []
        shoulder_rotations: list[float] = []

        for frame in frames:
            lw = lm_xy(frame, lead_wrist_id)
            ow = lm_xy(frame, off_wrist_id)
            lh = lm_xy(frame, LM_LEFT_HIP)
            rh = lm_xy(frame, LM_RIGHT_HIP)
            ls = lm_xy(frame, LM_LEFT_SHOULDER)
            rs = lm_xy(frame, LM_RIGHT_SHOULDER)
            nose = lm_xy(frame, LM_NOSE)

            if lw:
                lead_positions.append(lw)
            if ow:
                off_positions.append(ow)
            if lh and rh:
                hip_centers.append(midpoint(lh, rh))
            if nose:
                nose_ys.append(nose[1])
            if ls and rs:
                sh_mid = midpoint(ls, rs)
                shoulder_ys.append(sh_mid[1])
            if ls and rs and lh and rh:
                hp_mid = midpoint(lh, rh)
                torso_heights.append(abs(hp_mid[1] - midpoint(ls, rs)[1]))
                shoulder_rotations.append(abs(ls[0] - rs[0]))

            angles = frame.get("joint_angles", {})
            if lead_elbow_key in angles:
                lead_elbow_angles.append(angles[lead_elbow_key])

        # Need sufficient data to make a call.
        if len(lead_positions) < _MIN_FRAMES or len(hip_centers) < _MIN_FRAMES:
            return zero

        avg_torso = (
            sum(torso_heights) / len(torso_heights) if torso_heights else 0.15
        )
        if avg_torso < 0.01:
            avg_torso = 0.15  # fallback to avoid division by zero

        # ---- 1. Reject serves: lead wrist above nose/shoulder height ---- #
        if self._looks_like_serve(lead_positions, nose_ys, shoulder_ys):
            return zero

        # ---- 2. Lateral displacement direction ---- #
        lateral_ok, lateral_score = self._check_lateral_direction(
            lead_positions, hip_centers, handedness, avg_torso
        )
        if not lateral_ok:
            return zero

        # ---- 3. Reject slices (high-to-low path) ---- #
        if self._looks_like_slice(lead_positions, avg_torso):
            return zero

        # ---- 4. One-handed vs two-handed ---- #
        is_two_handed = self._is_two_handed(
            lead_positions, off_positions, avg_torso
        )
        variant = "two_handed" if is_two_handed else "one_handed"

        # ---- 5. Confidence from multiple signals ---- #
        elbow_score = self._elbow_score(lead_elbow_angles)
        rotation_score = self._shoulder_rotation_score(
            shoulder_rotations, avg_torso
        )
        height_score = self._wrist_height_score(lead_positions, hip_centers)

        confidence = self._aggregate_confidence(
            lateral_score, elbow_score, rotation_score, height_score
        )

        # ---- 6. Phase detection (optional enrichment) ---- #
        phase_timestamps = self._detect_phases(frames, lead_wrist_id)

        is_clean = self.validate_clean(frames)

        return ClassificationResult(
            shot_type=self.shot_type,
            confidence=round(confidence, 4),
            is_clean=is_clean,
            camera_angle=camera_angle,
            handedness=handedness,
            phase_timestamps={**phase_timestamps, "variant": variant},
        )

    def validate_clean(self, frames: list[dict]) -> bool:
        """Check that this looks like a single clean backhand.

        Rejects:
        - Very short or very long sequences.
        - Serve-like motions (wrist above head).
        - Possible multi-shot sequences (multiple direction reversals).
        """
        if len(frames) < _MIN_FRAMES:
            return False

        duration_ms = (
            frames[-1].get("timestamp_ms", 0.0)
            - frames[0].get("timestamp_ms", 0.0)
        )
        if duration_ms < _MIN_DURATION_MS or duration_ms > _MAX_DURATION_MS:
            return False

        # Reject if wrist goes above head (serve indicator).
        handedness = detect_handedness(frames)
        lead_wrist_id, _ = self._wrist_ids(handedness)

        wrist_ys: list[float] = []
        nose_ys: list[float] = []
        for frame in frames:
            w = lm_xy(frame, lead_wrist_id)
            n = lm_xy(frame, LM_NOSE)
            if w:
                wrist_ys.append(w[1])
            if n:
                nose_ys.append(n[1])

        if wrist_ys and nose_ys:
            # y increases downward in image coords; wrist above head means
            # wrist y < nose y.
            min_wrist_y = min(wrist_ys)
            avg_nose_y = sum(nose_ys) / len(nose_ys)
            if avg_nose_y > 0 and min_wrist_y < avg_nose_y * _SERVE_WRIST_HEIGHT_RATIO:
                return False

        # Reject multi-shot sequences: count lateral direction reversals.
        wrist_xs: list[float] = []
        for frame in frames:
            w = lm_xy(frame, lead_wrist_id)
            if w:
                wrist_xs.append(w[0])

        if len(wrist_xs) >= 3:
            reversals = 0
            for i in range(2, len(wrist_xs)):
                d1 = wrist_xs[i - 1] - wrist_xs[i - 2]
                d2 = wrist_xs[i] - wrist_xs[i - 1]
                if d1 * d2 < 0 and abs(d1) > 0.01 and abs(d2) > 0.01:
                    reversals += 1
            # A single groundstroke has at most a few small reversals.
            if reversals > len(wrist_xs) * 0.4:
                return False

        return True

    # --------------------------------------------------------------------- #
    #  Private helpers                                                        #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _wrist_ids(handedness: str) -> tuple[int, int]:
        """Return (lead_wrist_id, off_wrist_id) based on handedness.

        For a right-handed backhand the *left* wrist leads the swing.
        """
        if handedness == "right":
            return LM_LEFT_WRIST, LM_RIGHT_WRIST
        return LM_RIGHT_WRIST, LM_LEFT_WRIST

    @staticmethod
    def _elbow_keys(handedness: str) -> tuple[str, str]:
        """Return (lead_elbow_key, off_elbow_key) for joint_angles dict."""
        if handedness == "right":
            return "left_elbow", "right_elbow"
        return "right_elbow", "left_elbow"

    # -- Serve rejection -------------------------------------------------- #

    @staticmethod
    def _looks_like_serve(
        lead_positions: list[tuple[float, float]],
        nose_ys: list[float],
        shoulder_ys: list[float] | None = None,
    ) -> bool:
        """Return True if the lead wrist rises above head/shoulder height.

        Checks two conditions (either triggers rejection):
        1. Wrist goes above nose height (original check).
        2. Wrist goes significantly above average shoulder height.  This
           catches serves even when the nose landmark is not visible.
        """
        if not lead_positions:
            return False

        min_wrist_y = min(p[1] for p in lead_positions)

        # Check 1: wrist above nose
        if nose_ys:
            avg_nose_y = sum(nose_ys) / len(nose_ys)
            if avg_nose_y > 0 and min_wrist_y < avg_nose_y * _SERVE_WRIST_HEIGHT_RATIO:
                return True

        # Check 2: wrist significantly above shoulder (catches serves when
        # nose is not visible).  In image coords, shoulder y is typically
        # ~0.25; if the wrist goes more than 0.08 above the shoulder
        # (lower y), it looks like a serve.
        if shoulder_ys:
            avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
            if avg_shoulder_y > 0 and min_wrist_y < avg_shoulder_y - 0.08:
                return True

        return False

    # -- Lateral direction ------------------------------------------------ #

    @staticmethod
    def _check_lateral_direction(
        lead_positions: list[tuple[float, float]],
        hip_centers: list[tuple[float, float]],
        handedness: str,
        avg_torso: float,
    ) -> tuple[bool, float]:
        """Verify the lead wrist crosses from non-dominant to dominant side.

        For a right-handed player the wrist should move left-to-right in
        image coordinates (x increases to the right).

        Returns (passes, normalised_score).
        """
        n = min(len(lead_positions), len(hip_centers))
        if n < _MIN_FRAMES:
            return False, 0.0

        # Relative lateral position of the wrist versus the hip centre.
        rel_xs = [lead_positions[i][0] - hip_centers[i][0] for i in range(n)]

        start_x = sum(rel_xs[: max(1, n // 4)]) / max(1, n // 4)
        end_x = sum(rel_xs[-(max(1, n // 4)):]) / max(1, n // 4)
        displacement = end_x - start_x

        # Backhand direction: right-handed => wrist moves left→right (positive
        # displacement in image x).  Left-handed => right→left (negative).
        if handedness == "right":
            correct_direction = displacement > 0
        else:
            correct_direction = displacement < 0

        if not correct_direction:
            return False, 0.0

        normalised = abs(displacement) / avg_torso
        if normalised < _MIN_LATERAL_DISPLACEMENT:
            return False, 0.0

        # Score: ramp from 0 at threshold to 1 at 4x threshold.
        score = min(1.0, normalised / (_MIN_LATERAL_DISPLACEMENT * 4))
        return True, score

    # -- Slice rejection -------------------------------------------------- #

    @staticmethod
    def _looks_like_slice(
        lead_positions: list[tuple[float, float]],
        avg_torso: float,
    ) -> bool:
        """Return True if the swing has a pronounced high-to-low trajectory.

        In image coordinates y increases downward, so a slice (high start,
        low finish) has a *positive* vertical displacement.
        """
        if len(lead_positions) < _MIN_FRAMES:
            return False

        quarter = max(1, len(lead_positions) // 4)
        start_y = sum(p[1] for p in lead_positions[:quarter]) / quarter
        end_y = sum(p[1] for p in lead_positions[-quarter:]) / quarter
        drop = end_y - start_y  # positive = downward

        return drop > avg_torso * _SLICE_VERTICAL_DROP_RATIO

    # -- Two-handed detection --------------------------------------------- #

    @staticmethod
    def _is_two_handed(
        lead_positions: list[tuple[float, float]],
        off_positions: list[tuple[float, float]],
        avg_torso: float,
    ) -> bool:
        """Determine if both hands are on the racket."""
        n = min(len(lead_positions), len(off_positions))
        if n < _MIN_FRAMES:
            # Not enough off-hand data => assume one-handed.
            return False

        close_count = 0
        for i in range(n):
            d = distance(lead_positions[i], off_positions[i])
            if d / avg_torso < _TWO_HAND_WRIST_PROXIMITY:
                close_count += 1

        return close_count / n >= _TWO_HAND_FRAME_RATIO

    # -- Sub-scores ------------------------------------------------------- #

    @staticmethod
    def _elbow_score(elbow_angles: list[float]) -> float:
        """Score based on how well the lead elbow angle matches a backhand."""
        if not elbow_angles:
            return 0.5  # neutral if no data

        avg = sum(elbow_angles) / len(elbow_angles)
        if _BACKHAND_ELBOW_MIN <= avg <= _BACKHAND_ELBOW_MAX:
            # Linearly scale within the ideal range.
            mid = (_BACKHAND_ELBOW_MIN + _BACKHAND_ELBOW_MAX) / 2
            deviation = abs(avg - mid) / ((_BACKHAND_ELBOW_MAX - _BACKHAND_ELBOW_MIN) / 2)
            return 1.0 - 0.3 * deviation
        # Outside ideal range – penalise proportionally.
        if avg < _BACKHAND_ELBOW_MIN:
            return max(0.0, 1.0 - (_BACKHAND_ELBOW_MIN - avg) / 60)
        return max(0.0, 1.0 - (avg - _BACKHAND_ELBOW_MAX) / 60)

    @staticmethod
    def _shoulder_rotation_score(
        rotations: list[float], avg_torso: float
    ) -> float:
        """Score based on shoulder rotation change during the swing.

        A backhand typically shows increasing shoulder width (opening up)
        as the player rotates into the shot.
        """
        if len(rotations) < 4:
            return 0.5

        quarter = max(1, len(rotations) // 4)
        start_rot = sum(rotations[:quarter]) / quarter
        end_rot = sum(rotations[-quarter:]) / quarter
        delta = end_rot - start_rot

        # Positive delta means shoulders opening – consistent with backhand.
        if delta <= 0:
            return 0.2
        normalised = delta / avg_torso
        return min(1.0, 0.4 + normalised * 2)

    @staticmethod
    def _wrist_height_score(
        lead_positions: list[tuple[float, float]],
        hip_centers: list[tuple[float, float]],
    ) -> float:
        """Score based on wrist staying below shoulder/head height.

        A backhand wrist should remain roughly between hip and shoulder
        height for most of the swing.
        """
        n = min(len(lead_positions), len(hip_centers))
        if n == 0:
            return 0.5

        good = 0
        for i in range(n):
            wrist_y = lead_positions[i][1]
            hip_y = hip_centers[i][1]
            # Wrist y should be less than (above) or near the hip y.
            # But not drastically above (which would indicate a serve).
            # In image coords: wrist_y < hip_y means above.
            diff = hip_y - wrist_y
            # Allow wrist to be up to 0.3 above the hip (normalised coords)
            if -0.1 < diff < 0.35:
                good += 1

        return good / n

    @staticmethod
    def _aggregate_confidence(
        lateral: float,
        elbow: float,
        rotation: float,
        height: float,
    ) -> float:
        """Weighted combination of sub-scores into [0, 1] confidence."""
        # Lateral direction is the strongest discriminator.
        weights = {
            "lateral": 0.40,
            "elbow": 0.15,
            "rotation": 0.25,
            "height": 0.20,
        }
        raw = (
            weights["lateral"] * lateral
            + weights["elbow"] * elbow
            + weights["rotation"] * rotation
            + weights["height"] * height
        )
        # Clamp to [0, 1].
        return max(0.0, min(1.0, raw))

    # -- Phase detection -------------------------------------------------- #

    @staticmethod
    def _detect_phases(
        frames: list[dict], lead_wrist_id: int
    ) -> dict[str, float | None]:
        """Estimate preparation, forward-swing, and follow-through phases.

        Returns a dict with ``preparation_end``, ``contact``, and
        ``follow_through_start`` timestamps (ms), or ``None`` for each
        phase that could not be determined.
        """
        result: dict[str, float | None] = {
            "preparation_end": None,
            "contact": None,
            "follow_through_start": None,
        }

        positions: list[tuple[float, float]] = []
        timestamps: list[float] = []
        for frame in frames:
            w = lm_xy(frame, lead_wrist_id)
            if w is not None:
                positions.append(w)
                timestamps.append(frame.get("timestamp_ms", 0.0))

        if len(positions) < _MIN_FRAMES:
            return result

        # Compute inter-frame speeds.
        avg_dt = (timestamps[-1] - timestamps[0]) / max(1, len(timestamps) - 1)
        if avg_dt <= 0:
            return result

        speeds = velocity(positions, avg_dt)

        if not speeds:
            return result

        # Contact is approximately at peak wrist speed.
        peak_idx = speeds.index(max(speeds))
        # Map back to frame timestamps (speeds list is 1 shorter).
        contact_idx = min(peak_idx + 1, len(timestamps) - 1)
        result["contact"] = timestamps[contact_idx]

        # Preparation ends ~30 % before contact.
        prep_idx = max(0, int(contact_idx * 0.3))
        result["preparation_end"] = timestamps[prep_idx]

        # Follow-through starts shortly after contact.
        ft_idx = min(len(timestamps) - 1, contact_idx + max(1, (len(timestamps) - contact_idx) // 3))
        result["follow_through_start"] = timestamps[ft_idx]

        return result
