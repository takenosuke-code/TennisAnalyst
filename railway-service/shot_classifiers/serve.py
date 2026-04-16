"""Serve shot classifier.

Detects flat, kick, and slice serves by tracking the distinctive upward
wrist trajectory, trophy position, knee bend, and body extension that
differentiate a serve from all other tennis strokes.

The serve is the most biomechanically distinctive shot in tennis: the
dominant wrist travels from below the shoulder to well above the head,
pausing briefly in a "trophy position" before explosive forward swing
and contact at full arm extension overhead.
"""

from __future__ import annotations

import math

from .base import (
    BaseShotClassifier,
    ClassificationResult,
    LM_LEFT_ANKLE,
    LM_LEFT_ELBOW,
    LM_LEFT_HIP,
    LM_LEFT_KNEE,
    LM_LEFT_SHOULDER,
    LM_LEFT_WRIST,
    LM_RIGHT_ANKLE,
    LM_RIGHT_ELBOW,
    LM_RIGHT_HIP,
    LM_RIGHT_KNEE,
    LM_RIGHT_SHOULDER,
    LM_RIGHT_WRIST,
    detect_handedness,
    distance,
    estimate_camera_angle,
    lm_xy,
)

# ---------------------------------------------------------------------------
# Thresholds (normalized coordinate space, y=0 top, y=1 bottom)
# ---------------------------------------------------------------------------

# How far above shoulder (in normalized y) the wrist must rise to count
# as a serve.  A positive delta here means wrist y < shoulder y.
_MIN_WRIST_ABOVE_SHOULDER: float = 0.06

# Minimum knee bend depth (angle decrease in degrees from standing)
_MIN_KNEE_BEND_DEGREES: float = 10.0

# Duration window for a single serve sequence (ms).  Slow-motion clips
# stretch real time, so we accept a wide range.
_MIN_DURATION_MS: float = 300.0
_MAX_DURATION_MS: float = 12_000.0

# Minimum number of frames for a usable sequence
_MIN_FRAMES: int = 8

# Peak-count threshold for multi-serve rejection
_MULTI_PEAK_REASCENT: float = 0.04


class ServeClassifier(BaseShotClassifier):
    """Classifies tennis serve sequences from pose landmark data.

    Detection strategy:
    1. Resolve handedness to pick dominant/non-dominant landmarks.
    2. Build per-frame wrist-height and knee-angle trajectories.
    3. Find peak wrist height; verify it exceeds shoulder height by a margin.
    4. Locate the trophy position (peak wrist + ~90-degree elbow).
    5. Verify knee bend (flexion then extension).
    6. Score confidence from multiple biomechanical signals.
    """

    @property
    def shot_type(self) -> str:
        return "serve"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, frames: list[dict]) -> ClassificationResult:
        """Classify *frames* as a serve or not.

        Returns a ClassificationResult with confidence > 0 only when the
        sequence exhibits clear serve biomechanics.
        """
        camera_angle = estimate_camera_angle(frames)
        handedness = detect_handedness(frames)

        # Quick rejection: not enough data
        if len(frames) < _MIN_FRAMES:
            return self._no_match(camera_angle, handedness)

        duration_ms = (
            frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        )
        if duration_ms < _MIN_DURATION_MS or duration_ms > _MAX_DURATION_MS:
            return self._no_match(camera_angle, handedness)

        # Resolve landmark indices based on handedness
        dom_wrist, dom_elbow, dom_shoulder = self._dominant_arm(handedness)
        nondom_wrist, _, _ = self._nondominant_arm(handedness)
        dom_hip, dom_knee, dom_ankle = self._dominant_leg(handedness)

        # Build trajectories
        wrist_ys = self._extract_y_trajectory(frames, dom_wrist)
        shoulder_ys = self._extract_y_trajectory(frames, dom_shoulder)
        knee_angles = self._extract_knee_angles(frames, dom_hip, dom_knee, dom_ankle)
        nondom_wrist_ys = self._extract_y_trajectory(frames, nondom_wrist)

        # Must have enough valid data points
        if len(wrist_ys) < _MIN_FRAMES or len(shoulder_ys) < _MIN_FRAMES:
            return self._no_match(camera_angle, handedness)

        # ----- Signal 1: wrist peak above shoulder -----
        peak_idx, peak_wrist_y = self._find_peak_wrist(wrist_ys)
        if peak_idx is None:
            return self._no_match(camera_angle, handedness)

        # Interpolate shoulder y at peak frame
        shoulder_y_at_peak = self._value_at_index(shoulder_ys, peak_idx)
        if shoulder_y_at_peak is None:
            return self._no_match(camera_angle, handedness)

        wrist_above_shoulder = shoulder_y_at_peak - peak_wrist_y
        if wrist_above_shoulder < _MIN_WRIST_ABOVE_SHOULDER:
            return self._no_match(camera_angle, handedness)

        # ----- Signal 2: trophy position (elbow ~90 degrees) -----
        trophy_frame_idx, _ = self._find_trophy_position(
            frames, dom_wrist, dom_elbow, dom_shoulder, peak_idx
        )
        has_trophy = trophy_frame_idx is not None

        # ----- Signal 3: knee bend -----
        knee_bend_depth = self._compute_knee_bend_depth(knee_angles)
        has_knee_bend = knee_bend_depth >= _MIN_KNEE_BEND_DEGREES

        # ----- Signal 4: body extension (hip-shoulder distance increase) -----
        extension_ratio = self._compute_body_extension(
            frames, dom_shoulder, dom_hip, peak_idx
        )

        # ----- Signal 5: non-dominant arm rise (ball toss proxy) -----
        toss_signal = self._detect_toss_arm(nondom_wrist_ys, peak_idx)

        # ----- Confidence scoring -----
        confidence = self._compute_confidence(
            wrist_above_shoulder=wrist_above_shoulder,
            has_trophy=has_trophy,
            has_knee_bend=has_knee_bend,
            knee_bend_depth=knee_bend_depth,
            extension_ratio=extension_ratio,
            toss_signal=toss_signal,
            duration_ms=duration_ms,
            camera_angle=camera_angle,
        )

        if confidence < 0.15:
            return self._no_match(camera_angle, handedness)

        # ----- Phase timestamps -----
        phase_timestamps = self._compute_phases(
            frames, wrist_ys, peak_idx, trophy_frame_idx, knee_angles
        )

        is_clean = self.validate_clean(frames)

        return ClassificationResult(
            shot_type=self.shot_type,
            confidence=round(min(confidence, 1.0), 3),
            is_clean=is_clean,
            camera_angle=camera_angle,
            handedness=handedness,
            phase_timestamps=phase_timestamps,
        )

    def validate_clean(self, frames: list[dict]) -> bool:
        """Validate that the sequence is a single clean serve.

        Rejects:
        - Sequences where wrist never reaches above shoulder (not a serve)
        - Multi-serve sequences (multiple distinct peaks)
        - Sequences with no clear knee bend
        """
        if len(frames) < _MIN_FRAMES:
            return False

        duration_ms = (
            frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        )
        if duration_ms > _MAX_DURATION_MS:
            return False

        handedness = detect_handedness(frames)
        dom_wrist, _, dom_shoulder = self._dominant_arm(handedness)
        dom_hip, dom_knee, dom_ankle = self._dominant_leg(handedness)

        # Check 1: wrist above shoulder
        wrist_ys = self._extract_y_trajectory(frames, dom_wrist)
        shoulder_ys = self._extract_y_trajectory(frames, dom_shoulder)

        if len(wrist_ys) < _MIN_FRAMES or len(shoulder_ys) < _MIN_FRAMES:
            return False

        peak_idx, peak_wrist_y = self._find_peak_wrist(wrist_ys)
        if peak_idx is None:
            return False

        shoulder_y_at_peak = self._value_at_index(shoulder_ys, peak_idx)
        if shoulder_y_at_peak is None:
            return False

        if (shoulder_y_at_peak - peak_wrist_y) < _MIN_WRIST_ABOVE_SHOULDER:
            return False

        # Check 2: no multiple peaks (multi-serve rejection)
        if self._has_multiple_peaks(wrist_ys):
            return False

        # Check 3: knee bend present
        knee_angles = self._extract_knee_angles(frames, dom_hip, dom_knee, dom_ankle)
        if self._compute_knee_bend_depth(knee_angles) < _MIN_KNEE_BEND_DEGREES:
            return False

        return True

    # ------------------------------------------------------------------
    # Landmark resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_arm(handedness: str) -> tuple[int, int, int]:
        """Return (wrist, elbow, shoulder) landmark indices for the dominant arm."""
        if handedness == "left":
            return LM_LEFT_WRIST, LM_LEFT_ELBOW, LM_LEFT_SHOULDER
        return LM_RIGHT_WRIST, LM_RIGHT_ELBOW, LM_RIGHT_SHOULDER

    @staticmethod
    def _nondominant_arm(handedness: str) -> tuple[int, int, int]:
        """Return (wrist, elbow, shoulder) landmark indices for the non-dominant arm."""
        if handedness == "left":
            return LM_RIGHT_WRIST, LM_RIGHT_ELBOW, LM_RIGHT_SHOULDER
        return LM_LEFT_WRIST, LM_LEFT_ELBOW, LM_LEFT_SHOULDER

    @staticmethod
    def _dominant_leg(handedness: str) -> tuple[int, int, int]:
        """Return (hip, knee, ankle) landmark indices for the dominant side."""
        if handedness == "left":
            return LM_LEFT_HIP, LM_LEFT_KNEE, LM_LEFT_ANKLE
        return LM_RIGHT_HIP, LM_RIGHT_KNEE, LM_RIGHT_ANKLE

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_y_trajectory(
        frames: list[dict], landmark_idx: int
    ) -> list[tuple[int, float]]:
        """Return list of (frame_list_index, y_value) for visible frames.

        Uses the frame's position in the list (not frame_index from the dict)
        so that indices are contiguous and usable for array-style lookups.
        """
        trajectory: list[tuple[int, float]] = []
        for i, frame in enumerate(frames):
            pos = lm_xy(frame, landmark_idx)
            if pos is not None:
                trajectory.append((i, pos[1]))
        return trajectory

    @staticmethod
    def _extract_knee_angles(
        frames: list[dict],
        hip_idx: int,
        knee_idx: int,
        ankle_idx: int,
    ) -> list[tuple[int, float]]:
        """Compute knee angle per frame from hip-knee-ankle landmarks.

        The angle is measured at the knee joint: 180 = fully extended,
        smaller values = more flexion.
        """
        angles: list[tuple[int, float]] = []
        for i, frame in enumerate(frames):
            hip = lm_xy(frame, hip_idx)
            knee = lm_xy(frame, knee_idx)
            ankle = lm_xy(frame, ankle_idx)
            if hip is None or knee is None or ankle is None:
                continue
            # Vectors from knee to hip and knee to ankle
            v1 = (hip[0] - knee[0], hip[1] - knee[1])
            v2 = (ankle[0] - knee[0], ankle[1] - knee[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if mag1 < 1e-6 or mag2 < 1e-6:
                continue
            cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle_deg = math.degrees(math.acos(cos_angle))
            angles.append((i, angle_deg))
        return angles

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peak_wrist(
        wrist_ys: list[tuple[int, float]],
    ) -> tuple[int | None, float]:
        """Find the frame index where wrist is highest (lowest y value).

        Returns (frame_list_index, y_value) or (None, 0.0) if no data.
        """
        if not wrist_ys:
            return None, 0.0
        best_idx, best_y = min(wrist_ys, key=lambda t: t[1])
        return best_idx, best_y

    @staticmethod
    def _value_at_index(
        trajectory: list[tuple[int, float]], target_idx: int
    ) -> float | None:
        """Get the trajectory value at or nearest to *target_idx*."""
        if not trajectory:
            return None
        # Exact match first
        for idx, val in trajectory:
            if idx == target_idx:
                return val
        # Nearest neighbor
        nearest = min(trajectory, key=lambda t: abs(t[0] - target_idx))
        return nearest[1]

    @staticmethod
    def _find_trophy_position(
        frames: list[dict],
        wrist_idx: int,
        elbow_idx: int,
        shoulder_idx: int,
        peak_frame_idx: int,
    ) -> tuple[int | None, float]:
        """Locate the trophy position near the peak wrist frame.

        The trophy position is characterized by the elbow at roughly 90
        degrees and the wrist near its highest point.  We search a window
        around the peak frame — biased earlier since trophy precedes contact.

        Returns (frame_list_index, elbow_angle_degrees) or (None, 0.0).
        """
        search_start = max(0, peak_frame_idx - len(frames) // 3)
        search_end = min(len(frames), peak_frame_idx + len(frames) // 6)

        best_trophy_idx: int | None = None
        best_trophy_score: float = float("inf")
        best_elbow_angle: float = 0.0

        for i in range(search_start, search_end):
            frame = frames[i]
            shoulder = lm_xy(frame, shoulder_idx)
            elbow = lm_xy(frame, elbow_idx)
            wrist = lm_xy(frame, wrist_idx)
            if shoulder is None or elbow is None or wrist is None:
                continue

            # Elbow angle: shoulder-elbow-wrist
            v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
            v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if mag1 < 1e-6 or mag2 < 1e-6:
                continue
            cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            elbow_angle = math.degrees(math.acos(cos_a))

            # Score: deviation from 90 degrees, weighted by distance from peak
            angle_dev = abs(elbow_angle - 90.0)
            frame_dist = abs(i - peak_frame_idx)
            score = angle_dev + frame_dist * 0.5

            if score < best_trophy_score:
                best_trophy_score = score
                best_trophy_idx = i
                best_elbow_angle = elbow_angle

        # Accept trophy if elbow is within 50 degrees of 90
        if best_trophy_idx is not None and abs(best_elbow_angle - 90.0) <= 50.0:
            return best_trophy_idx, best_elbow_angle
        return None, 0.0

    @staticmethod
    def _compute_knee_bend_depth(
        knee_angles: list[tuple[int, float]],
    ) -> float:
        """Return the depth of knee bend in degrees.

        Measures the drop from the maximum (standing) angle to the minimum
        (deepest bend) that occurs before the subsequent extension.
        """
        if len(knee_angles) < 3:
            return 0.0

        angles_only = [a for _, a in knee_angles]
        max_angle = max(angles_only)
        min_angle = min(angles_only)
        return max_angle - min_angle

    @staticmethod
    def _compute_body_extension(
        frames: list[dict],
        shoulder_idx: int,
        hip_idx: int,
        peak_frame_idx: int,
    ) -> float:
        """Compute body extension ratio: torso length at peak vs early frames.

        A ratio > 1.0 indicates the body stretched upward at contact.
        """
        early_distances: list[float] = []
        peak_distances: list[float] = []

        early_end = max(1, len(frames) // 4)
        peak_start = max(0, peak_frame_idx - len(frames) // 8)
        peak_end = min(len(frames), peak_frame_idx + len(frames) // 8 + 1)

        for i in range(0, early_end):
            s = lm_xy(frames[i], shoulder_idx)
            h = lm_xy(frames[i], hip_idx)
            if s and h:
                early_distances.append(distance(s, h))

        for i in range(peak_start, peak_end):
            s = lm_xy(frames[i], shoulder_idx)
            h = lm_xy(frames[i], hip_idx)
            if s and h:
                peak_distances.append(distance(s, h))

        if not early_distances or not peak_distances:
            return 1.0

        avg_early = sum(early_distances) / len(early_distances)
        avg_peak = sum(peak_distances) / len(peak_distances)

        if avg_early < 1e-6:
            return 1.0
        return avg_peak / avg_early

    @staticmethod
    def _detect_toss_arm(
        nondom_wrist_ys: list[tuple[int, float]],
        peak_idx: int,
    ) -> float:
        """Detect upward motion of the non-dominant arm (ball toss).

        Returns a score 0-1 indicating how strong the toss signal is.
        The toss arm should rise (y decreases) before the contact point.
        """
        if len(nondom_wrist_ys) < 4:
            return 0.0

        # Look for the non-dominant wrist rising (y decreasing) before the peak
        pre_peak = [(i, y) for i, y in nondom_wrist_ys if i <= peak_idx]
        if len(pre_peak) < 2:
            return 0.0

        # Find the range of y motion before peak
        ys_pre = [y for _, y in pre_peak]
        y_range = max(ys_pre) - min(ys_pre)

        # The minimum y (highest point) should occur in the latter portion of
        # pre-peak frames — the toss arm rises toward the contact.
        min_y_idx_in_list = ys_pre.index(min(ys_pre))
        rises_late = min_y_idx_in_list > len(ys_pre) // 3

        if y_range > 0.15 and rises_late:
            return min(1.0, y_range / 0.3)
        if y_range > 0.08 and rises_late:
            return y_range / 0.15
        return 0.0

    @staticmethod
    def _has_multiple_peaks(
        wrist_ys: list[tuple[int, float]],
    ) -> bool:
        """Detect multiple distinct wrist peaks indicating multiple serves.

        Looks for the wrist descending significantly after the global peak
        and then re-ascending to a similar height, which would indicate a
        second serve motion in the same clip.
        """
        if len(wrist_ys) < 10:
            return False

        ys = [y for _, y in wrist_ys]
        peak_y = min(ys)
        peak_pos = ys.index(peak_y)

        # After the peak, look for a significant descent then re-ascent
        post_peak = ys[peak_pos:]
        if len(post_peak) < 4:
            return False

        max_after_peak = max(post_peak)
        descent = max_after_peak - peak_y

        # If the wrist does not drop significantly, no second peak possible
        if descent < _MULTI_PEAK_REASCENT:
            return False

        # Find where the descent reaches its lowest point (highest y)
        valley_y = max_after_peak
        valley_pos = post_peak.index(valley_y)

        # Check for re-ascent after valley
        if valley_pos >= len(post_peak) - 2:
            return False

        post_valley = post_peak[valley_pos:]
        reascent = valley_y - min(post_valley)

        # If re-ascent is significant, this is likely a multi-serve clip
        return reascent > _MULTI_PEAK_REASCENT * 2

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        *,
        wrist_above_shoulder: float,
        has_trophy: bool,
        has_knee_bend: bool,
        knee_bend_depth: float,
        extension_ratio: float,
        toss_signal: float,
        duration_ms: float,
        camera_angle: str,
    ) -> float:
        """Combine biomechanical signals into a 0-1 confidence score.

        The wrist-above-shoulder signal is the primary gate.  Other signals
        (trophy position, knee bend, body extension, toss arm, duration)
        add or subtract from the base confidence to refine the score.
        """
        # Base confidence from wrist height above shoulder.
        # Scale: 0.06 (barely above) -> 0.3, 0.20+ (clearly above) -> 0.6
        base = min(
            0.6,
            0.3 + (wrist_above_shoulder - _MIN_WRIST_ABOVE_SHOULDER) * 3.0,
        )

        # Trophy position bonus: strong indicator of a serve
        trophy_bonus = 0.15 if has_trophy else 0.0

        # Knee bend bonus: scaled from 0 to 0.12 based on bend depth
        knee_bonus = 0.0
        if has_knee_bend:
            knee_bonus = min(0.12, knee_bend_depth / 30.0 * 0.12)

        # Body extension bonus: ratio > 1.03 suggests upward leg drive
        extension_bonus = 0.0
        if extension_ratio > 1.03:
            extension_bonus = min(0.08, (extension_ratio - 1.03) * 0.5)

        # Toss arm bonus: scaled by toss signal strength
        toss_bonus = toss_signal * 0.08

        # Duration signal: serves in slow-mo are typically 0.8-6 seconds
        duration_bonus = 0.0
        if 800.0 <= duration_ms <= 6000.0:
            duration_bonus = 0.05
        elif 500.0 <= duration_ms <= 10_000.0:
            duration_bonus = 0.02

        # Camera angle adjustment
        camera_bonus = 0.0
        if camera_angle == "behind":
            camera_bonus = 0.02
        elif camera_angle == "side":
            # Side view partially obscures the vertical trajectory
            camera_bonus = -0.03

        total = (
            base
            + trophy_bonus
            + knee_bonus
            + extension_bonus
            + toss_bonus
            + duration_bonus
            + camera_bonus
        )
        return max(0.0, min(1.0, total))

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phases(
        frames: list[dict],
        wrist_ys: list[tuple[int, float]],
        peak_idx: int,
        trophy_idx: int | None,
        knee_angles: list[tuple[int, float]],
    ) -> dict[str, dict[str, int | float]]:
        """Assign frame indices and timestamps to each serve phase.

        Phases: preparation, backswing, trophy_position, forward_swing,
        contact, follow_through.

        Returns a dict mapping phase names to ``{frame_index, timestamp_ms}``.
        """
        if not wrist_ys or peak_idx is None:
            return {}

        num_frames = len(frames)

        def _frame_info(idx: int) -> dict[str, int | float]:
            clamped = max(0, min(idx, num_frames - 1))
            return {
                "frame_index": frames[clamped].get("frame_index", clamped),
                "timestamp_ms": frames[clamped].get("timestamp_ms", 0.0),
            }

        # Key anchors
        contact_idx = peak_idx
        trophy_resolved = (
            trophy_idx
            if trophy_idx is not None
            else max(0, peak_idx - max(1, num_frames // 6))
        )

        # Preparation: start of sequence
        preparation_idx = 0

        # Backswing: midpoint between preparation and trophy
        backswing_idx = max(1, trophy_resolved // 2)

        # Forward swing: midpoint between trophy and contact
        forward_swing_idx = trophy_resolved + (contact_idx - trophy_resolved) // 2
        forward_swing_idx = max(
            trophy_resolved + 1, min(forward_swing_idx, num_frames - 1)
        )

        # Follow-through: midpoint between contact and end
        follow_through_idx = min(
            num_frames - 1,
            contact_idx + max(1, (num_frames - contact_idx) // 2),
        )

        return {
            "preparation": _frame_info(preparation_idx),
            "backswing": _frame_info(backswing_idx),
            "trophy_position": _frame_info(trophy_resolved),
            "forward_swing": _frame_info(forward_swing_idx),
            "contact": _frame_info(contact_idx),
            "follow_through": _frame_info(follow_through_idx),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _no_match(self, camera_angle: str, handedness: str) -> ClassificationResult:
        """Return a zero-confidence result indicating no serve detected."""
        return ClassificationResult(
            shot_type=self.shot_type,
            confidence=0.0,
            is_clean=False,
            camera_angle=camera_angle,
            handedness=handedness,
            phase_timestamps=None,
        )
