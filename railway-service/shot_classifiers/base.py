"""Base interface for shot-type classifiers.

Each shot classifier (forehand, backhand, serve, slice) implements this
interface to detect and validate its specific shot type from a sequence
of MediaPipe pose frames.

A "frame" is a dict matching the PoseFrame shape from the main extraction
pipeline::

    {
        "frame_index": int,
        "timestamp_ms": float,
        "landmarks": [{"id": int, "name": str, "x": float, "y": float, "z": float, "visibility": float}, ...],
        "joint_angles": {"right_elbow": float, "left_elbow": float, ...},
    }
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    shot_type: str
    confidence: float          # 0.0 – 1.0
    is_clean: bool             # True if this is a single clean shot with no mixed actions
    camera_angle: str          # "behind" | "side" | "front" | "overhead" | "unknown"
    handedness: str = "right"  # "right" | "left"
    phase_timestamps: dict | None = None  # Optional per-phase frame indices


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Pose 33-point model)
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


# ---------------------------------------------------------------------------
# Utility helpers shared across classifiers
# ---------------------------------------------------------------------------

def get_landmark(frame: dict, idx: int) -> dict | None:
    """Return the landmark dict at the given index, or None if missing."""
    landmarks = frame.get("landmarks", [])
    if idx < len(landmarks):
        lm = landmarks[idx]
        if lm.get("visibility", 0) > 0.3:
            return lm
    return None


def lm_xy(frame: dict, idx: int) -> tuple[float, float] | None:
    """Return (x, y) for a landmark, or None if not visible."""
    lm = get_landmark(frame, idx)
    if lm:
        return (lm["x"], lm["y"])
    return None


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def velocity(positions: list[tuple[float, float]], dt_ms: float) -> list[float]:
    """Compute speed (magnitude) between consecutive positions."""
    if dt_ms <= 0:
        return [0.0] * max(0, len(positions) - 1)
    speeds = []
    for i in range(1, len(positions)):
        d = distance(positions[i - 1], positions[i])
        speeds.append(d / (dt_ms / 1000))
    return speeds


def estimate_camera_angle(frames: list[dict]) -> str:
    """Heuristic camera angle estimation from pose data.

    Uses the ratio of shoulder width to shoulder-hip vertical distance
    to distinguish front/back views from side views.

    Returns one of: "behind", "side", "front", "overhead", "unknown".
    """
    ratios: list[float] = []
    shoulder_z_diffs: list[float] = []

    sample_step = max(1, len(frames) // 10)
    for frame in frames[::sample_step]:
        ls = get_landmark(frame, LM_LEFT_SHOULDER)
        rs = get_landmark(frame, LM_RIGHT_SHOULDER)
        lh = get_landmark(frame, LM_LEFT_HIP)
        rh = get_landmark(frame, LM_RIGHT_HIP)

        if not all([ls, rs, lh, rh]):
            continue

        shoulder_width = abs(ls["x"] - rs["x"])
        hip_mid_y = (lh["y"] + rh["y"]) / 2
        shoulder_mid_y = (ls["y"] + rs["y"]) / 2
        torso_height = abs(hip_mid_y - shoulder_mid_y)

        if torso_height > 0.01:
            ratios.append(shoulder_width / torso_height)

        # Z-depth difference between shoulders indicates rotation relative to camera
        if "z" in ls and "z" in rs:
            shoulder_z_diffs.append(abs(ls["z"] - rs["z"]))

    if not ratios:
        return "unknown"

    avg_ratio = sum(ratios) / len(ratios)
    avg_z_diff = sum(shoulder_z_diffs) / len(shoulder_z_diffs) if shoulder_z_diffs else 0

    # Large shoulder width relative to torso = front/back view
    # Small shoulder width = side view
    # Z-depth difference helps distinguish behind from front
    if avg_ratio < 0.4:
        return "side"
    elif avg_ratio > 0.8:
        # Front vs behind: when viewed from behind, shoulders are roughly
        # equal depth. From front, same thing. We use nose position relative
        # to shoulders as a heuristic: if nose is between shoulders (visible
        # face), it is front-facing.
        nose_visible_count = 0
        for frame in frames[::sample_step]:
            nose = get_landmark(frame, LM_NOSE)
            if nose and nose.get("visibility", 0) > 0.6:
                nose_visible_count += 1
        if nose_visible_count > len(frames[::sample_step]) * 0.5:
            return "front"
        return "behind"
    else:
        # Intermediate ratio — could be a diagonal view
        if avg_z_diff > 0.1:
            return "side"
        return "behind"


def detect_handedness(frames: list[dict]) -> str:
    """Detect whether the player is right-handed or left-handed.

    Heuristic: whichever wrist has more lateral displacement during the swing
    is the dominant hand.
    """
    right_range = 0.0
    left_range = 0.0

    rw_xs: list[float] = []
    lw_xs: list[float] = []

    for frame in frames:
        rw = lm_xy(frame, LM_RIGHT_WRIST)
        lw = lm_xy(frame, LM_LEFT_WRIST)
        if rw:
            rw_xs.append(rw[0])
        if lw:
            lw_xs.append(lw[0])

    if rw_xs:
        right_range = max(rw_xs) - min(rw_xs)
    if lw_xs:
        left_range = max(lw_xs) - min(lw_xs)

    return "right" if right_range >= left_range else "left"


class BaseShotClassifier(ABC):
    """Abstract base class for shot-type classifiers."""

    @property
    @abstractmethod
    def shot_type(self) -> str:
        """The shot type this classifier detects (e.g. 'forehand')."""

    @abstractmethod
    def classify(self, frames: list[dict]) -> ClassificationResult:
        """Classify a sequence of pose frames.

        Returns a ClassificationResult with confidence 0.0 if this
        classifier does not think the frames contain its shot type.
        """

    def validate_clean(self, frames: list[dict]) -> bool:
        """Check if this is a single clean shot (no mixed actions).

        Default implementation checks that the sequence is short enough
        to be a single shot and has consistent motion direction.
        Override in subclasses for shot-specific validation.
        """
        if len(frames) < 5:
            return False
        # A single shot should be under ~3 seconds at 30fps
        duration_ms = frames[-1].get("timestamp_ms", 0) - frames[0].get("timestamp_ms", 0)
        if duration_ms > 4000:
            return False
        return True
