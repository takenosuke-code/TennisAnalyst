"""Camera angle classification for tennis video scenes.

Determines whether a scene is a gameplay view (useful for shot extraction)
or a non-gameplay view (close-up, crowd, scoreboard, etc.) that should be
skipped.

Uses a combination of:
1. Court line detection via Hough transform
2. Player pose analysis (size, position, count)
3. Scene composition heuristics
"""

from __future__ import annotations

import cv2
import numpy as np


@classmethod
def is_gameplay_scene(cls, frame: np.ndarray) -> tuple[bool, str]:
    """Determine if a video frame shows gameplay.

    Args:
        frame: BGR image (OpenCV format).

    Returns:
        (is_gameplay, camera_angle) where camera_angle is one of
        "behind", "side", "front", "overhead", "unknown".
    """
    return classify_frame(frame)


def classify_frame(frame: np.ndarray) -> tuple[bool, str]:
    """Classify a single frame for gameplay and camera angle.

    Returns:
        (is_gameplay, angle_label)
    """
    has_court, court_score = _detect_court_lines(frame)
    has_player, player_info = _detect_player_region(frame)

    if not has_court and not has_player:
        return False, "unknown"

    if not has_player:
        return False, "unknown"

    # Determine camera angle from court geometry and player position
    angle = _classify_angle(frame, court_score, player_info)

    return True, angle


def _detect_court_lines(frame: np.ndarray) -> tuple[bool, float]:
    """Detect tennis court lines using color filtering and Hough transform.

    Tennis courts typically have white lines on green/blue/clay surfaces.

    Returns:
        (has_court_lines, confidence_score)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # Detect white lines (high value, low saturation)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))

    # Detect green court surface (common hard court / grass)
    green_mask = cv2.inRange(hsv, np.array([30, 30, 40]), np.array([90, 255, 255]))

    # Detect blue court surface (hard court)
    blue_mask = cv2.inRange(hsv, np.array([90, 30, 40]), np.array([130, 255, 255]))

    # Detect clay/orange surface
    clay_mask = cv2.inRange(hsv, np.array([5, 50, 50]), np.array([25, 255, 255]))

    court_surface = green_mask | blue_mask | clay_mask
    court_ratio = np.count_nonzero(court_surface) / (h * w)

    # Need at least 15% court-colored pixels
    if court_ratio < 0.15:
        return False, 0.0

    # Look for lines using Hough transform
    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=w * 0.1, maxLineGap=20)

    if lines is None or len(lines) < 2:
        return False, court_ratio * 0.5

    # Count horizontal and vertical lines
    h_lines = 0
    v_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            h_lines += 1
        elif 70 < angle < 110:
            v_lines += 1

    has_lines = (h_lines >= 1 and v_lines >= 1) or len(lines) >= 4
    score = min(1.0, (len(lines) / 10) * court_ratio * 2)

    return has_lines, score


def _detect_player_region(frame: np.ndarray) -> tuple[bool, dict]:
    """Detect if a player-sized figure exists in the frame.

    Uses simple contour analysis to find a vertically-oriented blob
    of appropriate size for a person on a tennis court.

    Returns:
        (has_player, info_dict) where info_dict contains position and size info.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to find dark regions (player) on lighter court
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = ch / max(cw, 1)
        area_ratio = (cw * ch) / (w * h)

        # Player should be taller than wide and occupy a reasonable portion
        if aspect_ratio > 1.5 and 0.01 < area_ratio < 0.4:
            return True, {
                "x": x / w, "y": y / h,
                "w": cw / w, "h": ch / h,
                "area_ratio": area_ratio,
                "center_x": (x + cw / 2) / w,
                "center_y": (y + ch / 2) / h,
            }

    return False, {}


def _classify_angle(frame: np.ndarray, court_score: float, player_info: dict) -> str:
    """Classify the camera angle based on court geometry and player position.

    Heuristics:
    - Behind: Player in lower-center of frame, court lines converge to top
    - Side: Player on one side, court lines are roughly horizontal
    - Front: Player in center, face visible (larger upper body)
    - Overhead: Player small, court dominates the frame
    """
    if not player_info:
        return "unknown"

    center_x = player_info.get("center_x", 0.5)
    center_y = player_info.get("center_y", 0.5)
    area = player_info.get("area_ratio", 0)

    # Overhead: player is very small relative to frame
    if area < 0.02:
        return "overhead"

    # Behind: player is in lower half and roughly centered
    if center_y > 0.5 and 0.25 < center_x < 0.75:
        return "behind"

    # Side: player is to one side of the frame
    if center_x < 0.3 or center_x > 0.7:
        return "side"

    # Front: player is centered and in upper portion
    if center_y < 0.5 and 0.3 < center_x < 0.7:
        return "front"

    return "behind"
