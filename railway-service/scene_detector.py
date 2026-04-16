"""Scene detection for tennis videos using PySceneDetect.

Splits a video into individual scenes (camera cuts / shot boundaries)
so that each scene can be independently classified for shot type and
camera angle.
"""

from __future__ import annotations

import subprocess
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scene:
    """A single scene (camera shot) within a video."""
    index: int
    start_time: float     # seconds
    end_time: float       # seconds
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len_sec: float = 0.5) -> list[Scene]:
    """Detect scene boundaries in a video using PySceneDetect.

    Uses the ContentDetector for clean cuts (broadcast footage) and
    falls back to adaptive detection for amateur footage with gradual
    transitions.

    Args:
        video_path: Path to the video file.
        threshold: Content detector threshold. Higher = fewer splits.
        min_scene_len_sec: Minimum scene duration in seconds.

    Returns:
        List of Scene objects sorted by start time.
    """
    try:
        from scenedetect import detect, ContentDetector, AdaptiveDetector
    except ImportError:
        # Fallback: treat entire video as one scene
        return _fallback_single_scene(video_path)

    try:
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
    except Exception:
        try:
            scene_list = detect(video_path, AdaptiveDetector())
        except Exception:
            return _fallback_single_scene(video_path)

    scenes: list[Scene] = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        if (end_sec - start_sec) < min_scene_len_sec:
            continue
        scenes.append(Scene(
            index=i,
            start_time=round(start_sec, 3),
            end_time=round(end_sec, 3),
            start_frame=start.get_frames(),
            end_frame=end.get_frames(),
        ))

    # If no scenes were detected (e.g. amateur single-shot video), treat
    # the whole video as one scene
    if not scenes:
        return _fallback_single_scene(video_path)

    return scenes


def _fallback_single_scene(video_path: str) -> list[Scene]:
    """Create a single scene spanning the entire video."""
    duration = _get_video_duration(video_path)
    return [Scene(index=0, start_time=0.0, end_time=duration, start_frame=0, end_frame=0)]


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 300.0  # Default 5 minutes


def extract_scene_clip(
    video_path: str,
    scene: Scene,
    output_path: str,
    padding_sec: float = 0.2,
) -> str:
    """Extract a scene as a separate video clip using ffmpeg.

    Uses stream copy (no re-encoding) for speed.

    Args:
        video_path: Source video path.
        scene: Scene to extract.
        output_path: Destination file path.
        padding_sec: Extra time before/after the scene boundaries.

    Returns:
        The output_path.
    """
    start = max(0, scene.start_time - padding_sec)
    duration = scene.duration + 2 * padding_sec

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ],
        capture_output=True, timeout=60,
    )

    return output_path
