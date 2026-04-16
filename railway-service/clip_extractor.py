"""Clean clip extraction and upload for tennis shots.

Extracts individual shot clips from a longer video, validates them,
and uploads to storage.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedClip:
    """A single extracted tennis shot clip."""
    path: str                 # Local file path
    start_time: float         # Start time in source video (seconds)
    end_time: float           # End time in source video (seconds)
    shot_type: str            # forehand, backhand, serve, slice
    camera_angle: str         # behind, side, front, overhead
    confidence: float         # Classification confidence
    duration_ms: int          # Clip duration in milliseconds
    fps: float                # Source video FPS
    is_clean: bool            # Whether this is a clean single shot
    handedness: str           # right or left


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str | None = None,
    reencode: bool = False,
) -> str:
    """Extract a clip from a video file using ffmpeg.

    Args:
        video_path: Source video path.
        start_sec: Start timestamp in seconds.
        end_sec: End timestamp in seconds.
        output_path: Destination path. Auto-generated if None.
        reencode: If True, re-encode for precise cuts. If False, use
                  stream copy (fast but may have keyframe alignment issues).

    Returns:
        Path to the extracted clip.
    """
    if output_path is None:
        suffix = Path(video_path).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = tmp.name
        tmp.close()

    duration = end_sec - start_sec
    # Add small padding for clean boundaries
    adjusted_start = max(0, start_sec - 0.1)
    adjusted_duration = duration + 0.2

    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{adjusted_start:.3f}",
            "-i", video_path,
            "-t", f"{adjusted_duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{adjusted_start:.3f}",
            "-i", video_path,
            "-t", f"{adjusted_duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

    subprocess.run(cmd, capture_output=True, timeout=120)

    if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        raise RuntimeError(f"Failed to extract clip: {start_sec:.1f}s - {end_sec:.1f}s")

    return output_path


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe.

    Returns:
        Dict with keys: duration, fps, width, height, codec.
    """
    import json

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path,
        ],
        capture_output=True, text=True, timeout=30,
    )

    info = json.loads(result.stdout)

    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse FPS from r_frame_rate (e.g. "30/1" or "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "fps": round(fps, 2),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
    }


def compute_real_time_rate(clip_duration_ms: int, shot_type: str) -> float:
    """Compute the playback rate needed to show a clip at realistic speed.

    Many pro tennis clips are slow-motion. This estimates how much to
    speed them up to show at real-time pace.

    Args:
        clip_duration_ms: Actual clip duration in milliseconds.
        shot_type: The shot type for estimating real-time duration.

    Returns:
        Playback rate multiplier (e.g. 4.0 means play 4x speed for real-time).
    """
    # Estimated real-time durations for each shot type (in ms)
    REAL_TIME_DURATIONS = {
        "forehand": 700,
        "backhand": 750,
        "serve": 1500,
        "slice": 800,
        "volley": 500,
    }

    real_duration = REAL_TIME_DURATIONS.get(shot_type, 800)

    if clip_duration_ms <= 0:
        return 1.0

    rate = clip_duration_ms / real_duration
    # Clamp to browser-supported range
    return max(0.0625, min(16.0, rate))
