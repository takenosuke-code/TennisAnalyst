"""YouTube video processing pipeline for tennis shot extraction.

Orchestrates the full pipeline:
1. Download YouTube video at 720p via yt-dlp
2. Detect scene boundaries
3. Classify scenes for gameplay and camera angle
4. Extract pose data from gameplay scenes
5. Classify shots by type
6. Extract clean individual clips
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yt_dlp

from scene_detector import detect_scenes, Scene
from camera_classifier import classify_frame
from clip_extractor import extract_clip, get_video_info, ExtractedClip, compute_real_time_rate
from shot_classifiers import classify_shot
from main import extract_keypoints_from_video


@dataclass
class ProcessingResult:
    """Result of processing a YouTube video."""
    youtube_url: str
    video_title: str
    total_scenes: int
    gameplay_scenes: int
    clips: list[ExtractedClip] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def download_youtube_video(url: str, output_dir: str | None = None, max_duration: int = 600) -> tuple[str, dict]:
    """Download a YouTube video at 720p.

    Args:
        url: YouTube URL.
        output_dir: Directory for the downloaded file. Uses temp dir if None.
        max_duration: Maximum video duration in seconds. Rejects longer videos.

    Returns:
        (file_path, metadata_dict)

    Raises:
        ValueError: If the video is too long or unavailable.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="tennis_analyst_")

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    # First pass: get metadata without downloading
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError(f"Could not extract info from URL: {url}")
        duration = info.get("duration", 0)
        if duration > max_duration:
            raise ValueError(
                f"Video too long: {duration}s (max {max_duration}s). "
                "Submit shorter clips or timestamps."
            )

    # Second pass: download at 720p
    opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "outtmpl": output_template,
        "no_playlist": True,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    video_id = info.get("id", "unknown")
    expected_path = os.path.join(output_dir, f"{video_id}.mp4")
    if not os.path.exists(expected_path):
        # Try to find any mp4 in the output dir
        mp4s = list(Path(output_dir).glob("*.mp4"))
        if mp4s:
            expected_path = str(mp4s[0])
        else:
            raise ValueError("Download completed but no video file found")

    metadata = {
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", ""),
        "video_id": video_id,
        "original_url": url,
    }

    return expected_path, metadata


def sample_frame(video_path: str, timestamp_sec: float) -> "np.ndarray | None":
    """Extract a single frame from a video at a given timestamp."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _filter_gameplay_scenes(
    video_path: str,
    scenes: list[Scene],
) -> list[tuple[Scene, str]]:
    """Filter scenes to only gameplay views and classify camera angles.

    Returns:
        List of (scene, camera_angle) tuples for gameplay scenes.
    """
    gameplay: list[tuple[Scene, str]] = []

    for scene in scenes:
        # Sample frame from the middle of the scene
        mid_time = (scene.start_time + scene.end_time) / 2
        frame = sample_frame(video_path, mid_time)
        if frame is None:
            continue

        is_gameplay, angle = classify_frame(frame)
        if is_gameplay:
            gameplay.append((scene, angle))

    return gameplay


def process_youtube_video(
    url: str,
    target_shot_types: list[str] | None = None,
    max_duration: int = 600,
    sample_fps: int = 30,
) -> ProcessingResult:
    """Process a YouTube video end-to-end.

    Args:
        url: YouTube video URL.
        target_shot_types: If set, only extract these shot types.
            e.g. ["forehand"] to only extract forehands.
        max_duration: Maximum video duration in seconds.
        sample_fps: FPS for pose extraction.

    Returns:
        ProcessingResult with extracted clips and metadata.
    """
    result = ProcessingResult(
        youtube_url=url,
        video_title="",
        total_scenes=0,
        gameplay_scenes=0,
    )

    video_path = None
    try:
        # Step 1: Download
        video_path, metadata = download_youtube_video(url, max_duration=max_duration)
        result.video_title = metadata.get("title", "")

        video_info = get_video_info(video_path)

        # Step 2: Detect scenes
        scenes = detect_scenes(video_path)
        result.total_scenes = len(scenes)

        # Step 3: Filter to gameplay scenes
        gameplay_scenes = _filter_gameplay_scenes(video_path, scenes)
        result.gameplay_scenes = len(gameplay_scenes)

        # Step 4-5: For each gameplay scene, extract poses and classify shot
        for scene, camera_angle in gameplay_scenes:
            try:
                # Skip very short scenes (< 0.5s) and very long ones (> 10s)
                if scene.duration < 0.5 or scene.duration > 10:
                    continue

                # Extract scene clip to a temp file for pose processing
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    clip_path = tmp.name

                try:
                    extract_clip(video_path, scene.start_time, scene.end_time, clip_path, reencode=True)

                    # Extract keypoints from the clip
                    keypoints = extract_keypoints_from_video(clip_path, sample_fps=sample_fps)
                    frames = keypoints.get("frames", [])

                    if len(frames) < 5:
                        continue

                    # Classify the shot type
                    classification = classify_shot(frames)

                    if classification.confidence < 0.3:
                        continue

                    # Filter by target shot types if specified
                    if target_shot_types and classification.shot_type not in target_shot_types:
                        continue

                    if not classification.is_clean:
                        continue

                    duration_ms = keypoints.get("duration_ms", int(scene.duration * 1000))
                    real_time_rate = compute_real_time_rate(duration_ms, classification.shot_type)

                    result.clips.append(ExtractedClip(
                        path=clip_path,
                        start_time=scene.start_time,
                        end_time=scene.end_time,
                        shot_type=classification.shot_type,
                        camera_angle=classification.camera_angle or camera_angle,
                        confidence=classification.confidence,
                        duration_ms=duration_ms,
                        fps=video_info.get("fps", 30.0),
                        is_clean=classification.is_clean,
                        handedness=classification.handedness,
                    ))

                except Exception as e:
                    result.errors.append(f"Scene {scene.index}: {e}")
                finally:
                    # Only clean up clips that weren't kept
                    if not any(c.path == clip_path for c in result.clips):
                        if os.path.exists(clip_path):
                            os.unlink(clip_path)

            except Exception as e:
                result.errors.append(f"Scene {scene.index} outer: {e}")

    except Exception as e:
        result.errors.append(f"Pipeline error: {e}")

    finally:
        # Clean up downloaded video (but NOT extracted clips)
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
            # Clean up the temp directory too
            parent = Path(video_path).parent
            if str(parent).startswith(tempfile.gettempdir()):
                try:
                    parent.rmdir()
                except OSError:
                    pass

    return result


async def process_youtube_video_async(
    url: str,
    target_shot_types: list[str] | None = None,
    max_duration: int = 600,
    sample_fps: int = 30,
) -> ProcessingResult:
    """Async wrapper for process_youtube_video."""
    return await asyncio.to_thread(
        process_youtube_video,
        url,
        target_shot_types,
        max_duration,
        sample_fps,
    )
