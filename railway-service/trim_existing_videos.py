"""
Trim existing pro video files to contain only a single clean shot.

Most files were downloaded as full YouTube compilations (4-33 minutes)
but should be 3-8 seconds of a single slow-motion shot.

This script:
1. Trims each video to the specified time range using ffmpeg
2. Re-extracts keypoints from the trimmed video
3. Updates the Supabase database with the new keypoints
4. Replaces the original file with the trimmed version

Usage:
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python trim_existing_videos.py [--dry-run]
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Import just the extraction function without triggering main.py's module-level Supabase init
import importlib.util
_spec = importlib.util.spec_from_file_location("main", str(Path(__file__).parent / "main.py"), submodule_search_locations=[])
# We can't import main.py directly because it creates a Supabase client at module scope.
# Instead, inline the extraction function import path.
# The extraction function only needs cv2, mediapipe, numpy, and math — no Supabase.
_extract_available = False
try:
    from main import extract_keypoints_from_video
    _extract_available = True
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRO_VIDEOS_DIR = PROJECT_ROOT / "public" / "pro-videos"
FFMPEG = str(PROJECT_ROOT / "pro-videos" / "bin" / "ffmpeg")
FFPROBE = str(PROJECT_ROOT / "pro-videos" / "bin" / "ffprobe")

# For each file, specify (start_time, end_time) in seconds.
# These are the portions that contain a single clean shot.
# Videos not listed here are assumed to already be the correct length.
TRIM_MAP = {
    # ROGER FEDERER
    "roger_federer_forehand_forehand_side.mp4": (0.0, 6.0),
    "roger_federer_forehand_forehand_court_level.mp4": (0.0, 6.0),
    "roger_federer_backhand_backhand_side.mp4": (0.0, 6.0),
    "roger_federer_serve_serve_side.mp4": (0.0, 8.0),

    # RAFAEL NADAL
    "rafael_nadal_forehand_forehand_practice.mp4": (0.0, 6.0),
    "rafael_nadal_forehand_forehand_court_level.mp4": (0.0, 6.0),
    "rafael_nadal_forehand_forehand_aus_open.mp4": (0.0, 6.0),
    "rafael_nadal_serve_serve_aus_open.mp4": (0.0, 8.0),

    # NOVAK DJOKOVIC
    "novak_djokovic_forehand_forehand_behind.mp4": (0.0, 6.0),
    "novak_djokovic_forehand_forehand_court_level.mp4": (0.0, 6.0),
    "novak_djokovic_backhand_backhand_side.mp4": (0.0, 6.0),
    "novak_djokovic_backhand_backhand_behind.mp4": (0.0, 6.0),
    "novak_djokovic_serve_serve_behind.mp4": (0.0, 8.0),
    "novak_djokovic_serve_serve_court_level.mp4": (0.0, 8.0),

    # SERENA WILLIAMS
    "serena_williams_forehand_forehand_practice.mp4": (0.0, 6.0),
    "serena_williams_backhand_backhand_super_slow_mo.mp4": (0.0, 6.0),
    "serena_williams_serve_serve_volley_practice.mp4": (0.0, 8.0),

    # CARLOS ALCARAZ
    "carlos_alcaraz_forehand_forehand_side.mp4": (0.0, 6.0),
    "carlos_alcaraz_forehand_forehand_indian_wells_2024.mp4": (0.0, 6.0),
    "carlos_alcaraz_backhand_backhand_side.mp4": (0.0, 6.0),
    "carlos_alcaraz_serve_serve_side.mp4": (0.0, 8.0),
    "carlos_alcaraz_serve_serve_court_level.mp4": (0.0, 8.0),

    # JANNIK SINNER
    "jannik_sinner_forehand_forehand_side.mp4": (0.0, 6.0),
    "jannik_sinner_forehand_forehand_court_level.mp4": (0.0, 6.0),
    "jannik_sinner_forehand_forehand_practice_sonego.mp4": (0.0, 6.0),
    "jannik_sinner_backhand_backhand_side.mp4": (0.0, 6.0),
    "jannik_sinner_serve_serve_side.mp4": (0.0, 8.0),
    "jannik_sinner_serve_serve_iw_practice.mp4": (0.0, 8.0),

    # IGA SWIATEK
    "iga_swiatek_forehand_forehand_slow_motion.mp4": (0.0, 6.0),
    "iga_swiatek_forehand_forehand_court_level_4k.mp4": (0.0, 6.0),
    "iga_swiatek_forehand_forehand_ao_2021.mp4": (0.0, 6.0),
    "iga_swiatek_backhand_backhand_4k.mp4": (0.0, 6.0),
    "iga_swiatek_serve_serve_4k.mp4": (0.0, 8.0),

    # NAOMI OSAKA
    "naomi_osaka_forehand_forehand_practice.mp4": (0.0, 6.0),
    "naomi_osaka_backhand_backhand_practice.mp4": (0.0, 6.0),
    "naomi_osaka_serve_serve_practice.mp4": (0.0, 8.0),

    # DANIIL MEDVEDEV
    "daniil_medvedev_forehand_forehand_side.mp4": (0.0, 6.0),
    "daniil_medvedev_forehand_forehand_court_level.mp4": (0.0, 6.0),
    "daniil_medvedev_backhand_backhand_side.mp4": (0.0, 6.0),
    "daniil_medvedev_serve_serve_court_level.mp4": (0.0, 8.0),

    # ANDY RODDICK
    "andy_roddick_serve_serve_slow_motion.mp4": (0.0, 8.0),
}


def get_duration(path: str) -> float:
    """Get video duration in seconds."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def trim_video(input_path: str, output_path: str, start: float, end: float) -> bool:
    """Trim video with re-encoding for frame-accurate cuts."""
    duration = end - start
    try:
        result = subprocess.run(
            [
                FFMPEG, "-y",
                "-ss", f"{start:.2f}",
                "-i", input_path,
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"    ffmpeg error: {e}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv
    update_db = "--update-db" in sys.argv

    supabase = None
    if update_db:
        from supabase import create_client
        supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    print(f"Pro videos directory: {PRO_VIDEOS_DIR}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    if update_db:
        print("Will update Supabase with new keypoints")
    print()

    trimmed = 0
    skipped = 0
    errors = 0

    for filename, (start, end) in TRIM_MAP.items():
        filepath = PRO_VIDEOS_DIR / filename
        if not filepath.exists():
            print(f"SKIP {filename} (file not found)")
            skipped += 1
            continue

        current_duration = get_duration(str(filepath))
        target_duration = end - start

        # Skip if already trimmed (within 1 second tolerance)
        if current_duration <= target_duration + 1.0:
            print(f"OK   {filename} ({current_duration:.1f}s, already trimmed)")
            skipped += 1
            continue

        print(f"TRIM {filename}: {current_duration:.1f}s -> {target_duration:.1f}s ({start:.1f}-{end:.1f})")

        if dry_run:
            trimmed += 1
            continue

        # Trim to temp file, then replace original
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not trim_video(str(filepath), tmp_path, start, end):
                print(f"  ERROR: ffmpeg trim failed")
                errors += 1
                continue

            new_duration = get_duration(tmp_path)
            if new_duration < 1.0:
                print(f"  ERROR: trimmed file too short ({new_duration:.1f}s)")
                errors += 1
                continue

            # Replace original with trimmed version
            shutil.move(tmp_path, str(filepath))
            print(f"  Trimmed to {new_duration:.1f}s")

            # Re-extract keypoints from trimmed video
            if _extract_available:
                print(f"  Re-extracting keypoints...")
                keypoints = extract_keypoints_from_video(str(filepath), sample_fps=30, max_seconds=0)
                print(f"  Extracted {keypoints['frame_count']} frames ({keypoints.get('duration_ms', 0)}ms)")
            else:
                keypoints = None
                print(f"  Skipping keypoint extraction (main.py import failed, set SUPABASE env vars)")

            # Update database if requested
            if supabase and keypoints and keypoints["frame_count"] > 0:
                video_url = f"/pro-videos/{filename}"
                result = supabase.table("pro_swings").update({
                    "keypoints_json": keypoints,
                    "frame_count": keypoints["frame_count"],
                    "duration_ms": keypoints.get("duration_ms"),
                    "fps": keypoints["fps_sampled"],
                }).eq("video_url", video_url).execute()
                updated = len(result.data) if result.data else 0
                print(f"  Updated {updated} DB row(s)")

            trimmed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    print(f"\nDone: {trimmed} trimmed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
