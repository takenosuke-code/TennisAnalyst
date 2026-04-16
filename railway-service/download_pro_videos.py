"""
Migration script: Download YouTube videos for pro_swings and store locally.

Reads all pro_swings from Supabase, downloads any that still have a YouTube URL
as video_url, saves to public/pro-videos/{sanitized}.mp4, and updates the DB
to point to the local path.

Usage:
  cd railway-service
  pip install -r requirements.txt
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python download_pro_videos.py

Reads NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY from ../.env as
fallbacks if the standard env vars are not set.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Load ../.env so we can pick up NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Missing Supabase credentials.")
    print("Set SUPABASE_URL + SUPABASE_SERVICE_KEY, or ensure ../.env has")
    print("NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Output directory (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRO_VIDEOS_DIR = PROJECT_ROOT / "public" / "pro-videos"


def is_youtube_url(url: str) -> bool:
    """Return True if the URL points to YouTube."""
    if not url:
        return False
    return "youtube.com" in url or "youtu.be" in url


def sanitize_filename(name: str) -> str:
    """Convert a string into a safe filename component (lowercase, underscores)."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def download_youtube(url: str, output_path: str) -> bool:
    """Download a YouTube video to output_path using yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "mp4[height<=720]/best[ext=mp4]/best",
                "--no-playlist",
                "-o", output_path,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"    yt-dlp stderr: {result.stderr[:300]}")
        return result.returncode == 0
    except Exception as e:
        print(f"    yt-dlp failed: {e}")
        return False


def main():
    print("Pro Video Migration: YouTube -> local files")
    print("=" * 60)

    # Ensure output directory exists
    PRO_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {PRO_VIDEOS_DIR}")

    # Fetch all pro_swings joined with pro name
    swings = (
        supabase.table("pro_swings")
        .select("id, video_url, shot_type, metadata, pro_id, pros(name)")
        .execute()
    )

    if not swings.data:
        print("No pro_swings found in database.")
        return

    total = len(swings.data)
    youtube_count = 0
    skipped_count = 0
    success_count = 0
    fail_count = 0

    for swing in swings.data:
        swing_id = swing["id"]
        video_url = swing.get("video_url") or ""
        shot_type = swing.get("shot_type", "unknown")
        metadata = swing.get("metadata") or {}
        label = metadata.get("label", shot_type)

        # Get pro name from the joined relation
        pro_info = swing.get("pros")
        if isinstance(pro_info, dict):
            pro_name = pro_info.get("name", "unknown")
        elif isinstance(pro_info, list) and pro_info:
            pro_name = pro_info[0].get("name", "unknown")
        else:
            pro_name = "unknown"

        # Skip if not a YouTube URL (already migrated or local)
        if not is_youtube_url(video_url):
            skipped_count += 1
            continue

        youtube_count += 1
        sanitized_name = f"{sanitize_filename(pro_name)}_{sanitize_filename(shot_type)}_{sanitize_filename(label)}"
        filename = f"{sanitized_name}.mp4"
        local_path = PRO_VIDEOS_DIR / filename
        web_path = f"/pro-videos/{filename}"

        print(f"\n  [{youtube_count}] {pro_name} - {shot_type} ({label})")
        print(f"    YouTube: {video_url}")
        print(f"    -> {web_path}")

        # Skip download if file already exists locally
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"    File already exists ({local_path.stat().st_size} bytes), updating DB only.")
            supabase.table("pro_swings").update({"video_url": web_path}).eq("id", swing_id).execute()
            success_count += 1
            continue

        if not download_youtube(video_url, str(local_path)):
            print(f"    FAILED to download.")
            fail_count += 1
            continue

        file_size = local_path.stat().st_size if local_path.exists() else 0
        print(f"    Downloaded ({file_size} bytes)")

        # Update Supabase to point to local path
        supabase.table("pro_swings").update({"video_url": web_path}).eq("id", swing_id).execute()
        print(f"    DB updated.")
        success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Migration complete.")
    print(f"  Total swings:   {total}")
    print(f"  Skipped (non-YT): {skipped_count}")
    print(f"  YouTube found:  {youtube_count}")
    print(f"  Downloaded OK:  {success_count}")
    print(f"  Failed:         {fail_count}")


if __name__ == "__main__":
    main()
