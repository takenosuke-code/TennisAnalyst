"""
Local Pro Seeder — Extract keypoints from a local video file and insert into Supabase.
No Railway deployment needed.

Usage:
  cd railway-service
  pip install -r requirements.txt
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python seed_local.py \
    --video /path/to/federer_forehand.mp4 \
    --pro "Roger Federer" \
    --shot-type forehand

The video should be a 2-5 second clip of a single swing, ideally slow-motion
with the player's full body visible.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from main import extract_keypoints_from_video
from supabase import create_client


def main():
    parser = argparse.ArgumentParser(description="Seed a pro swing from a local video file")
    parser.add_argument("--video", required=True, help="Path to the video file (MP4/MOV)")
    parser.add_argument("--pro", required=True, help='Pro player name (e.g. "Roger Federer")')
    parser.add_argument(
        "--shot-type",
        required=True,
        choices=["forehand", "backhand", "serve", "volley", "slice"],
    )
    parser.add_argument("--fps", type=int, default=30, help="Sample FPS (default 30)")
    parser.add_argument(
        "--output-json",
        help="Optionally save keypoints to a JSON file instead of uploading to Supabase",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Extracting keypoints from: {video_path}")
    print(f"Pro: {args.pro}, Shot: {args.shot_type}, FPS: {args.fps}")
    print()

    keypoints_json = extract_keypoints_from_video(video_path, sample_fps=args.fps)
    print(f"Extracted {keypoints_json['frame_count']} frames ({keypoints_json.get('duration_ms', 0)}ms)")

    if keypoints_json["frame_count"] == 0:
        print("Warning: No poses detected. Ensure the player's full body is visible.")
        sys.exit(1)

    # Optionally save to file
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(keypoints_json, f, indent=2)
        print(f"Saved keypoints to: {args.output_json}")

    # Upload to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("\nSUPABASE_URL and SUPABASE_SERVICE_KEY not set.")
        if not args.output_json:
            print("Use --output-json to save locally instead.")
        sys.exit(0)

    supabase = create_client(url, key)

    # Find or create the pro
    existing = supabase.table("pros").select("id").eq("name", args.pro).execute()
    if existing.data:
        pro_id = existing.data[0]["id"]
        print(f"Found existing pro: {args.pro} (id={pro_id})")
    else:
        result = supabase.table("pros").insert({"name": args.pro}).execute()
        pro_id = result.data[0]["id"]
        print(f"Created pro: {args.pro} (id={pro_id})")

    # Check if swing already exists
    existing_swing = (
        supabase.table("pro_swings")
        .select("id")
        .eq("pro_id", pro_id)
        .eq("shot_type", args.shot_type)
        .execute()
    )

    if existing_swing.data:
        swing_id = existing_swing.data[0]["id"]
        print(f"Updating existing {args.shot_type} swing (id={swing_id})...")
        supabase.table("pro_swings").update({
            "keypoints_json": keypoints_json,
            "fps": keypoints_json["fps_sampled"],
            "frame_count": keypoints_json["frame_count"],
            "duration_ms": keypoints_json.get("duration_ms"),
        }).eq("id", swing_id).execute()
    else:
        print(f"Inserting new {args.shot_type} swing...")
        supabase.table("pro_swings").insert({
            "pro_id": pro_id,
            "shot_type": args.shot_type,
            "video_url": f"local://{video_path}",
            "keypoints_json": keypoints_json,
            "fps": keypoints_json["fps_sampled"],
            "frame_count": keypoints_json["frame_count"],
            "duration_ms": keypoints_json.get("duration_ms"),
            "metadata": {"source": "local", "original_path": video_path},
        }).execute()

    print("Done! Pro swing data is now in the database.")


if __name__ == "__main__":
    main()
