"""Extract keypoints from a video clip and print JSON to stdout."""
import os
import sys
import json
from pathlib import Path

# Set dummy env vars so main.py can import without Supabase
os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy-key")

sys.path.insert(0, str(Path(__file__).parent))

# Patch supabase.create_client to avoid validation error
import unittest.mock
with unittest.mock.patch('supabase.create_client', return_value=None):
    from main import extract_keypoints_from_video


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: extract_clip_keypoints.py <video_path>"}))
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"File not found: {video_path}"}))
        sys.exit(1)

    result = extract_keypoints_from_video(video_path, sample_fps=30, max_seconds=0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
