"""
Pro Player Database Seeder
Downloads reference clips, extracts pose keypoints, and inserts into Supabase.

Usage:
  pip install -r requirements.txt
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python seed_pros.py

Pro clips are sourced from publicly available coaching/analysis videos.
Only keypoints (not full video) are stored in the database.

To add a new pro or swing:
  1. Find a slow-motion YouTube clip (2-10 seconds of a single swing is ideal)
  2. Add the youtube_url to the appropriate pro's swings list below
  3. Run this script
"""

import os
import re
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path
from supabase import create_client

# Import local extraction logic
sys.path.insert(0, str(Path(__file__).parent))
from main import extract_keypoints_from_video

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Directory for saving pro video files alongside the Next.js public assets
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRO_VIDEOS_DIR = PROJECT_ROOT / "public" / "pro-videos"


def sanitize_filename(name: str) -> str:
    """Convert a string into a safe filename component (lowercase, underscores)."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

# ---------------------------------------------------------------------------
# Pro player definitions with YouTube URLs
#
# camera_angle values: "side", "behind", "front", "court_level"
# Compilations are marked; the seeder processes the full video and
# detectSwings() on the client separates individual shots.
# ---------------------------------------------------------------------------

PROS_DATA = [
    # ------------------------------------------------------------------
    # ROGER FEDERER
    # ------------------------------------------------------------------
    {
        "name": "Roger Federer",
        "nationality": "Switzerland",
        "ranking": 1,
        "bio": "20x Grand Slam champion. Effortless one-handed backhand, precise footwork, and fluid serve motion.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=stEhSvoou4g",
                "video_url": "",
                "label": "forehand_side",
                "phase_labels": {"preparation": 0, "loading": 10, "contact": 22, "follow_through": 32, "finish": 42},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "240fps", "event": "Indian Wells practice"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=xNPaZj4yn00",
                "video_url": "",
                "label": "forehand_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "backhand",
                "youtube_url": "https://www.youtube.com/watch?v=-Bteu4ULtaI",
                "video_url": "",
                "label": "backhand_side",
                "phase_labels": {"preparation": 0, "loading": 12, "contact": 26, "follow_through": 36, "finish": 46},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion", "style": "one-handed"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=mKXtVQnqhB4",
                "video_url": "",
                "label": "serve_side",
                "phase_labels": {"preparation": 0, "loading": 15, "contact": 35, "follow_through": 45, "finish": 55},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # RAFAEL NADAL
    # ------------------------------------------------------------------
    {
        "name": "Rafael Nadal",
        "nationality": "Spain",
        "ranking": 2,
        "bio": "22x Grand Slam champion. Extreme topspin forehand, relentless defense, and the greatest clay court player ever.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=f1qdyFC-Yas",
                "video_url": "",
                "label": "forehand_practice",
                "phase_labels": {"preparation": 0, "loading": 12, "contact": 28, "follow_through": 38, "finish": 48},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=2gGNKs_Bnbs",
                "video_url": "",
                "label": "forehand_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "240fps", "event": "ATP practice 2021"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=5USvWVeEog4",
                "video_url": "",
                "label": "forehand_aus_open",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "4K 240fps", "event": "Australian Open 2022"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=jpuBBohh5SQ",
                "video_url": "",
                "label": "serve_aus_open",
                "phase_labels": {"preparation": 0, "loading": 18, "contact": 38, "follow_through": 48, "finish": 58},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "240fps", "event": "Australian Open 2021"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # NOVAK DJOKOVIC
    # ------------------------------------------------------------------
    {
        "name": "Novak Djokovic",
        "nationality": "Serbia",
        "ranking": 3,
        "bio": "24x Grand Slam champion. Elite return of serve, flexible two-handed backhand, and unmatched court coverage.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=xJ5rJJ_pXCo",
                "video_url": "",
                "label": "forehand_behind",
                "phase_labels": {},
                "metadata": {"camera_angle": "behind", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=Jg9KenIzDy4",
                "video_url": "",
                "label": "forehand_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "backhand",
                "youtube_url": "https://www.youtube.com/watch?v=eFmJMwUI07k",
                "video_url": "",
                "label": "backhand_side",
                "phase_labels": {"preparation": 0, "loading": 12, "contact": 26, "follow_through": 36, "finish": 46},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion", "style": "two-handed"},
            },
            {
                "shot_type": "backhand",
                "youtube_url": "https://www.youtube.com/watch?v=GcPACY6gqpM",
                "video_url": "",
                "label": "backhand_behind",
                "phase_labels": {},
                "metadata": {"camera_angle": "behind", "source": "youtube", "quality": "slow_motion", "style": "two-handed"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=32nyPw9YUEA",
                "video_url": "",
                "label": "serve_court_level",
                "phase_labels": {"preparation": 0, "loading": 18, "contact": 38, "follow_through": 48, "finish": 58},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "240fps", "event": "AO 2021"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=0zgQN0sgGmI",
                "video_url": "",
                "label": "serve_behind",
                "phase_labels": {},
                "metadata": {"camera_angle": "behind", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # SERENA WILLIAMS
    # ------------------------------------------------------------------
    {
        "name": "Serena Williams",
        "nationality": "United States",
        "ranking": 4,
        "bio": "23x Grand Slam champion. Most powerful serve in women's tennis history and aggressive baseline game.",
        "swings": [
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=2UFwsISQxyQ",
                "video_url": "",
                "label": "serve_volley_practice",
                "phase_labels": {"preparation": 0, "loading": 20, "contact": 40, "follow_through": 50, "finish": 60},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "240fps", "event": "AO 2021"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=2UFwsISQxyQ",
                "video_url": "",
                "label": "forehand_practice",
                "phase_labels": {"preparation": 0, "loading": 10, "contact": 22, "follow_through": 32, "finish": 42},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "240fps", "event": "AO 2021"},
            },
            {
                "shot_type": "backhand",
                "youtube_url": "https://www.youtube.com/watch?v=XYVMurYMHhw",
                "video_url": "",
                "label": "backhand_super_slow_mo",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "super_slow_motion", "style": "two-handed"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # CARLOS ALCARAZ
    # ------------------------------------------------------------------
    {
        "name": "Carlos Alcaraz",
        "nationality": "Spain",
        "ranking": 5,
        "bio": "4x Grand Slam champion (youngest in Open Era). Explosive forehand, creative shot-making, and elite athleticism.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=TJkOuNnNt8E",
                "video_url": "",
                "label": "forehand_indian_wells_2024",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "4K 120fps", "event": "Indian Wells 2024"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=MFpSeZRx83g",
                "video_url": "",
                "label": "forehand_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=NsYdWHVfQKo",
                "video_url": "",
                "label": "forehand_evolution_2021_2024",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion", "note": "forehand evolution 2021-2024"},
            },
            {
                "shot_type": "backhand",
                "youtube_url": "https://www.youtube.com/watch?v=Rzas-SIDRO0",
                "video_url": "",
                "label": "backhand_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=kTO7o5A0sys",
                "video_url": "",
                "label": "serve_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=603D0f8mLs8",
                "video_url": "",
                "label": "serve_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # JANNIK SINNER
    # ------------------------------------------------------------------
    {
        "name": "Jannik Sinner",
        "nationality": "Italy",
        "ranking": 6,
        "bio": "World #1. Flat, penetrating groundstrokes off both wings, outstanding timing, and improving serve.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=ZU0tVNMI1qo",
                "video_url": "",
                "label": "forehand_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=2lLCIzntsXk",
                "video_url": "",
                "label": "forehand_practice_sonego",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "4K", "note": "practice with Lorenzo Sonego"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=m7VvitGvm40",
                "video_url": "",
                "label": "forehand_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion", "note": "vs Cameron Norrie practice"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=ZU0tVNMI1qo",
                "video_url": "",
                "label": "serve_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=SPt6XyrTyuQ",
                "video_url": "",
                "label": "serve_iw_practice",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion", "event": "Indian Wells 2023"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # IGA SWIATEK
    # ------------------------------------------------------------------
    {
        "name": "Iga Swiatek",
        "nationality": "Poland",
        "ranking": 7,
        "bio": "5x Grand Slam champion. Extreme western grip forehand, heavy topspin, and dominant clay court game.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=muQxdqOqhxI",
                "video_url": "",
                "label": "forehand_court_level_4k",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "4K 60fps"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=hmcrCasO7V4",
                "video_url": "",
                "label": "forehand_slow_motion",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=c_XAVNpco0s",
                "video_url": "",
                "label": "forehand_ao_2021",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "240fps", "event": "Australian Open 2021"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=muQxdqOqhxI",
                "video_url": "",
                "label": "serve_4k",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "4K 120fps"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # NAOMI OSAKA
    # ------------------------------------------------------------------
    {
        "name": "Naomi Osaka",
        "nationality": "Japan",
        "ranking": 8,
        "bio": "4x Grand Slam champion. Powerful flat serve, aggressive forehand, and strong mental game at biggest moments.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=QWkoYFXOSDc",
                "video_url": "",
                "label": "forehand_practice",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=QWkoYFXOSDc",
                "video_url": "",
                "label": "serve_practice",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # DANIIL MEDVEDEV
    # ------------------------------------------------------------------
    {
        "name": "Daniil Medvedev",
        "nationality": "Russia",
        "ranking": 9,
        "bio": "1x Grand Slam champion. Unorthodox flat groundstrokes, elite defensive skills, and strong serve.",
        "swings": [
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=IbLqTVTZO3s",
                "video_url": "",
                "label": "forehand_side",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=vf8FbH8_tCo",
                "video_url": "",
                "label": "serve_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion"},
            },
            {
                "shot_type": "forehand",
                "youtube_url": "https://www.youtube.com/watch?v=vf8FbH8_tCo",
                "video_url": "",
                "label": "forehand_court_level",
                "phase_labels": {},
                "metadata": {"camera_angle": "court_level", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },

    # ------------------------------------------------------------------
    # ANDY RODDICK
    # ------------------------------------------------------------------
    {
        "name": "Andy Roddick",
        "nationality": "United States",
        "ranking": 10,
        "bio": "1x Grand Slam champion. One of the fastest serves in tennis history, peaking at 155 mph.",
        "swings": [
            {
                "shot_type": "serve",
                "youtube_url": "https://www.youtube.com/watch?v=rTXgMbyjMuU",
                "video_url": "",
                "label": "serve_slow_motion",
                "phase_labels": {},
                "metadata": {"camera_angle": "side", "source": "youtube", "quality": "slow_motion"},
            },
        ],
    },
]


ALLOWED_YOUTUBE_PREFIXES = (
    "https://www.youtube.com/",
    "https://youtube.com/",
    "https://youtu.be/",
)


def download_youtube(url: str, output_path: str) -> bool:
    """Download a YouTube video to output_path using yt-dlp."""
    if not url.startswith(ALLOWED_YOUTUBE_PREFIXES):
        print(f"  Rejected URL (not a recognized YouTube domain): {url}")
        return False
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
            print(f"  yt-dlp stderr: {result.stderr[:200]}")
        return result.returncode == 0
    except Exception as e:
        print(f"  yt-dlp failed: {e}")
        return False


def trim_video(input_path: str, output_path: str, start: float, end: float) -> bool:
    """Trim a video to [start, end] seconds using ffmpeg with re-encoding for precision."""
    duration = end - start
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.2f}",
                "-i", input_path,
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  ffmpeg trim stderr: {result.stderr[:200]}")
        return result.returncode == 0
    except Exception as e:
        print(f"  ffmpeg trim failed: {e}")
        return False



def seed_pro(pro_data: dict):
    print(f"\n{'='*60}")
    print(f"Seeding: {pro_data['name']}")
    print(f"{'='*60}")

    # Upsert pro record
    existing = supabase.table("pros").select("id").eq("name", pro_data["name"]).execute()
    if existing.data:
        pro_id = existing.data[0]["id"]
        print(f"  Pro exists, id={pro_id}")
        # Update bio/nationality if changed
        supabase.table("pros").update({
            "nationality": pro_data.get("nationality"),
            "ranking": pro_data.get("ranking"),
            "bio": pro_data.get("bio"),
        }).eq("id", pro_id).execute()
    else:
        result = supabase.table("pros").insert({
            "name": pro_data["name"],
            "nationality": pro_data.get("nationality"),
            "ranking": pro_data.get("ranking"),
            "bio": pro_data.get("bio"),
        }).execute()
        pro_id = result.data[0]["id"]
        print(f"  Created pro, id={pro_id}")

    for swing in pro_data["swings"]:
        shot_type = swing["shot_type"]
        label = swing.get("label", shot_type)
        print(f"\n  Processing: {shot_type} ({label})")

        # Dedup by (pro_id, shot_type, label) to allow multiple angles
        existing_swings = (
            supabase.table("pro_swings")
            .select("id, metadata")
            .eq("pro_id", pro_id)
            .eq("shot_type", shot_type)
            .execute()
        )

        # Check if this specific label already exists
        already_exists = False
        for es in existing_swings.data:
            existing_label = (es.get("metadata") or {}).get("label", "")
            if existing_label == label:
                already_exists = True
                print(f"    Already exists with label '{label}', skipping.")
                break

        if already_exists:
            continue

        video_url = swing.get("video_url", "")
        youtube_url = swing.get("youtube_url", "")

        if not video_url and not youtube_url:
            print(f"    No video URL provided, inserting placeholder.")
            metadata = swing.get("metadata", {})
            metadata["label"] = label
            supabase.table("pro_swings").insert({
                "pro_id": pro_id,
                "shot_type": shot_type,
                "video_url": "https://placeholder.example.com/video.mp4",
                "keypoints_json": {"fps_sampled": 30, "frame_count": 0, "frames": []},
                "phase_labels": swing.get("phase_labels", {}),
                "metadata": metadata,
            }).execute()
            continue

        # Download video if YouTube URL provided
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            # Remove the empty file so yt-dlp can create it fresh
            os.unlink(tmp_path)

            if youtube_url:
                print(f"    Downloading from YouTube...")
                success = download_youtube(youtube_url, tmp_path)
                if not success:
                    print(f"    Download failed, skipping.")
                    continue
                # Trim to start_time/end_time if specified
                start_time = swing.get("start_time")
                end_time = swing.get("end_time")
                if start_time is not None and end_time is not None:
                    trimmed_path = tmp_path + ".trimmed.mp4"
                    print(f"    Trimming to {start_time:.1f}s - {end_time:.1f}s...")
                    if trim_video(tmp_path, trimmed_path, start_time, end_time):
                        os.unlink(tmp_path)
                        os.rename(trimmed_path, tmp_path)
                    else:
                        print(f"    Trim failed, using full video.")
                        if os.path.exists(trimmed_path):
                            os.unlink(trimmed_path)
            else:
                # Direct MP4 URL
                import httpx
                print(f"    Downloading from URL...")
                with httpx.Client(timeout=120) as client:
                    r = client.get(video_url)
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)

            # If the clip was trimmed, extract all frames. Otherwise cap at max_seconds.
            clip_max = swing.get("max_seconds", 8)
            if swing.get("start_time") is not None and swing.get("end_time") is not None:
                clip_max = 0  # 0 = process entire trimmed video
            print(f"    Extracting pose keypoints (max {clip_max}s)...")
            keypoints_json = extract_keypoints_from_video(tmp_path, sample_fps=30, max_seconds=clip_max)
            print(f"    Extracted {keypoints_json['frame_count']} frames ({keypoints_json.get('duration_ms', 0)}ms)")

            if keypoints_json["frame_count"] == 0:
                print(f"    WARNING: No poses detected, skipping.")
                continue

            metadata = swing.get("metadata", {})
            metadata["label"] = label
            metadata["original_url"] = youtube_url or video_url

            # Save video to public/pro-videos/ for local serving
            sanitized_name = (
                f"{sanitize_filename(pro_data['name'])}"
                f"_{sanitize_filename(shot_type)}"
                f"_{sanitize_filename(label)}"
            )
            video_filename = f"{sanitized_name}.mp4"
            PRO_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            dest_path = PRO_VIDEOS_DIR / video_filename
            shutil.copy2(tmp_path, dest_path)
            local_video_url = f"/pro-videos/{video_filename}"
            print(f"    Saved video to {dest_path}")

            supabase.table("pro_swings").insert({
                "pro_id": pro_id,
                "shot_type": shot_type,
                "video_url": local_video_url,
                "keypoints_json": keypoints_json,
                "fps": keypoints_json["fps_sampled"],
                "frame_count": keypoints_json["frame_count"],
                "duration_ms": keypoints_json.get("duration_ms"),
                "phase_labels": swing.get("phase_labels", {}),
                "metadata": metadata,
            }).execute()

            print(f"    Inserted swing into database.")

        except Exception as e:
            print(f"    ERROR: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def main():
    print("Swingframe Pro Database Seeder")
    print("=" * 60)
    print(f"Total pros: {len(PROS_DATA)}")
    print(f"Total swings: {sum(len(p['swings']) for p in PROS_DATA)}")
    print()

    for pro_data in PROS_DATA:
        try:
            seed_pro(pro_data)
        except Exception as e:
            print(f"  ERROR seeding {pro_data['name']}: {e}")

    print(f"\n{'='*60}")
    print("Seeding complete.")
    print(f"{'='*60}")

    # Print summary
    pros = supabase.table("pros").select("name").execute()
    swings = supabase.table("pro_swings").select("shot_type, metadata").execute()
    print(f"\nDatabase totals: {len(pros.data)} pros, {len(swings.data)} swings")

    # Breakdown by pro
    pros_with_counts = supabase.table("pros").select("id, name").execute()
    swings_with_pro = supabase.table("pro_swings").select("pro_id").execute()
    for pro in pros_with_counts.data:
        count = sum(1 for s in swings_with_pro.data if s["pro_id"] == pro["id"])
        print(f"  {pro['name']}: {count} swings")


if __name__ == "__main__":
    main()
