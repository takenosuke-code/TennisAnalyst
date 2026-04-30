"""
Tennis Analyst Railway Microservice
FastAPI service for server-side pose extraction from videos.
Used for processing pro player videos offline and seeding the database.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
import re
import subprocess
import tempfile
import uuid
from urllib.parse import urlparse
import cv2
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from supabase import create_client, Client
import httpx

# Re-export the extraction entrypoint + helpers used by other modules
# in this service (seeders, youtube pipeline, classify-angle endpoint).
# `extraction.py` is the lean import surface — it does NOT pull in
# supabase / fastapi / mediapipe at module load.
from extraction import (  # noqa: F401  (re-exported for callers)
    extract_keypoints_from_video,
    _open_video_autorotated,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Eagerly initialize pose-extraction models on container boot so
    the first user upload doesn't pay the lazy-load tax (~2-3s for
    onnxruntime + RTMPose, plus ~1-2s if rtmlib needs to download the
    ONNX bundle from the openmmlab CDN — and that download happens on
    every cold start since Railway containers are ephemeral and
    ~/.cache/rtmlib isn't a persisted volume).

    The Dockerfile also pre-downloads RTMPose at build time so the
    cached ONNX is baked into the image and rtmlib's download_checkpoint
    is a no-op here. This warmup hook then just builds the ORT session.

    Failures are logged but not raised — the lazy-load path is still
    in place per-frame as a fallback. Don't crash the whole service
    over a transient warmup failure.
    """
    try:
        from pose_rtmpose import _ensure_yolo, _ensure_rtmpose
        await asyncio.to_thread(_ensure_yolo)
        await asyncio.to_thread(_ensure_rtmpose)
        print("[startup] pose models warmed (YOLO + RTMPose)")
    except Exception as e:  # noqa: BLE001
        print(f"[startup] pose-model warmup failed (will lazy-load): {e}")
    yield


app = FastAPI(title="Tennis Analyst Pose Service", lifespan=_lifespan)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EXTRACT_API_KEY = os.environ.get("EXTRACT_API_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Only allow fetching videos from these domains
ALLOWED_VIDEO_DOMAINS = {
    "blob.vercel-storage.com",
    "public.blob.vercel-storage.com",
    "supabase.co",
    "supabase.com",
    "storage.googleapis.com",
}


def _is_url_allowed(url: str) -> bool:
    """Validate that a video URL uses https and points to an allowed domain."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = parsed.hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_VIDEO_DOMAINS)


class ExtractRequest(BaseModel):
    video_url: str
    session_id: str | None = None
    pro_swing_id: str | None = None
    callback_url: str | None = None


@app.post("/extract")
async def extract_pose(
    req: ExtractRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(default=""),
):
    """
    Trigger pose extraction for a video URL.
    Runs in background and updates Supabase when done.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    if req.callback_url and not _is_url_allowed(req.callback_url):
        raise HTTPException(
            status_code=400,
            detail="callback_url must use https and point to an allowed domain",
        )

    background_tasks.add_task(_run_extraction, req)
    return {"status": "queued", "session_id": req.session_id}


MODAL_INFERENCE_URL = os.environ.get("MODAL_INFERENCE_URL", "").strip()


async def _extract_via_modal(video_url: str, sample_fps: int = 30) -> dict:
    """Forward the video URL to a Modal GPU endpoint and return its
    keypoints_json response. Same shape as extract_keypoints_from_video
    so the rest of _run_extraction is agnostic to which path ran.

    No shared bearer token: the Modal endpoint enforces an HTTPS+domain
    allowlist on the video_url field instead. The Modal URL itself is
    server-to-server only (Railway → Modal), never sent to the browser.

    Raises on transport failure or non-2xx so the outer try/except in
    _run_extraction can mark the session as errored just like for the
    inline path.
    """
    # Bumped to 600s to match Modal's @app.function(timeout=600). See
    # Issue 2 in the planning doc — 2-min user clips on cold Modal
    # containers can run 90-120s, occasionally longer. Headroom keeps
    # the legitimate-but-slow path from getting clipped.
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            MODAL_INFERENCE_URL,
            json={"video_url": video_url, "sample_fps": sample_fps},
        )
        resp.raise_for_status()
        return resp.json()


async def _run_extraction(req: ExtractRequest):
    """Run pose extraction (Modal or local) and persist the result.

    Stage timing is logged via `[extract] timing=...` so future timeouts
    can be diagnosed from Railway logs without instrumentation hacks.
    Fields:
      modal_call_ms     — round-trip time of the Modal HTTP call (None if local)
      download_ms       — Railway-side video download time (None if Modal owned it)
      inference_ms      — extract_keypoints_from_video wall time on the local path
      supabase_write_ms — DB update time
      total_ms          — end-to-end wall time of _run_extraction
    Modal-side stage timings live under `keypoints_json['timing']`.
    """
    timing: dict = {
        "modal_call_ms": None,
        "download_ms": None,
        "inference_ms": None,
        "supabase_write_ms": None,
        "total_ms": None,
    }
    t_start = time.perf_counter()
    try:
        # Modal-proxy fast path: when MODAL_INFERENCE_URL is configured,
        # forward the blob URL to the Modal GPU endpoint instead of
        # running ONNX locally on the Railway CPU. Modal downloads the
        # blob itself (so we skip the Railway-side download) and runs
        # the same extract_keypoints_from_video pipeline on T4. Falls
        # through to local extraction on any Modal failure so a Modal
        # outage doesn't break uploads.
        if MODAL_INFERENCE_URL:
            try:
                t_modal = time.perf_counter()
                keypoints_json = await _extract_via_modal(req.video_url)
                timing["modal_call_ms"] = round(
                    (time.perf_counter() - t_modal) * 1000, 1
                )
                tmp_path = None  # skipped; Modal owns the temp file
            except Exception as modal_err:  # noqa: BLE001
                print(f"[extract] Modal path failed, falling back to local: {modal_err}")
                keypoints_json = None
                tmp_path = None
        else:
            keypoints_json = None
            tmp_path = None

        if keypoints_json is None:
            # Local fallback (or default when MODAL_* env not set):
            # download the video to Railway, run extraction inline.
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                t_dl = time.perf_counter()
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.get(req.video_url)
                    response.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        f.write(response.content)
                timing["download_ms"] = round(
                    (time.perf_counter() - t_dl) * 1000, 1
                )

                # Run blocking CPU/IO work in a thread pool so the event
                # loop stays free.
                t_inf = time.perf_counter()
                keypoints_json = await asyncio.to_thread(
                    extract_keypoints_from_video, tmp_path
                )
                timing["inference_ms"] = round(
                    (time.perf_counter() - t_inf) * 1000, 1
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Update the appropriate DB record
        t_db = time.perf_counter()
        if req.pro_swing_id:
            supabase.table("pro_swings").update({
                "keypoints_json": keypoints_json,
                "frame_count": keypoints_json["frame_count"],
                "duration_ms": keypoints_json["duration_ms"],
                "fps": keypoints_json["fps_sampled"],
            }).eq("id", req.pro_swing_id).execute()

        elif req.session_id:
            supabase.table("user_sessions").update({
                "keypoints_json": keypoints_json,
                "status": "complete",
            }).eq("id", req.session_id).execute()
        timing["supabase_write_ms"] = round(
            (time.perf_counter() - t_db) * 1000, 1
        )

        # Optional callback — only to allowed domains
        if req.callback_url and _is_url_allowed(req.callback_url):
            async with httpx.AsyncClient() as client:
                await client.post(req.callback_url, json={"status": "complete"})

    except Exception as e:
        print(f"Extraction failed: {e}")
        if req.session_id:
            supabase.table("user_sessions").update({
                "status": "error",
                "error_message": str(e),
            }).eq("id", req.session_id).execute()
    finally:
        timing["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        # Surface Modal's own per-stage timing if present in the
        # response, so a single log line tells the full story.
        modal_timing = None
        try:
            if isinstance(keypoints_json, dict):  # type: ignore[name-defined]
                modal_timing = keypoints_json.get("timing")
        except NameError:
            pass
        if modal_timing:
            timing["modal_stage"] = modal_timing
        print(f"[extract] timing={timing}")


# ---------------------------------------------------------------------------
# YouTube processing
# ---------------------------------------------------------------------------

# In-memory job status tracker (sufficient for single-instance Railway deploy)
_youtube_jobs: dict[str, dict] = {}

YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com|youtu\.be)/"
)


def _is_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a valid YouTube URL."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


class YouTubeProcessRequest(BaseModel):
    youtube_url: str
    target_shot_types: list[str] | None = None
    max_duration: int = 600


@app.post("/process-youtube")
async def process_youtube(
    req: YouTubeProcessRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(default=""),
):
    """Queue processing of a YouTube video for tennis shot extraction."""
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_youtube_url(req.youtube_url):
        raise HTTPException(
            status_code=400,
            detail="youtube_url must be a valid youtube.com or youtu.be URL",
        )

    job_id = str(uuid.uuid4())
    _youtube_jobs[job_id] = {"status": "processing", "clips": [], "error": None}

    background_tasks.add_task(_run_youtube_processing, job_id, req)
    return {"status": "queued", "job_id": job_id}


@app.get("/process-youtube/{job_id}")
async def get_youtube_job_status(
    job_id: str,
    authorization: str = Header(default=""),
):
    """Check the status of a YouTube processing job."""
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job = _youtube_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response: dict = {"status": job["status"]}
    if job["status"] == "complete":
        response["clips"] = job["clips"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


async def _run_youtube_processing(job_id: str, req: YouTubeProcessRequest):
    """Background task: process a YouTube video and store results."""
    from youtube_processor import process_youtube_video

    try:
        result = await asyncio.to_thread(
            process_youtube_video,
            req.youtube_url,
            req.target_shot_types,
            req.max_duration,
        )

        created_ids: list[dict] = []

        for clip in result.clips:
            # Upload each extracted clip to Vercel Blob via the public upload URL
            clip_path = clip.path
            if not os.path.exists(clip_path):
                continue

            try:
                blob_url = await _upload_clip_to_blob(clip_path, clip.shot_type)

                # Extract keypoints for the clip
                keypoints_json = await asyncio.to_thread(
                    extract_keypoints_from_video, clip_path
                )

                # pro_id is required by the schema, so DB insertion
                # is deferred until the caller associates clips with a
                # pro.  Return clip data so the caller can do that.
                created_ids.append({
                    "blob_url": blob_url,
                    "shot_type": clip.shot_type,
                    "confidence": clip.confidence,
                    "camera_angle": clip.camera_angle,
                    "duration_ms": clip.duration_ms,
                    "handedness": clip.handedness,
                    "keypoints_frame_count": keypoints_json.get("frame_count"),
                    "metadata": {
                        "source": "youtube",
                        "original_url": req.youtube_url,
                        "video_title": result.video_title,
                        "camera_angle": clip.camera_angle,
                        "confidence": clip.confidence,
                        "handedness": clip.handedness,
                        "start_time": clip.start_time,
                        "end_time": clip.end_time,
                    },
                })

            except Exception as e:
                print(f"Failed to process clip {clip_path}: {e}")
            finally:
                if os.path.exists(clip_path):
                    os.unlink(clip_path)

        _youtube_jobs[job_id] = {
            "status": "complete",
            "clips": created_ids,
            "error": None,
        }

    except Exception as e:
        print(f"YouTube processing failed for job {job_id}: {e}")
        _youtube_jobs[job_id] = {
            "status": "error",
            "clips": [],
            "error": str(e),
        }


async def _upload_clip_to_blob(clip_path: str, shot_type: str) -> str:
    """Upload a clip file to Vercel Blob and return the public URL.

    Requires BLOB_READ_WRITE_TOKEN in the environment.
    """
    return await _upload_file_to_blob(clip_path, f"pro-clips/{shot_type}/{uuid.uuid4()}.mp4")


async def _upload_file_to_blob(local_path: str, blob_path: str) -> str:
    """Upload an arbitrary local file to Vercel Blob at *blob_path* and return the public URL.

    Requires BLOB_READ_WRITE_TOKEN in the environment. *blob_path* is treated
    as the pathname inside the store and must not start with '/'.
    """
    blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not blob_token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN not set — cannot upload clips")

    filename = blob_path.lstrip("/")

    with open(local_path, "rb") as f:
        file_bytes = f.read()

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.put(
            f"https://blob.vercel-storage.com/{filename}",
            content=file_bytes,
            headers={
                "Authorization": f"Bearer {blob_token}",
                "x-content-type": "video/mp4",
                "x-cache-control-max-age": "31536000",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    url = data.get("url")
    if not url:
        raise RuntimeError("Blob upload succeeded but no URL in response")
    return url


# ---------------------------------------------------------------------------
# Camera angle classification (telemetry-only)
# ---------------------------------------------------------------------------
#
# Wraps camera_classifier.classify_frame in an HTTP endpoint so Node analyze
# routes can tag each analysis_events row with a capture_quality_flag. The
# classifier exposes five raw labels -- behind / side / front / overhead /
# unknown. There is NO oblique-angle branch in the underlying heuristics, so
# the Node-facing enum only uses the three-bucket schema:
#   side     -> green_side
#   behind   -> red_front_or_back
#   front    -> red_front_or_back
#   overhead -> unknown   (classifier can't see the player clearly enough)
#   unknown  -> unknown
# yellow_oblique exists in the DB enum but we never emit it -- if we ever add
# oblique detection in camera_classifier.py, map it here.


class ClassifyAngleRequest(BaseModel):
    video_url: str


_RAW_TO_TELEMETRY_FLAG = {
    "side": "green_side",
    "behind": "red_front_or_back",
    "front": "red_front_or_back",
    "overhead": "unknown",
    "unknown": "unknown",
}


def _sample_frame_indices(total_frames: int) -> list[int]:
    """Pick ~5 frames spread across the clip: start, 25%, 50%, 75%, end.

    Deduped and clamped so very short clips (fewer than 5 sampleable frames)
    still return something sensible. Returns an empty list if there are no
    frames to sample.
    """
    if total_frames <= 0:
        return []
    if total_frames <= 5:
        return list(range(total_frames))
    markers = [0.0, 0.25, 0.5, 0.75, 1.0]
    indices: list[int] = []
    for m in markers:
        idx = min(total_frames - 1, max(0, int(round(m * (total_frames - 1)))))
        if idx not in indices:
            indices.append(idx)
    return indices


def _aggregate_angle_labels(labels: list[str]) -> str:
    """Pick the majority raw label, or 'unknown' if the vote is split.

    We only count non-'unknown' labels toward majority. If every sample is
    'unknown' (classifier couldn't find court/player), we return 'unknown'.
    A single dominant label needs STRICT majority of the non-unknown votes
    -- ties and near-ties fall through to 'unknown' so mixed-angle clips
    don't get a confident flag they don't deserve.
    """
    if not labels:
        return "unknown"
    considered = [lbl for lbl in labels if lbl and lbl != "unknown"]
    if not considered:
        return "unknown"
    counts: dict[str, int] = {}
    for lbl in considered:
        counts[lbl] = counts.get(lbl, 0) + 1
    top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count * 2 > len(considered):
        return top_label
    return "unknown"


def _classify_video_frames(video_path: str) -> tuple[str, int]:
    """Sample frames, classify each, and aggregate to a raw label.

    Returns (raw_label, samples_considered). raw_label is one of the five
    classifier labels. samples_considered is how many frames we actually ran
    classify_frame on (0 if the video couldn't be opened).
    """
    from camera_classifier import classify_frame  # local import so tests can patch

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        return "unknown", 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = _sample_frame_indices(total)
        labels: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                _, raw = classify_frame(frame)
            except Exception as e:  # noqa: BLE001
                print(f"[classify-angle] classify_frame failed on frame {idx}: {e}")
                continue
            labels.append(raw or "unknown")
        return _aggregate_angle_labels(labels), len(labels)
    finally:
        cap.release()


@app.post("/classify-angle")
async def classify_angle(
    req: ClassifyAngleRequest,
    authorization: str = Header(default=""),
):
    """Classify the dominant camera angle for a clip and return a telemetry flag.

    Always returns 200 with a capture_quality_flag. Any failure path falls
    through to {"capture_quality_flag": "unknown", "error": "..."} so the
    Node caller never has to branch on HTTP errors -- telemetry must never
    block coaching.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(req.video_url)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(response.content)

        raw_label, samples = await asyncio.to_thread(_classify_video_frames, tmp_path)
        flag = _RAW_TO_TELEMETRY_FLAG.get(raw_label, "unknown")
        return {
            "capture_quality_flag": flag,
            "raw_label": raw_label,
            "samples_considered": samples,
        }
    except Exception as e:  # noqa: BLE001
        # Telemetry contract: never surface a non-200 to the Node caller.
        print(f"[classify-angle] failed for {req.video_url}: {e}")
        return {
            "capture_quality_flag": "unknown",
            "raw_label": "unknown",
            "samples_considered": 0,
            "error": str(e),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:  # noqa: BLE001
                print(f"[classify-angle] temp cleanup failed: {e}")


# ---------------------------------------------------------------------------
# Segment trimming for baseline capture
# ---------------------------------------------------------------------------
#
# Trims a source video by [start_ms, end_ms] using ffmpeg stream-copy and
# uploads the result to Vercel Blob. Stream-copy (no re-encode) keeps the
# clip lossless and avoids a multi-second CPU hit per call; ffmpeg will snap
# the cut to the nearest keyframe, which is accurate enough for a baseline
# thumbnail/preview.
#
# Requires ffmpeg on PATH (nixpacks.toml provides it) and BLOB_READ_WRITE_TOKEN.


class TrimVideoRequest(BaseModel):
    video_url: str
    start_ms: int
    end_ms: int


def _trim_with_ffmpeg(src_path: str, dst_path: str, start_ms: int, end_ms: int) -> None:
    """Stream-copy the [start_ms, end_ms] range from src_path to dst_path.

    Raises RuntimeError with the captured stderr if ffmpeg fails.
    """
    start_s = max(0, start_ms) / 1000.0
    duration_s = max(0, end_ms - start_ms) / 1000.0
    if duration_s <= 0:
        raise RuntimeError("end_ms must be greater than start_ms")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        src_path,
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed (exit {result.returncode}): {result.stderr[:2000]}"
        )


@app.post("/trim-video")
async def trim_video(
    req: TrimVideoRequest,
    authorization: str = Header(default=""),
):
    """Download *video_url*, trim to [start_ms, end_ms], upload to Vercel Blob.

    Returns { blob_url }. Uses stream-copy (no re-encode) so it's fast and
    lossless but cuts snap to the nearest keyframe.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    if req.end_ms <= req.start_ms:
        raise HTTPException(status_code=400, detail="end_ms must be greater than start_ms")

    src_path: str | None = None
    dst_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dst:
            dst_path = dst.name

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(req.video_url)
            response.raise_for_status()
            with open(src_path, "wb") as f:
                f.write(response.content)

        await asyncio.to_thread(
            _trim_with_ffmpeg, src_path, dst_path, req.start_ms, req.end_ms
        )

        blob_path = f"baseline-trims/{uuid.uuid4()}.mp4"
        blob_url = await _upload_file_to_blob(dst_path, blob_path)
        return {"blob_url": blob_url}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[trim-video] failed for {req.video_url}: {e}")
        raise HTTPException(status_code=500, detail=f"trim failed: {e}")
    finally:
        for p in (src_path, dst_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError as e:  # noqa: BLE001
                    print(f"[trim-video] temp cleanup failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/racket-detector")
def debug_racket_detector():
    """One-shot diagnostic: is the YOLO racket detector actually wired up on
    this deploy? Returns model-file paths + existence, attempts a session
    load, runs inference on a blank image. Meant for debugging "racket
    detected: 0 / N" situations where the frontend has no visibility into
    server state.
    """
    import numpy as np
    from pathlib import Path

    from racket_detector import (
        MODEL_DIR,
        YOLO_ONNX_PATH,
        YOLO_PT_PATH,
        detect_racket,
    )
    import racket_detector as rd

    def _size_or_none(p: Path):
        try:
            return p.stat().st_size if p.exists() else None
        except OSError:
            return None

    status = {
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.exists(),
        "onnx_path": str(YOLO_ONNX_PATH),
        "onnx_exists": YOLO_ONNX_PATH.exists(),
        "onnx_size_bytes": _size_or_none(YOLO_ONNX_PATH),
        "pt_path": str(YOLO_PT_PATH),
        "pt_exists": YOLO_PT_PATH.exists(),
        "pt_size_bytes": _size_or_none(YOLO_PT_PATH),
        "session_loaded": rd._session is not None,
    }

    try:
        rd._ensure_model()
        status["ensure_model"] = "ok"
        status["session_loaded"] = rd._session is not None
    except Exception as e:  # noqa: BLE001
        status["ensure_model"] = f"FAILED: {type(e).__name__}: {e}"
        return status

    try:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        result = detect_racket(blank)
        status["blank_inference"] = "ok"
        status["blank_detection"] = result  # expected: None
    except Exception as e:  # noqa: BLE001
        status["blank_inference"] = f"FAILED: {type(e).__name__}: {e}"

    return status
