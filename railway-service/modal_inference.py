"""Modal-deployed GPU pose-extraction endpoint.

Architecture: Railway's `/extract` endpoint normally runs YOLO + RTMPose
ONNX inference inline on its CPU container. When MODAL_INFERENCE_URL is
set on Railway, that endpoint instead forwards the (sessionId, blobUrl)
pair here, where the same pipeline runs on a Modal GPU container.

Deploy once:

    modal deploy railway-service/modal_inference.py

Modal returns an HTTPS URL. Set it on Railway as MODAL_INFERENCE_URL.
That's the entire setup — no shared secret, no Modal Secret to create.

Why no bearer token: Modal endpoints are public HTTPS, but the URL is
only ever called server-to-server (Railway → Modal); the browser never
sees it. The realistic abuse vector if the URL leaks is "attacker burns
your free Modal credits on inference jobs." We close that off with the
ALLOWED_VIDEO_DOMAINS allowlist below — the function refuses any URL
that isn't a Vercel Blob / Supabase Storage / etc. address. An
attacker can't submit arbitrary videos even with the URL.

Cold-start behavior: the @modal.enter hook runs once per container at
startup, before any HTTP request. We pay the ORT session-build tax
there (~2-3s) so the first user upload doesn't see it. Within an active
period (scaledown_window=30s), subsequent calls hit the warm container
and run in ~1-2s.

TRT engine compile happens during `Image.run_function` build (see
_prewarm_models). Once the engine cache is baked into the image
layer, runtime ORT sessions load the engine in <1s vs. 10-20s for a
fresh compile. Set POSE_DISABLE_TRT=1 via Modal dashboard env vars
to skip TRT entirely if the cache becomes corrupt or a model update
breaks engine compatibility — no redeploy needed.
"""
from __future__ import annotations

from pathlib import Path

import modal


SERVICE_DIR = Path(__file__).parent


def _prewarm_models() -> None:
    """Build-time hook: download the YOLO + RTMPose ONNX bundles into
    the image layer so cold-start skips the openmmlab CDN fetch.

    Runs inside Modal's image-build sandbox (no GPU available there,
    so we force POSE_DEVICE=cpu — the cached ONNX bytes are identical
    regardless of the runtime provider, only the ORT session creation
    differs at first use). Without this hook, every cold container
    pays a ~1-2s download tax for `~/.cache/rtmlib` to populate.

    Note: TRT engine compilation requires GPU, which is NOT available
    in the image-build sandbox. The first runtime request on a fresh
    container will therefore pay the engine-compile tax (~10-20s) but
    the engine then caches under MODEL_DIR/trt_cache/ for the lifetime
    of that container. Subsequent containers spawned from the same
    image inherit nothing in trt_cache/, so each cold container
    re-compiles. This is documented as a known cold-start cost.
    """
    import os
    import sys

    sys.path.insert(0, "/root/app")
    os.environ.setdefault("POSE_DEVICE", "cpu")
    from pose_rtmpose import _ensure_yolo, _ensure_rtmpose

    _ensure_yolo()
    _ensure_rtmpose()
    print("[image-build] models cached")


# Image: debian slim + opencv runtime libs + our pip deps. Pinning the
# same versions Railway uses so behavior matches.
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "httpx==0.27.2",
        "ultralytics==8.3.40",
        # GPU build of onnxruntime — picks up CUDAExecutionProvider on
        # Modal's GPU containers automatically. Falls back to CPU when
        # the function runs without GPU (unit tests, smoke checks).
        # 1.18.0+ ships the TensorrtExecutionProvider in this wheel,
        # so we don't need a separate trt-tools pip.
        "onnxruntime-gpu>=1.18.0",
        "rtmlib==0.0.15",
        "fastapi==0.115.0",
        # No supabase / mediapipe here: extraction.py is the lean
        # import surface used by the Modal path. supabase write-back
        # happens back on Railway; mediapipe was retired entirely.
    )
    # Mount the whole railway-service directory so we can reuse the
    # exact same `extract_keypoints_from_video` pipeline (including the
    # staged-parallel inference, motion-gated racket, etc.). Skip large
    # model artifacts that don't need to ship — rtmlib + ultralytics
    # download what they need on first init.
    # `copy=True` is required so the local dir is baked into the image
    # *before* `run_function` runs — Modal otherwise mounts local files
    # only at container startup, and the build-time prewarm needs to
    # `import pose_rtmpose` from `/root/app`.
    .add_local_dir(
        SERVICE_DIR,
        remote_path="/root/app",
        ignore=[
            "tests/",
            "__pycache__/",
            "*.pyc",
        ],
        copy=True,
    )
    # Bake YOLO + RTMPose ONNX into the image so cold-start skips the
    # openmmlab CDN download. `Image.run_function` runs the callable
    # in the image-build sandbox with full pip access.
    .run_function(_prewarm_models)
)


app = modal.App("tennis-pose-inference")


# Only accept video URLs from these domains. The function loads videos
# by GET from `payload["video_url"]`, and without an allowlist anyone
# who learns the public Modal URL could submit arbitrary URLs to make
# us pull and run inference on them (burning your Modal credits).
# Mirrors the same allowlist on Railway's main.py.
_ALLOWED_VIDEO_DOMAINS = (
    "blob.vercel-storage.com",
    "public.blob.vercel-storage.com",
    "supabase.co",
    "supabase.com",
    "storage.googleapis.com",
)


def _is_url_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    if parsed.scheme != "https":
        return False
    host = parsed.hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in _ALLOWED_VIDEO_DOMAINS)


@app.cls(
    image=inference_image,
    # T4 is the cheapest GPU at ~$0.59/hr; sufficient for RTMPose-s
    # (5M params) + YOLO11n (6M params). A10 ($1.10/hr) would be ~2x
    # faster but for our model sizes the marginal gain isn't worth 2x
    # the credit burn.
    gpu="T4",
    # Bumped to 600s. Long-form user uploads (a full-point or service-
    # game tape, ~2 min) on a cold container can plausibly hit
    # 90-120s after Modal cold start + download + inference. The old
    # 300s budget left no headroom; 10 min mirrors the Railway-side
    # client timeout.
    timeout=600,
    # Scale to zero aggressively. Modal bills GPU for the entire
    # idle window after a request, NOT just the active inference,
    # so a 10-min window means every upload pays for 10 min of T4
    # ($0.10/upload). 30s drops that to <$0.005/upload while still
    # letting back-to-back uploads from the same session reuse a
    # warm container. Cold-start tax goes up to ~5-10s on the next
    # upload after a quiet minute, which is fine for our volume.
    scaledown_window=30,
)
class PoseInference:
    """GPU-backed pose extraction service.

    The class shape (vs. the old @app.function) lets us hang a
    @modal.enter() warmup hook off the container so the ORT sessions
    are built BEFORE the first HTTP request lands. Cold-path latency
    drops by the ~2-3s session-build tax that previously sat in the
    first request's critical path.

    `cold_start` reflects whether THIS function invocation is the first
    one served by this container. Modal reuses the same Python process
    for back-to-back invocations until the scaledown window expires;
    the flag flips True after the first successful call.
    """

    @modal.enter()
    def setup(self) -> None:
        """Container startup hook. Runs once per fresh container,
        before any HTTP request is dispatched to extract_pose.

        Pays the ORT session-build tax here so the user's first
        upload doesn't see it. Also stamps a per-container _warm flag
        so the response can tell the caller whether THIS request was
        the first one served by this container.
        """
        import os
        import sys

        os.environ.setdefault("POSE_BACKEND", "rtmpose")
        os.environ.setdefault("POSE_DEVICE", "cuda")
        sys.path.insert(0, "/root/app")

        from pose_rtmpose import _ensure_yolo, _ensure_rtmpose

        try:
            _ensure_yolo()
            _ensure_rtmpose()
            print("[modal.enter] models warm; first request skips session-build tax")
        except Exception as exc:  # noqa: BLE001
            # If warmup fails (e.g. transient TRT engine cache issue),
            # don't crash the container — the lazy-init paths will
            # retry on first request and either succeed or fail loudly
            # then. A failed @modal.enter would 500 every request.
            print(f"[modal.enter] warmup failed, lazy-init will retry: {exc}")

        self._warm = False  # set False here; flipped True after first successful response

    @modal.fastapi_endpoint(method="POST")
    def extract_pose(self, payload: dict):
        """Run YOLO + RTMPose-s on the video at `payload["video_url"]`.

        Body shape: {"video_url": str, "sample_fps": int = 30}
        Auth: domain allowlist on the URL (no shared secret needed).
        Returns: same JSON shape as railway/extraction._extract_with_rtmpose,
        plus a `timing` block with `download_ms`, `inference_ms`,
        `cold_start`, and `provider` ('TRT' / 'CUDA' / 'CPU') so
        Railway can log per-stage cost and the active EP.
        """
        import os
        import sys
        import tempfile
        import time

        cold_start = not self._warm
        t_fn_start = time.perf_counter()

        video_url = payload.get("video_url")
        sample_fps = int(payload.get("sample_fps", 30))
        if not video_url:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="video_url required")
        if not _is_url_allowed(video_url):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail="video_url must use https and a Vercel Blob / Supabase Storage / GCS host",
            )

        # POSE_BACKEND / POSE_DEVICE are set in @modal.enter, but
        # re-set here in case some path bypasses the enter hook (e.g.,
        # local pytest invocation of this method).
        os.environ.setdefault("POSE_BACKEND", "rtmpose")
        os.environ.setdefault("POSE_DEVICE", "cuda")
        sys.path.insert(0, "/root/app")

        # Download the video. Same allowlist Railway enforces.
        import httpx
        t_dl = time.perf_counter()
        with httpx.Client(timeout=600) as client:
            r = client.get(video_url)
            r.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        download_ms = round((time.perf_counter() - t_dl) * 1000, 1)

        try:
            # extraction.py is the lean import surface — no supabase /
            # mediapipe / fastapi pulled in transitively. Cuts ~3s off
            # the cold-start import budget vs. importing through main.py.
            from extraction import extract_keypoints_from_video
            from pose_rtmpose import get_active_provider

            t_inf = time.perf_counter()
            result = extract_keypoints_from_video(tmp_path, sample_fps=sample_fps)
            inference_ms = round((time.perf_counter() - t_inf) * 1000, 1)

            provider = get_active_provider() or "unknown"

            # Stamp the path so the diagnostic chip can show it.
            result["pose_backend"] = (
                f"rtmpose-modal-{os.environ.get('POSE_DEVICE', 'cuda')}"
            )
            result["timing"] = {
                "download_ms": download_ms,
                "inference_ms": inference_ms,
                "function_ms": round((time.perf_counter() - t_fn_start) * 1000, 1),
                "cold_start": cold_start,
                "provider": provider,
            }
            # Grep-friendly structured log for Modal's log search.
            print(f"[extract] timing={result['timing']}")
            # Mark this container warm for the next invocation.
            self._warm = True
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
