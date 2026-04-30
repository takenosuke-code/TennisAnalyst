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
only ever called server-to-server (Railway -> Modal); the browser never
sees it. The realistic abuse vector if the URL leaks is "attacker burns
your free Modal credits on inference jobs." We close that off with the
ALLOWED_VIDEO_DOMAINS allowlist below — the function refuses any URL
that isn't a Vercel Blob / Supabase Storage / etc. address. An
attacker can't submit arbitrary videos even with the URL.

Cold-start behavior: Modal scales to zero by default. First request
after >10 min idle pays ~5-10s container spin + model load. Within an
active period, subsequent calls hit the warm container and run in ~2s.

CRITICAL: this image is now based on nvidia/cuda:12.4.1-cudnn-runtime
(NOT debian_slim). The prior `debian_slim` base shipped only the CUDA
*driver* stub, not the *runtime* or cuDNN — `onnxruntime-gpu` 1.19+
silently fell back to CPU when it couldn't `dlopen("libcudnn.so.9")`,
which is why the Modal logs showed
`Available providers: 'AzureExecutionProvider, CPUExecutionProvider'`
on every request. The base swap fixes that. The build-time provider
assertion in _prewarm_models guards against a future regression.

Memory snapshots (Modal docs: https://modal.com/docs/guide/memory-snapshots)
require `@app.cls` + `@modal.enter(snap=True)`, which would change the
endpoint URL pattern from `…-extract-pose.modal.run` to
`…-poseinference-extract-pose.modal.run`. Last week's revert showed
the URL rotation silently breaks Railway until MODAL_INFERENCE_URL is
re-pointed. Skipping memory snapshots for now — revisit as a deliberate
URL-rotation deploy with a Railway env-var update lined up at the same
time.
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

    Provider-availability assertion: after the YOLO session builds, we
    set POSE_DEVICE=cuda and rebuild the YOLO session in 'GPU mode',
    then verify CUDAExecutionProvider is in the registered providers
    list. The build sandbox itself is GPU-less, so 'registered' here
    means 'CUDA + cuDNN libraries are present and dlopen-able' — the
    runtime kernel launch still has to succeed at request time. But
    'libs present' is the bar that actually broke last time, and
    failing the build loudly when libs are missing is the signal we
    didn't have before.
    """
    import os
    import sys

    sys.path.insert(0, "/root/app")
    os.environ["POSE_DEVICE"] = "cpu"

    # CPU pass: download weights + materialize sessions in CPU mode
    # so the layer cache is populated.
    import pose_rtmpose
    pose_rtmpose._reset_for_tests()
    pose_rtmpose._ensure_yolo()
    pose_rtmpose._ensure_rtmpose()
    print("[image-build] CPU sessions built; weights cached")

    # GPU pass: re-run init in GPU mode and assert CUDA EP is
    # registered. _gpu_enabled() reads POSE_DEVICE on every call now
    # (was a module-level constant before — see pose_rtmpose.py for
    # the regression history).
    os.environ["POSE_DEVICE"] = "cuda"
    pose_rtmpose._reset_for_tests()
    pose_rtmpose._ensure_yolo()
    pose_rtmpose._ensure_rtmpose()

    yolo_providers = pose_rtmpose._yolo_session.get_providers()  # type: ignore[union-attr]
    rtm_providers = pose_rtmpose._rtm_session.session.get_providers()  # type: ignore[union-attr]
    print(f"[image-build] YOLO providers (GPU mode): {yolo_providers}")
    print(f"[image-build] RTMPose providers (GPU mode): {rtm_providers}")

    if "CUDAExecutionProvider" not in yolo_providers:
        raise RuntimeError(
            "CUDAExecutionProvider not registered for YOLO. "
            f"providers={yolo_providers}. cuDNN/CUDA toolkit missing in image."
        )
    if "CUDAExecutionProvider" not in rtm_providers:
        raise RuntimeError(
            "CUDAExecutionProvider not registered for RTMPose. "
            f"providers={rtm_providers}. cuDNN/CUDA toolkit missing in image."
        )

    print("[image-build] CUDA EP available; image build OK")


# Image: NVIDIA CUDA 12.4 runtime + cuDNN 9 + python 3.12. Contains the
# CUDA toolkit + cuDNN that onnxruntime-gpu 1.19+ requires to register
# CUDAExecutionProvider — that's the bit `debian_slim` was missing.
inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "httpx==0.27.2",
        "ultralytics==8.3.40",
        # GPU build of onnxruntime — requires CUDA 12 + cuDNN 9 in the
        # image, which the base image now provides.
        "onnxruntime-gpu>=1.19.0",
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
    # in the image-build sandbox with full pip access. The assertion
    # inside _prewarm_models fails the build if CUDA EP isn't registered.
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


# Per-clip cache. Keyed by a hash of the first 256KB of the downloaded
# blob (URLs change per signed-token issuance even when content is the
# same; the prefix-hash is stable). `modal.Dict` is a globally-shared
# key-value store across containers. Hit rate is near-zero on first
# user upload but helps the retry-loop UX: extractPoseViaRailway polls
# every 1s; if the user double-taps Save the duplicate request finds
# the cached result and skips inference entirely.
clip_cache = modal.Dict.from_name("tennis-pose-clip-cache", create_if_missing=True)


# Container-local warm flag. Modal reuses the same Python process for
# back-to-back invocations until the scaledown window expires; this
# flag flips True after the first successful call inside a container,
# so we can stamp `cold_start=True` on the very first request a
# container ever serves and `cold_start=False` on every subsequent one.
# A fresh container starts with the module re-imported and the flag at
# False again, which is exactly what we want.
_warm = False


@app.function(
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
@modal.fastapi_endpoint(method="POST")
def extract_pose(payload: dict):
    """Run YOLO + RTMPose-s on the video at `payload["video_url"]`.

    Body shape: {"video_url": str, "sample_fps": int = 30}
    Auth: domain allowlist on the URL (no shared secret needed).
    Returns: same JSON shape as railway/extraction._extract_with_rtmpose,
    plus a `timing` block with `download_ms`, `inference_ms`, and
    `cold_start` so Railway can log per-stage cost.
    """
    import hashlib
    import os
    import sys
    import tempfile
    import time

    global _warm
    cold_start = not _warm
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

    os.environ["POSE_BACKEND"] = "rtmpose"
    # Switch the inference path from CPU to CUDA. Read by pose_rtmpose
    # to swap onnxruntime providers + rtmlib's `device=` arg. Use plain
    # assignment, not setdefault: the build-sandbox prewarm sets these
    # to CPU and we want to override that here.
    os.environ["POSE_DEVICE"] = "cuda"

    sys.path.insert(0, "/root/app")

    # Download the video. Same allowlist Railway enforces.
    import httpx
    t_dl = time.perf_counter()
    with httpx.Client(timeout=600) as client:
        r = client.get(video_url)
        r.raise_for_status()
    download_ms = round((time.perf_counter() - t_dl) * 1000, 1)

    # Per-clip cache check. Hash the first 256KB of the blob bytes —
    # URLs change per signed token, content doesn't.
    cache_key = hashlib.sha256(r.content[:256 * 1024]).hexdigest()
    cache_key = f"{cache_key}:fps={sample_fps}"
    try:
        cached = clip_cache.get(cache_key)
    except Exception as e:  # noqa: BLE001
        # Cache lookup failures are non-fatal — fall through to
        # real inference.
        print(f"[extract] cache get failed (non-fatal): {e}")
        cached = None
    if cached is not None:
        cached_result = dict(cached)
        cached_result["timing"] = {
            "download_ms": download_ms,
            "inference_ms": 0.0,
            "function_ms": round((time.perf_counter() - t_fn_start) * 1000, 1),
            "cold_start": cold_start,
            "cache_hit": True,
        }
        print(f"[extract] cache hit timing={cached_result['timing']}")
        _warm = True
        return cached_result

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        # extraction.py is the lean import surface — no supabase /
        # mediapipe / fastapi pulled in transitively. Cuts ~3s off the
        # cold-start import budget vs. importing through main.py.
        from extraction import extract_keypoints_from_video

        t_inf = time.perf_counter()
        result = extract_keypoints_from_video(tmp_path, sample_fps=sample_fps)
        inference_ms = round((time.perf_counter() - t_inf) * 1000, 1)

        # Stamp the path so the diagnostic chip can show it.
        result["pose_backend"] = (
            f"rtmpose-modal-{os.environ.get('POSE_DEVICE', 'cuda')}"
        )
        result["timing"] = {
            "download_ms": download_ms,
            "inference_ms": inference_ms,
            "function_ms": round((time.perf_counter() - t_fn_start) * 1000, 1),
            "cold_start": cold_start,
            "cache_hit": False,
        }
        print(f"[extract] timing={result['timing']}")

        # Populate cache for future requests on the same content.
        # Cache the keypoints payload only (without timing) so a later
        # hit produces fresh timing data.
        try:
            cache_payload = {k: v for k, v in result.items() if k != "timing"}
            clip_cache[cache_key] = cache_payload
        except Exception as e:  # noqa: BLE001
            print(f"[extract] cache put failed (non-fatal): {e}")

        # Mark this container warm for the next invocation.
        _warm = True
        return result
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
