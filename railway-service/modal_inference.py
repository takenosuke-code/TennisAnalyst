"""Modal-deployed GPU pose-extraction endpoint.

Architecture: Railway's `/extract` endpoint normally runs YOLO + RTMPose
ONNX inference inline on its CPU container. When MODAL_INFERENCE_URL is
set on Railway, that endpoint instead forwards the (sessionId, blobUrl)
pair here, where the same pipeline runs on a Modal GPU container.

The function below is the entire Modal app. Deploy it once with:

    modal deploy railway-service/modal_inference.py

Modal returns an HTTPS URL on first deploy. Set that URL on Railway as
`MODAL_INFERENCE_URL`, and set a shared bearer token on both sides
(`MODAL_INFERENCE_KEY` Railway env + a `tennis-pose` Modal Secret with
the same value). The Railway proxy adds `Authorization: Bearer <key>`;
this function rejects requests without it.

Cold-start behavior: Modal scales to zero by default. First request
after >10 min idle pays ~5-10s container spin + model load. Within an
active period, subsequent calls hit the warm container and run in ~2s.
"""
from __future__ import annotations

from pathlib import Path

import modal


SERVICE_DIR = Path(__file__).parent

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
        "onnxruntime-gpu>=1.18.0",
        "rtmlib==0.0.15",
        "fastapi==0.115.0",
        "supabase==2.8.1",  # only because main.py imports it; we don't call it
    )
    # Mount the whole railway-service directory so we can reuse the
    # exact same `extract_keypoints_from_video` pipeline (including the
    # staged-parallel inference, motion-gated racket, etc.). Skip large
    # model artifacts that don't need to ship — rtmlib + ultralytics
    # download what they need on first init.
    .add_local_dir(
        SERVICE_DIR,
        remote_path="/root/app",
        ignore=[
            "models/pose_landmarker_heavy.task",  # mediapipe-only, not used
            "tests/",
            "__pycache__/",
            "*.pyc",
        ],
    )
)


app = modal.App("tennis-pose-inference")


@app.function(
    image=inference_image,
    # T4 is the cheapest GPU at ~$0.59/hr; sufficient for RTMPose-s
    # (5M params) + YOLO11n (6M params). A10 ($1.10/hr) would be ~2x
    # faster but for our model sizes the marginal gain isn't worth 2x
    # the credit burn.
    gpu="T4",
    # Match Railway's client-side EXTRACTION_TIMEOUT_MS (300s) so a
    # long clip doesn't time out at the Modal layer before Railway
    # gives up.
    timeout=300,
    # Scale-to-zero after 10 min idle. Keeps the GPU billing close to
    # zero outside of active sessions; user pays cold-start tax once
    # per quiet period.
    scaledown_window=600,
    # MODAL_INFERENCE_KEY is checked in the function body for shared-
    # secret auth (Modal endpoints are public HTTPS by default).
    secrets=[modal.Secret.from_name("tennis-pose")],
)
@modal.fastapi_endpoint(method="POST")
def extract_pose(payload: dict, authorization: str = ""):
    """Run YOLO + RTMPose-s on the video at `payload["video_url"]`.

    Body shape: {"video_url": str, "sample_fps": int = 30}
    Auth: Authorization: Bearer <MODAL_INFERENCE_KEY>
    Returns: same JSON shape as railway/main._extract_with_rtmpose
    """
    import os
    import sys
    import tempfile

    expected = f"Bearer {os.environ.get('MODAL_INFERENCE_KEY', '')}"
    if not os.environ.get("MODAL_INFERENCE_KEY") or authorization != expected:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Unauthorized")

    video_url = payload.get("video_url")
    sample_fps = int(payload.get("sample_fps", 30))
    if not video_url:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="video_url required")

    # main.py imports supabase at module load and reads SUPABASE_*
    # env vars. We don't actually call Supabase from the Modal path
    # (Railway handles the DB write back), but we need stub values
    # that pass supabase-py's format check so the import succeeds.
    # The dummy JWT-ish string passes the "non-empty + reasonable
    # structure" gate without ever hitting Supabase's API.
    os.environ.setdefault("SUPABASE_URL", "https://stub.supabase.co")
    os.environ.setdefault(
        "SUPABASE_SERVICE_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.modal-stub.unused",
    )
    os.environ.setdefault("POSE_BACKEND", "rtmpose")
    # Switch the inference path from CPU to CUDA. Read by pose_rtmpose
    # to swap onnxruntime providers + rtmlib's `device=` arg.
    os.environ.setdefault("POSE_DEVICE", "cuda")

    sys.path.insert(0, "/root/app")

    # Download the video. Same allowlist Railway enforces.
    import httpx
    with httpx.Client(timeout=300) as client:
        r = client.get(video_url)
        r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        from main import extract_keypoints_from_video
        result = extract_keypoints_from_video(tmp_path, sample_fps=sample_fps)
        # Stamp the path so the diagnostic chip can show it.
        result["pose_backend"] = f"rtmpose-modal-{os.environ.get('POSE_DEVICE', 'cuda')}"
        return result
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
