FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download RTMPose ONNX bundle so it's baked into the image
# instead of fetched from openmmlab's CDN on every container cold
# start. Railway containers are ephemeral; ~/.cache/rtmlib gets wiped
# on each redeploy, so without this every first-of-the-day user pays
# a ~1-2s download tax. The warmup imports pose_rtmpose directly so
# it doesn't need any Railway / Supabase env vars to run.
RUN python -c "from pose_rtmpose import _ensure_rtmpose, _ensure_yolo; _ensure_yolo(); _ensure_rtmpose(); print('models cached')"

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
