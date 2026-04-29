"""Smoke test for the rtmpose extraction pipeline.

Originally this file tested the POSE_BACKEND env-var dispatch, but
mediapipe was retired in favor of an rtmpose-only path. The dispatch
is gone, so the only valuable check left is the end-to-end shape of
`_extract_with_rtmpose` driven by stubbed YOLO + RTMPose seams. The
file name is kept for git-history continuity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# extraction.py doesn't touch Supabase, but main.py (which other tests
# may import) does. Stub the Supabase-related env vars defensively in
# case something pulls main.py in transitively.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")
os.environ.setdefault("EXTRACT_API_KEY", "test-api-key")


class TestRtmposeExtractWithStubs:
    """Drive _extract_with_rtmpose end-to-end using stubbed video I/O and
    stubbed pose inference. Verifies the per-frame loop, racket-head
    wiring, and the schema_version=3 stamp."""

    def test_emits_schema_v3_with_stubbed_pose(self, monkeypatch):
        import extraction
        import pose_rtmpose

        # The staged-parallel loop in _extract_with_rtmpose calls the
        # lower-level seams directly (so YOLO and RTMPose can be staged
        # separately for parallelism). Stub both.
        def fake_yolo_candidates(_frame):
            # One person candidate covering most of the frame.
            return [(10.0, 10.0, 90.0, 90.0, 0.9)], 100, 100

        def fake_rtmpose_for_bbox(_frame, bbox):
            if bbox is None:
                return None
            return [
                {
                    "id": i,
                    "name": f"landmark_{i}",
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9 if i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28) else 0.0,
                }
                for i in range(33)
            ]

        monkeypatch.setattr(pose_rtmpose, "_yolo_person_candidates", fake_yolo_candidates)
        monkeypatch.setattr(pose_rtmpose, "infer_rtmpose_for_bbox", fake_rtmpose_for_bbox)
        # _ensure_rtmpose is called once before the chunk loop; stub it
        # so we don't try to lazy-load real ONNX in tests.
        monkeypatch.setattr(pose_rtmpose, "_ensure_rtmpose", lambda: None)

        # Stub OpenCV VideoCapture so we don't need a real file.
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        # Yield 5 frames then EOF. Each frame is a 100x100 BGR ndarray.
        import numpy as np
        frames_iter = iter([
            (True, np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(5)
        ] + [(False, None)])
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0  # video_fps
        monkeypatch.setattr(extraction.cv2, "VideoCapture", lambda _path: fake_cap)

        # Stub racket detector to None (we test the no-racket branch here).
        monkeypatch.setattr(extraction, "_try_load_racket_detector", lambda: None)

        out = extraction._extract_with_rtmpose("/fake/path.mp4", sample_fps=30, max_seconds=0)

        assert out["schema_version"] == 3
        assert out["frame_count"] == 5
        assert len(out["frames"]) == 5
        # First frame: timestamp_ms=0, frame_index=0, has joint angles, racket_head=None
        f0 = out["frames"][0]
        assert f0["frame_index"] == 0
        assert f0["timestamp_ms"] == 0.0
        assert f0["racket_head"] is None
        # Joint angles must NOT include wrist-flexion (no index_finger landmarks).
        assert "right_wrist" not in f0["joint_angles"]
        assert "left_wrist" not in f0["joint_angles"]
        # But shoulder/elbow/knee/hip angles should be there (the joints we filled).
        assert "right_elbow" in f0["joint_angles"]
        assert "left_elbow" in f0["joint_angles"]

    def test_extract_keypoints_from_video_runs_rtmpose(self, monkeypatch):
        """The public entrypoint should always route to the rtmpose path —
        no env-var dispatch any more."""
        import extraction

        called = {}

        def fake_rtm(video_path, sample_fps, max_seconds):
            called["rtm"] = (video_path, sample_fps, max_seconds)
            return {"frame_count": 0, "frames": [], "schema_version": 3}

        monkeypatch.setattr(extraction, "_extract_with_rtmpose", fake_rtm)
        out = extraction.extract_keypoints_from_video("/tmp/x.mp4", sample_fps=24, max_seconds=2)
        assert called["rtm"] == ("/tmp/x.mp4", 24, 2)
        assert out["schema_version"] == 3
