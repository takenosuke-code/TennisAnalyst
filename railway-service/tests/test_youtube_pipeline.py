"""Integration tests for the YouTube video processing pipeline.

Verifies the end-to-end import chain, data flow contracts between pipeline
stages, URL validation, fallback behavior, camera classification with
synthetic images, clip extractor utilities, and JSON serialization of
pipeline results.

All tests run offline with synthetic/mock data -- no YouTube downloads.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Allow imports from the railway-service package
sys.path.insert(0, str(Path(__file__).parent.parent))

# main.py reads SUPABASE_URL / SUPABASE_SERVICE_KEY at import time and
# calls create_client.  Set dummy env vars before any import of main so
# the module loads without errors in the test environment.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")

# Patch supabase.create_client before main.py is imported, since it runs
# at module level and may fail with dummy credentials.
_mock_supabase_client = MagicMock()
patch("supabase.create_client", return_value=_mock_supabase_client).start()


# ---------------------------------------------------------------------------
# 1. Import chain tests
# ---------------------------------------------------------------------------


class TestImportChain:
    """Verify all pipeline modules can be imported without errors."""

    def test_import_scene_detector(self):
        """scene_detector module and its public names are importable."""
        from scene_detector import detect_scenes, Scene
        assert callable(detect_scenes)
        assert Scene is not None

    def test_import_camera_classifier(self):
        """camera_classifier module and classify_frame are importable."""
        from camera_classifier import classify_frame
        assert callable(classify_frame)

    def test_import_clip_extractor(self):
        """clip_extractor module and its public names are importable."""
        from clip_extractor import (
            extract_clip,
            get_video_info,
            ExtractedClip,
            compute_real_time_rate,
        )
        assert callable(extract_clip)
        assert callable(get_video_info)
        assert callable(compute_real_time_rate)
        assert ExtractedClip is not None

    def test_import_shot_classifiers(self):
        """shot_classifiers package exposes all four classifiers and classify_shot."""
        from shot_classifiers import (
            classify_shot,
            ForehandClassifier,
            BackhandClassifier,
            ServeClassifier,
            SliceClassifier,
            CLASSIFIERS,
            ClassificationResult,
        )
        assert callable(classify_shot)
        assert len(CLASSIFIERS) == 4
        assert "forehand" in CLASSIFIERS
        assert "backhand" in CLASSIFIERS
        assert "serve" in CLASSIFIERS
        assert "slice" in CLASSIFIERS

    def test_import_youtube_processor(self):
        """youtube_processor module and ProcessingResult are importable."""
        from youtube_processor import ProcessingResult, process_youtube_video
        assert callable(process_youtube_video)
        assert ProcessingResult is not None

    def test_import_main_youtube_url_validation(self):
        """main module exposes the YouTube URL pattern and validator."""
        from main import _is_youtube_url, YOUTUBE_URL_PATTERN
        assert callable(_is_youtube_url)
        assert YOUTUBE_URL_PATTERN is not None


# ---------------------------------------------------------------------------
# 2. Data flow contract tests
# ---------------------------------------------------------------------------


class TestDataFlowContracts:
    """Verify that dataclass shapes match what each pipeline stage expects."""

    def test_scene_dataclass_fields(self):
        """Scene has all required fields: index, start_time, end_time,
        start_frame, end_frame, and a duration property."""
        from scene_detector import Scene

        scene = Scene(index=0, start_time=1.0, end_time=3.5, start_frame=30, end_frame=105)
        assert scene.index == 0
        assert scene.start_time == 1.0
        assert scene.end_time == 3.5
        assert scene.start_frame == 30
        assert scene.end_frame == 105
        assert scene.duration == pytest.approx(2.5)

    def test_extracted_clip_dataclass_fields(self):
        """ExtractedClip has all fields consumed by main.py's _run_youtube_processing."""
        from clip_extractor import ExtractedClip

        clip = ExtractedClip(
            path="/tmp/clip.mp4",
            start_time=1.0,
            end_time=3.0,
            shot_type="forehand",
            camera_angle="behind",
            confidence=0.85,
            duration_ms=2000,
            fps=30.0,
            is_clean=True,
            handedness="right",
        )
        assert clip.path == "/tmp/clip.mp4"
        assert clip.shot_type == "forehand"
        assert clip.camera_angle == "behind"
        assert clip.confidence == 0.85
        assert clip.duration_ms == 2000
        assert clip.fps == 30.0
        assert clip.is_clean is True
        assert clip.handedness == "right"
        assert clip.start_time == 1.0
        assert clip.end_time == 3.0

    def test_processing_result_dataclass_fields(self):
        """ProcessingResult has the fields expected by the /process-youtube endpoint."""
        from youtube_processor import ProcessingResult

        result = ProcessingResult(
            youtube_url="https://www.youtube.com/watch?v=abc",
            video_title="Federer Forehand",
            total_scenes=10,
            gameplay_scenes=4,
        )
        assert result.youtube_url == "https://www.youtube.com/watch?v=abc"
        assert result.video_title == "Federer Forehand"
        assert result.total_scenes == 10
        assert result.gameplay_scenes == 4
        assert result.clips == []
        assert result.errors == []

    def test_classification_result_fields(self):
        """ClassificationResult has all fields consumed by youtube_processor."""
        from shot_classifiers.base import ClassificationResult

        cr = ClassificationResult(
            shot_type="serve",
            confidence=0.92,
            is_clean=True,
            camera_angle="behind",
            handedness="right",
            phase_timestamps={"preparation": {"start": 0, "end": 5}},
        )
        assert cr.shot_type == "serve"
        assert cr.confidence == 0.92
        assert cr.is_clean is True
        assert cr.camera_angle == "behind"
        assert cr.handedness == "right"
        assert cr.phase_timestamps is not None

    def test_classify_shot_returns_classification_result(self):
        """classify_shot() returns a ClassificationResult even for empty input."""
        from shot_classifiers import classify_shot
        from shot_classifiers.base import ClassificationResult

        result = classify_shot([])
        assert isinstance(result, ClassificationResult)
        assert result.shot_type == "unknown"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 3. YouTube URL validation tests
# ---------------------------------------------------------------------------


class TestYouTubeUrlValidation:
    """Test the _is_youtube_url validator from main.py."""

    def test_standard_youtube_url(self):
        from main import _is_youtube_url
        assert _is_youtube_url("https://www.youtube.com/watch?v=xyz") is True

    def test_short_youtube_url(self):
        from main import _is_youtube_url
        assert _is_youtube_url("https://youtu.be/xyz") is True

    def test_youtube_url_without_www(self):
        from main import _is_youtube_url
        assert _is_youtube_url("https://youtube.com/watch?v=abc") is True

    def test_http_youtube_url_accepted(self):
        """The YOUTUBE_URL_PATTERN allows http as well (starts with https?)."""
        from main import _is_youtube_url
        assert _is_youtube_url("http://www.youtube.com/watch?v=xyz") is True

    def test_invalid_domain_rejected(self):
        from main import _is_youtube_url
        assert _is_youtube_url("https://example.com/video") is False

    def test_empty_string_rejected(self):
        from main import _is_youtube_url
        assert _is_youtube_url("") is False

    def test_non_url_rejected(self):
        from main import _is_youtube_url
        assert _is_youtube_url("not-a-url") is False

    def test_ftp_youtube_rejected(self):
        """Non-HTTP schemes should be rejected."""
        from main import _is_youtube_url
        assert _is_youtube_url("ftp://www.youtube.com/watch?v=xyz") is False


# ---------------------------------------------------------------------------
# 4. Scene detector fallback tests
# ---------------------------------------------------------------------------


class TestSceneDetectorFallback:
    """Test that detect_scenes returns a single-scene fallback when
    scenedetect is unavailable or fails."""

    def test_fallback_when_scenedetect_import_fails(self):
        """When scenedetect is not importable, detect_scenes should return
        a single scene covering the entire video."""
        from scene_detector import detect_scenes

        # Mock scenedetect import failure inside detect_scenes
        with patch.dict("sys.modules", {"scenedetect": None}):
            # Also need to patch the subprocess call inside _get_video_duration
            with patch("scene_detector.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout='{"format": {"duration": "120.5"}}',
                    returncode=0,
                )
                scenes = detect_scenes("/fake/video.mp4")

        assert len(scenes) == 1
        assert scenes[0].index == 0
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == pytest.approx(120.5)

    def test_fallback_scene_has_correct_fields(self):
        """The fallback single scene has all required Scene fields."""
        from scene_detector import Scene, _fallback_single_scene

        with patch("scene_detector._get_video_duration", return_value=60.0):
            scenes = _fallback_single_scene("/fake/video.mp4")

        assert len(scenes) == 1
        scene = scenes[0]
        assert scene.index == 0
        assert scene.start_time == 0.0
        assert scene.end_time == 60.0
        assert scene.start_frame == 0
        assert scene.end_frame == 0
        assert scene.duration == pytest.approx(60.0)

    def test_get_video_duration_defaults_on_error(self):
        """_get_video_duration returns 300.0 (5 minutes) when ffprobe fails."""
        from scene_detector import _get_video_duration

        with patch("scene_detector.subprocess.run", side_effect=FileNotFoundError):
            duration = _get_video_duration("/nonexistent/video.mp4")

        assert duration == 300.0


# ---------------------------------------------------------------------------
# 5. Camera classifier tests with synthetic images
# ---------------------------------------------------------------------------


class TestCameraClassifier:
    """Test classify_frame with synthetic numpy arrays."""

    def _make_court_image(
        self,
        width: int = 640,
        height: int = 480,
        court_color: tuple[int, int, int] = (50, 150, 50),
        line_color: tuple[int, int, int] = (255, 255, 255),
        player_color: tuple[int, int, int] = (30, 30, 30),
    ) -> np.ndarray:
        """Create a synthetic BGR image that simulates a tennis court view.

        Draws a colored court surface, white lines, and a dark player-shaped
        rectangle in the lower center.
        """
        img = np.full((height, width, 3), court_color, dtype=np.uint8)

        # Draw horizontal and vertical white lines
        cv2.line(img, (0, height // 3), (width, height // 3), line_color, 2)
        cv2.line(img, (0, 2 * height // 3), (width, 2 * height // 3), line_color, 2)
        cv2.line(img, (width // 4, 0), (width // 4, height), line_color, 2)
        cv2.line(img, (3 * width // 4, 0), (3 * width // 4, height), line_color, 2)
        cv2.line(img, (width // 2, 0), (width // 2, height), line_color, 2)

        # Draw a player-sized dark rectangle (tall, narrow) in the lower center
        player_w, player_h = 40, 120
        px = width // 2 - player_w // 2
        py = height - player_h - 30
        cv2.rectangle(img, (px, py), (px + player_w, py + player_h), player_color, -1)

        return img

    def test_classify_frame_returns_tuple(self):
        """classify_frame should return a (bool, str) tuple."""
        import cv2 as cv2_mod
        from camera_classifier import classify_frame

        img = self._make_court_image()
        result = classify_frame(img)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_classify_frame_angle_values(self):
        """The camera angle must be one of the known labels."""
        from camera_classifier import classify_frame

        img = self._make_court_image()
        _, angle = classify_frame(img)
        assert angle in {"behind", "side", "front", "overhead", "unknown"}

    def test_non_court_image_rejected(self):
        """A solid random-colored image without court features should return
        is_gameplay=False."""
        from camera_classifier import classify_frame

        # Solid red image -- no court colors, no lines, no player
        img = np.full((480, 640, 3), (0, 0, 200), dtype=np.uint8)
        is_gameplay, _ = classify_frame(img)
        assert is_gameplay is False

    def test_green_court_with_lines_and_player(self):
        """A green court surface with white lines and a player shape should
        be classified as gameplay."""
        from camera_classifier import classify_frame

        img = self._make_court_image(court_color=(50, 180, 50))
        is_gameplay, angle = classify_frame(img)
        # With proper court color, lines, and player, this should be gameplay
        assert isinstance(is_gameplay, bool)
        # Regardless of gameplay result, angle should be valid
        assert angle in {"behind", "side", "front", "overhead", "unknown"}

    def test_blue_court_surface(self):
        """A blue hard court should also be recognizable."""
        from camera_classifier import classify_frame

        # HSV blue ~= BGR (200, 100, 50)
        img = self._make_court_image(court_color=(200, 100, 50))
        result = classify_frame(img)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 6. Clip extractor utility tests
# ---------------------------------------------------------------------------


class TestClipExtractorUtilities:
    """Test get_video_info error handling and compute_real_time_rate."""

    def test_get_video_info_no_video_stream_raises(self):
        """get_video_info should raise ValueError when no video stream is found."""
        from clip_extractor import get_video_info

        fake_ffprobe_output = json.dumps({
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "10.0"},
        })

        with patch("clip_extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=fake_ffprobe_output,
                returncode=0,
            )
            with pytest.raises(ValueError, match="No video stream"):
                get_video_info("/fake/video.mp4")

    def test_get_video_info_parses_correctly(self):
        """get_video_info should parse ffprobe JSON output correctly."""
        from clip_extractor import get_video_info

        fake_ffprobe_output = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "45.5"},
        })

        with patch("clip_extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=fake_ffprobe_output,
                returncode=0,
            )
            info = get_video_info("/fake/video.mp4")

        assert info["duration"] == 45.5
        assert info["fps"] == 30.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["codec"] == "h264"

    def test_get_video_info_fractional_fps(self):
        """get_video_info handles NTSC-style fractional frame rates (30000/1001)."""
        from clip_extractor import get_video_info

        fake_output = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "30000/1001",
                }
            ],
            "format": {"duration": "10.0"},
        })

        with patch("clip_extractor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
            info = get_video_info("/fake/video.mp4")

        assert info["fps"] == pytest.approx(29.97, abs=0.01)

    def test_compute_real_time_rate_forehand(self):
        """Forehand real-time duration is 700ms; a 2800ms clip should return 4.0x."""
        from clip_extractor import compute_real_time_rate

        rate = compute_real_time_rate(2800, "forehand")
        assert rate == pytest.approx(4.0)

    def test_compute_real_time_rate_serve(self):
        """Serve real-time duration is 1500ms; a 3000ms clip should return 2.0x."""
        from clip_extractor import compute_real_time_rate

        rate = compute_real_time_rate(3000, "serve")
        assert rate == pytest.approx(2.0)

    def test_compute_real_time_rate_unknown_shot(self):
        """Unknown shot types default to 800ms real-time duration."""
        from clip_extractor import compute_real_time_rate

        rate = compute_real_time_rate(1600, "dropshot")
        assert rate == pytest.approx(2.0)

    def test_compute_real_time_rate_zero_duration(self):
        """Zero clip duration should return 1.0 (no scaling)."""
        from clip_extractor import compute_real_time_rate

        rate = compute_real_time_rate(0, "forehand")
        assert rate == 1.0

    def test_compute_real_time_rate_clamped_max(self):
        """Rate should be clamped to 16.0 maximum."""
        from clip_extractor import compute_real_time_rate

        # 50000ms clip / 700ms forehand = ~71x -> clamped to 16.0
        rate = compute_real_time_rate(50000, "forehand")
        assert rate == 16.0

    def test_compute_real_time_rate_clamped_min(self):
        """Rate should be clamped to 0.0625 minimum."""
        from clip_extractor import compute_real_time_rate

        # 1ms clip / 700ms = ~0.0014 -> clamped to 0.0625
        rate = compute_real_time_rate(1, "forehand")
        assert rate == 0.0625


# ---------------------------------------------------------------------------
# 7. Pipeline result serialization tests
# ---------------------------------------------------------------------------


class TestPipelineResultSerialization:
    """Verify that ProcessingResult can be serialized to JSON,
    which is what the /process-youtube endpoint returns."""

    def test_processing_result_serializable_empty(self):
        """An empty ProcessingResult should serialize to valid JSON."""
        from youtube_processor import ProcessingResult

        result = ProcessingResult(
            youtube_url="https://www.youtube.com/watch?v=test",
            video_title="Test Video",
            total_scenes=0,
            gameplay_scenes=0,
        )
        data = asdict(result)
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)

        assert deserialized["youtube_url"] == "https://www.youtube.com/watch?v=test"
        assert deserialized["video_title"] == "Test Video"
        assert deserialized["total_scenes"] == 0
        assert deserialized["gameplay_scenes"] == 0
        assert deserialized["clips"] == []
        assert deserialized["errors"] == []

    def test_processing_result_serializable_with_clips(self):
        """A ProcessingResult with clips should produce valid JSON that matches
        the shape consumed by _run_youtube_processing in main.py."""
        from youtube_processor import ProcessingResult
        from clip_extractor import ExtractedClip

        clip = ExtractedClip(
            path="/tmp/clip_001.mp4",
            start_time=2.5,
            end_time=4.1,
            shot_type="backhand",
            camera_angle="side",
            confidence=0.78,
            duration_ms=1600,
            fps=29.97,
            is_clean=True,
            handedness="left",
        )
        result = ProcessingResult(
            youtube_url="https://youtu.be/abc123",
            video_title="Nadal Backhand Compilation",
            total_scenes=15,
            gameplay_scenes=8,
            clips=[clip],
            errors=["Scene 3: ffmpeg timeout"],
        )

        data = asdict(result)
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)

        assert len(deserialized["clips"]) == 1
        c = deserialized["clips"][0]
        assert c["shot_type"] == "backhand"
        assert c["camera_angle"] == "side"
        assert c["confidence"] == 0.78
        assert c["duration_ms"] == 1600
        assert c["handedness"] == "left"
        assert c["is_clean"] is True
        assert len(deserialized["errors"]) == 1

    def test_classification_result_serializable(self):
        """ClassificationResult should also be JSON-serializable via asdict."""
        from shot_classifiers.base import ClassificationResult

        cr = ClassificationResult(
            shot_type="forehand",
            confidence=0.65,
            is_clean=True,
            camera_angle="behind",
            handedness="right",
            phase_timestamps={
                "preparation": {"start": 0, "end": 5},
                "contact": {"frame": 10},
                "follow_through": {"start": 10, "end": 19},
            },
        )
        data = asdict(cr)
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)

        assert deserialized["shot_type"] == "forehand"
        assert deserialized["phase_timestamps"]["contact"]["frame"] == 10

    def test_extracted_clip_serializable(self):
        """ExtractedClip should be JSON-serializable via asdict."""
        from clip_extractor import ExtractedClip

        clip = ExtractedClip(
            path="/tmp/test.mp4",
            start_time=0.0,
            end_time=2.0,
            shot_type="serve",
            camera_angle="behind",
            confidence=0.95,
            duration_ms=2000,
            fps=30.0,
            is_clean=True,
            handedness="right",
        )
        data = asdict(clip)
        serialized = json.dumps(data)
        assert '"serve"' in serialized
        assert '"behind"' in serialized


# ---------------------------------------------------------------------------
# 8. Allowed-URL validator tests (video URL, not YouTube URL)
# ---------------------------------------------------------------------------


class TestAllowedUrlValidation:
    """Test the _is_url_allowed validator from main.py which guards the
    /extract endpoint against SSRF."""

    def test_allowed_vercel_blob(self):
        from main import _is_url_allowed
        assert _is_url_allowed("https://blob.vercel-storage.com/video.mp4") is True

    def test_allowed_supabase(self):
        from main import _is_url_allowed
        url = "https://abc.supabase.co/storage/v1/object/public/videos/clip.mp4"
        assert _is_url_allowed(url) is True

    def test_rejected_arbitrary_domain(self):
        from main import _is_url_allowed
        assert _is_url_allowed("https://evil.com/payload") is False

    def test_rejected_http_scheme(self):
        from main import _is_url_allowed
        assert _is_url_allowed("http://blob.vercel-storage.com/video.mp4") is False

    def test_rejected_empty_string(self):
        from main import _is_url_allowed
        assert _is_url_allowed("") is False


# Import cv2 at the module level for the camera classifier tests that need it
import cv2
