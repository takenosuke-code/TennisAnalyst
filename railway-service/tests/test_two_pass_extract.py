"""Tests for the two-pass pose extraction orchestration.

The orchestration lives in `extraction._two_pass_extract` (Python). The
JS layer makes a single round-trip to Modal/Railway and the Python
service decides internally whether to run pass 1 + pass 2 or fall back
to a single heavy pass.

These tests use stubbed seams (fake video probe, fake wrist pass, fake
heavy extractor) so they run in milliseconds and don't need real ONNX
weights or video files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Defensive env stubs in case main.py is pulled transitively.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")
os.environ.setdefault("EXTRACT_API_KEY", "test-api-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_wrist_trajectory(
    *,
    n_samples: int,
    sample_dt_ms: float = 125.0,
    peak_starts_at_ms: float = 30_000,
    peak_interval_ms: float = 30_000,
    n_peaks: int = 6,
    seed: int = 0,
) -> tuple[list[float], list[tuple[float, float] | None]]:
    """Construct a synthetic wrist trajectory: noisy idle floor with
    `n_peaks` injected peaks. Returns (timestamps_ms, wrists).

    The detector requires both `spread > 0` and `max > 0`, so a
    completely flat trajectory plus a few isolated peaks fails the
    threshold algorithm in pathological ways (90% of samples are
    identical → p90 = median → spread = 0). The small noise floor
    here mimics what real RTMPose-on-video looks like during idle
    frames.
    """
    import random
    rng = random.Random(seed)
    timestamps_ms = [i * sample_dt_ms for i in range(n_samples)]
    wrists: list[tuple[float, float] | None] = [
        (0.3 + rng.uniform(-0.005, 0.005), 0.5 + rng.uniform(-0.005, 0.005))
        for _ in range(n_samples)
    ]
    for k in range(n_peaks):
        peak_ms = peak_starts_at_ms + k * peak_interval_ms
        peak_idx = int(peak_ms / sample_dt_ms)
        if 0 < peak_idx < n_samples - 1:
            wrists[peak_idx - 1] = (0.5, 0.5)
            wrists[peak_idx] = (0.8, 0.5)
            wrists[peak_idx + 1] = (0.5, 0.5)
    return timestamps_ms, wrists


def _stub_video_probe(monkeypatch, *, fps: float = 30.0, total_frames: int = 9000):
    """Stub `_open_video_autorotated` so the orchestrator can probe a
    fake video without a real file. Used to control the duration check.

    `total_frames / fps` = duration_sec — a 5-min clip = 9000 / 30 = 300s.
    """
    import extraction

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.get.side_effect = lambda prop: {
        # cv2.CAP_PROP_FPS = 5, cv2.CAP_PROP_FRAME_COUNT = 7
        5: fps,
        7: float(total_frames),
    }.get(prop, 0)
    fake_cap.release.return_value = None

    monkeypatch.setattr(extraction, "_open_video_autorotated", lambda _p: fake_cap)
    return fake_cap


# ---------------------------------------------------------------------------
# _detect_strokes_from_wrist
# ---------------------------------------------------------------------------


class TestDetectStrokesFromWrist:
    """The Python reimplementation of the JS wrist-velocity peak-pick.

    Verifies it produces sensible candidate windows for synthetic
    trajectories and that the refractory + padding match the documented
    contract.
    """

    def test_no_strokes_on_flat_signal(self):
        from extraction import _detect_strokes_from_wrist

        # 100 frames at 10fps = 10s of static wrist position.
        timestamps_ms = [i * 100.0 for i in range(100)]
        wrists = [(0.5, 0.5)] * 100

        out = _detect_strokes_from_wrist(timestamps_ms, wrists, video_fps=30.0)
        assert out == []

    def test_detects_single_peak_with_padding(self):
        from extraction import (
            _detect_strokes_from_wrist,
            STROKE_PRE_PAD_MS,
            STROKE_POST_PAD_MS,
        )

        # 100 samples at 10fps = 10s. Realistic noise floor (small
        # random wrist drift) with a sharp peak at sample 50. Without
        # the noise floor, p90 of the speed series equals the median,
        # the threshold collapses, and the detector returns [] —
        # which is the correct flat-signal behavior, just not what
        # this test wants to exercise.
        import random
        random.seed(42)
        timestamps_ms = [i * 100.0 for i in range(100)]
        wrists: list[tuple[float, float] | None] = [
            (0.3 + random.uniform(-0.005, 0.005), 0.5 + random.uniform(-0.005, 0.005))
            for _ in range(100)
        ]
        # Sharp peak at sample 50. Three-sample ramp clears the
        # smoothing-window-3 dampening.
        wrists[49] = (0.4, 0.5)
        wrists[50] = (0.7, 0.5)
        wrists[51] = (0.4, 0.5)

        out = _detect_strokes_from_wrist(timestamps_ms, wrists, video_fps=30.0)
        assert len(out) == 1
        s, e, p = out[0]
        # Peak should map to ORIGINAL-VIDEO frame ~ (5000ms / 1000) * 30 = 150
        assert abs(p - 150) <= 3
        # Pre/post pad in original frames: 1000ms * 30 = 30 frames before,
        # 500ms * 30 = 15 frames after.
        assert s == max(0, p - int(STROKE_PRE_PAD_MS / 1000.0 * 30))
        assert e == p + int(STROKE_POST_PAD_MS / 1000.0 * 30)

    @staticmethod
    def _noisy_floor(n: int, seed: int = 42) -> list[tuple[float, float] | None]:
        """Build a noisy idle wrist trajectory of length `n`. Noise is
        small enough not to clear the prominence threshold but ensures
        spread > 0 so the algorithm doesn't fall through to its
        flat-signal early-return guard."""
        import random
        rng = random.Random(seed)
        return [
            (0.3 + rng.uniform(-0.005, 0.005), 0.5 + rng.uniform(-0.005, 0.005))
            for _ in range(n)
        ]

    def test_refractory_suppresses_close_peaks(self):
        from extraction import _detect_strokes_from_wrist

        # 100 samples at 10fps. Two peaks 200ms apart (within the
        # 500ms refractory). Stronger peak should survive.
        timestamps_ms = [i * 100.0 for i in range(100)]
        wrists = self._noisy_floor(100)
        # Peak 1 at sample 30 — moderate motion
        wrists[29] = (0.45, 0.5)
        wrists[30] = (0.55, 0.5)
        wrists[31] = (0.45, 0.5)
        # Peak 2 at sample 32 — bigger motion (within refractory)
        wrists[32] = (0.7, 0.5)
        wrists[33] = (0.4, 0.5)

        out = _detect_strokes_from_wrist(timestamps_ms, wrists, video_fps=30.0)
        # Only the stronger peak survives. Allow either to be chosen
        # if the smoothing window happens to merge them — primary
        # contract is "not two windows".
        assert len(out) == 1

    def test_refractory_allows_distant_peaks(self):
        from extraction import _detect_strokes_from_wrist

        # 100 samples at 10fps. Two peaks 1500ms apart — well clear of
        # the 500ms refractory. Both should survive.
        timestamps_ms = [i * 100.0 for i in range(100)]
        wrists = self._noisy_floor(100)
        # Peak 1 at sample 20
        wrists[19] = (0.5, 0.5)
        wrists[20] = (0.7, 0.5)
        wrists[21] = (0.5, 0.5)
        # Peak 2 at sample 35 (1500ms later)
        wrists[34] = (0.5, 0.5)
        wrists[35] = (0.7, 0.5)
        wrists[36] = (0.5, 0.5)

        out = _detect_strokes_from_wrist(timestamps_ms, wrists, video_fps=30.0)
        assert len(out) == 2
        # And they must be sorted by start frame.
        assert out[0][0] <= out[1][0]

    def test_handles_missing_wrist_frames(self):
        from extraction import _detect_strokes_from_wrist

        # Wrist tracking failure during the start of the clip should
        # not crash the detector.
        timestamps_ms = [i * 100.0 for i in range(100)]
        # First 30 frames untracked, then noisy floor, then peak.
        wrists: list[tuple[float, float] | None] = [None] * 30 + self._noisy_floor(70, seed=7)
        # Peak still detectable in the tracked half.
        wrists[60] = (0.7, 0.5)

        # Should not raise.
        out = _detect_strokes_from_wrist(timestamps_ms, wrists, video_fps=30.0)
        assert isinstance(out, list)

    def test_rejects_mismatched_lengths(self):
        from extraction import _detect_strokes_from_wrist

        with pytest.raises(ValueError):
            _detect_strokes_from_wrist(
                [0.0, 100.0, 200.0],
                [(0.5, 0.5)],  # length mismatch
                video_fps=30.0,
            )


# ---------------------------------------------------------------------------
# _merge_windows
# ---------------------------------------------------------------------------


class TestMergeWindows:
    """Window merging keeps pass 2 from running pose inference twice on
    the same frame when two strokes' pre/post pads overlap."""

    def test_empty_returns_empty(self):
        from extraction import _merge_windows

        assert _merge_windows([]) == []

    def test_non_overlapping_preserved(self):
        from extraction import _merge_windows

        windows = [(0, 10, 5), (20, 30, 25)]
        out = _merge_windows(windows)
        assert out == [(0, 10, 5), (20, 30, 25)]

    def test_overlapping_merge(self):
        from extraction import _merge_windows

        # Window A: 0-50, peak 20. Window B: 30-80, peak 60.
        # Should merge to 0-80, with the LATER peak (60) preserved.
        windows = [(0, 50, 20), (30, 80, 60)]
        out = _merge_windows(windows)
        assert out == [(0, 80, 60)]

    def test_unsorted_input_gets_sorted(self):
        from extraction import _merge_windows

        # Peak ordering is preserved by start_frame ordering, not by
        # input order.
        windows = [(20, 30, 25), (0, 10, 5)]
        out = _merge_windows(windows)
        assert out == [(0, 10, 5), (20, 30, 25)]

    def test_adjacent_within_gap_merges(self):
        from extraction import _merge_windows

        # Gap of 3 frames between the windows. With merge_gap_frames=5
        # they should fuse.
        windows = [(0, 10, 5), (13, 25, 20)]
        out = _merge_windows(windows, merge_gap_frames=5)
        assert out == [(0, 25, 20)]

    def test_default_zero_gap_does_not_merge_adjacent(self):
        from extraction import _merge_windows

        # With gap=0, only literal overlaps merge.
        windows = [(0, 10, 5), (15, 25, 20)]
        out = _merge_windows(windows)
        assert out == [(0, 10, 5), (15, 25, 20)]


# ---------------------------------------------------------------------------
# _two_pass_extract orchestration
# ---------------------------------------------------------------------------


class TestTwoPassOrchestration:
    """The full two-pass orchestration: pass 1 → stroke detect → pass 2.

    Stubs the heavy seams (`_run_wrist_pass`, `_extract_with_rtmpose`,
    `_open_video_autorotated`) so we can assert the orchestration
    decisions without running real inference.
    """

    def test_short_clip_skips_pass1(self, monkeypatch):
        """For < TWO_PASS_MIN_DURATION_SEC clips, the orchestrator
        skips pass 1 entirely and just runs single-pass directly. The
        speedup of two-pass isn't worth the overhead on a 15s clip.
        """
        import extraction

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=300)  # 10s clip

        wrist_called = {"called": False}

        def fake_wrist_pass(*_args, **_kwargs):
            wrist_called["called"] = True
            return [], [], 30.0, 10.0

        heavy_called = []

        def fake_heavy(video_path, sample_fps, max_seconds, candidate_windows=None):
            heavy_called.append({
                "video_path": video_path,
                "sample_fps": sample_fps,
                "max_seconds": max_seconds,
                "candidate_windows": candidate_windows,
            })
            return {
                "frames": [{"frame_index": 0}],
                "frame_count": 1,
                "schema_version": 3,
                "fps_sampled": sample_fps,
            }

        monkeypatch.setattr(extraction, "_run_wrist_pass", fake_wrist_pass)
        monkeypatch.setattr(extraction, "_extract_with_rtmpose", fake_heavy)

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )

        # Pass 1 should NOT have been called.
        assert wrist_called["called"] is False
        # Heavy should have been called WITHOUT candidate_windows
        # (i.e. single-pass).
        assert len(heavy_called) == 1
        assert heavy_called[0]["candidate_windows"] is None
        # Pass summary should reflect the short-clip fast path.
        assert result["pass_summary"]["mode"] == "single-pass-short-clip"

    def test_long_clip_with_strokes_uses_two_pass(self, monkeypatch):
        """For long clips with detectable strokes, the orchestrator
        runs pass 1, computes candidate windows, and passes them to
        pass 2."""
        import extraction
        import random

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)  # 5min clip

        # Pass 1 returns a wrist trajectory with 6 clean peaks ~3s
        # apart, plus a tiny noise floor (real video always has some
        # motion; the detector's threshold algorithm relies on
        # spread > 0).
        rng = random.Random(123)
        timestamps_ms = [i * 125.0 for i in range(2400)]  # 5min @ 8fps = 2400
        wrists: list[tuple[float, float] | None] = [
            (0.3 + rng.uniform(-0.005, 0.005), 0.5 + rng.uniform(-0.005, 0.005))
            for _ in range(2400)
        ]
        # 6 peaks at 3-second intervals starting at 30s.
        for k in range(6):
            peak_ms = 30_000 + k * 3000
            peak_idx = int(peak_ms / 125)
            if peak_idx < len(wrists):
                wrists[peak_idx - 1] = (0.5, 0.5)
                wrists[peak_idx] = (0.8, 0.5)
                wrists[peak_idx + 1] = (0.5, 0.5)

        def fake_wrist_pass(*_args, **_kwargs):
            return timestamps_ms, wrists, 30.0, 300.0

        heavy_called = []

        def fake_heavy(video_path, sample_fps, max_seconds, candidate_windows=None):
            heavy_called.append(candidate_windows)
            return {
                "frames": [{"frame_index": 0}],
                "frame_count": 1,
                "schema_version": 3,
                "fps_sampled": sample_fps,
            }

        monkeypatch.setattr(extraction, "_run_wrist_pass", fake_wrist_pass)
        monkeypatch.setattr(extraction, "_extract_with_rtmpose", fake_heavy)

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )

        # Heavy should have been called exactly ONCE, with non-empty
        # candidate windows. This is the load-bearing assertion: a
        # batched pass-2 call, not per-stroke calls.
        assert len(heavy_called) == 1
        windows = heavy_called[0]
        assert windows is not None
        assert len(windows) >= extraction.MIN_STROKES_FOR_TWO_PASS
        # And the orchestrator should report two-pass mode.
        assert result["pass_summary"]["mode"] == "two-pass"
        assert result["pass_summary"]["candidate_count"] == len(windows)

    def test_long_clip_with_no_strokes_falls_back_to_single_pass(
        self, monkeypatch
    ):
        """When pass 1 detects no strokes on a long clip (e.g. the user
        uploaded warm-up rallying or pure walking footage), fall back to
        single-pass rather than silently dropping all frames.
        """
        import extraction

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)  # 5min

        def fake_wrist_pass(*_args, **_kwargs):
            # Flat signal = no peaks.
            timestamps_ms = [i * 125.0 for i in range(2400)]
            wrists = [(0.5, 0.5)] * len(timestamps_ms)
            return timestamps_ms, wrists, 30.0, 300.0

        heavy_calls = []

        def fake_heavy(video_path, sample_fps, max_seconds, candidate_windows=None):
            heavy_calls.append(candidate_windows)
            return {
                "frames": [],
                "frame_count": 0,
                "schema_version": 3,
                "fps_sampled": sample_fps,
            }

        monkeypatch.setattr(extraction, "_run_wrist_pass", fake_wrist_pass)
        monkeypatch.setattr(extraction, "_extract_with_rtmpose", fake_heavy)

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )

        # Heavy called once with NO candidate windows (single-pass).
        assert len(heavy_calls) == 1
        assert heavy_calls[0] is None
        assert result["pass_summary"]["mode"] == "single-pass-few-strokes"

    def test_long_clip_with_few_strokes_falls_back_to_single_pass(
        self, monkeypatch
    ):
        """When pass 1 detects fewer than MIN_STROKES_FOR_TWO_PASS, the
        single-pass overhead is acceptable AND we'd rather over-process
        than risk pass 1 having missed a real stroke."""
        import extraction
        import random

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)

        # Inject 2 peaks (below MIN_STROKES_FOR_TWO_PASS=5) on a
        # noisy floor.
        rng = random.Random(99)
        timestamps_ms = [i * 125.0 for i in range(2400)]
        wrists: list[tuple[float, float] | None] = [
            (0.3 + rng.uniform(-0.005, 0.005), 0.5 + rng.uniform(-0.005, 0.005))
            for _ in range(2400)
        ]
        for k in range(2):
            peak_idx = int((30_000 + k * 3000) / 125)
            wrists[peak_idx - 1] = (0.5, 0.5)
            wrists[peak_idx] = (0.8, 0.5)
            wrists[peak_idx + 1] = (0.5, 0.5)

        monkeypatch.setattr(
            extraction, "_run_wrist_pass",
            lambda *_a, **_k: (timestamps_ms, wrists, 30.0, 300.0),
        )

        heavy_calls = []
        monkeypatch.setattr(
            extraction,
            "_extract_with_rtmpose",
            lambda vp, sf, ms, candidate_windows=None: (
                heavy_calls.append(candidate_windows)
                or {
                    "frames": [],
                    "frame_count": 0,
                    "schema_version": 3,
                    "fps_sampled": sf,
                }
            ),
        )

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )
        # Single-pass fallback: candidate_windows=None.
        assert heavy_calls == [None]
        assert result["pass_summary"]["mode"] == "single-pass-few-strokes"

    def test_pass1_failure_falls_back_to_single_pass(self, monkeypatch):
        """If pass 1 raises (transient YOLO / RTMPose init failure), do
        NOT take the user down — fall back to single-pass and log."""
        import extraction

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)

        def boom(*_args, **_kwargs):
            raise RuntimeError("YOLO died")

        heavy_calls = []
        monkeypatch.setattr(extraction, "_run_wrist_pass", boom)
        monkeypatch.setattr(
            extraction,
            "_extract_with_rtmpose",
            lambda vp, sf, ms, candidate_windows=None: (
                heavy_calls.append(candidate_windows)
                or {
                    "frames": [],
                    "frame_count": 0,
                    "schema_version": 3,
                    "fps_sampled": sf,
                }
            ),
        )

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )
        assert heavy_calls == [None]
        assert result["pass_summary"]["mode"] == "single-pass-pass1-failed"
        assert "YOLO died" in result["pass_summary"]["pass1_error"]

    def test_two_pass_false_skips_orchestration(self, monkeypatch):
        """`two_pass=False` should bypass the orchestrator entirely so
        callers that already know they want single-pass can opt out
        without paying the video-probe overhead."""
        import extraction

        # _open_video_autorotated should NOT be called when two_pass=False.
        probe_called = {"called": False}

        def fake_probe(_p):
            probe_called["called"] = True
            cap = MagicMock()
            cap.isOpened.return_value = True
            return cap

        monkeypatch.setattr(extraction, "_open_video_autorotated", fake_probe)
        monkeypatch.setattr(
            extraction,
            "_extract_with_rtmpose",
            lambda *a, **k: {
                "frames": [],
                "frame_count": 0,
                "schema_version": 3,
                "fps_sampled": 30,
            },
        )

        result = extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, two_pass=False,
        )
        assert probe_called["called"] is False
        # No pass_summary block when two_pass=False (caller asked for the
        # raw single-pass path).
        assert "pass_summary" not in result


# ---------------------------------------------------------------------------
# Speedup invariant (mocked timings)
# ---------------------------------------------------------------------------


class TestSpeedupInvariant:
    """Frame-count invariant: pass-1 frames + pass-2 candidate frames
    should be substantially less than processing every frame.

    Real timing on Modal T4 isn't measurable in unit tests, so we
    measure the FRAME COUNT each path would process — proportional to
    inference cost — and assert two-pass beats single-pass by a meaningful
    margin on a long clip with sparse strokes.
    """

    def test_long_clip_two_pass_frames_less_than_single_pass(self, monkeypatch):
        import extraction

        # 5-minute clip, 30fps = 9000 frames. At sample_fps=30, single
        # pass would process all 9000. With ~5 strokes spaced ~30s
        # apart and 1500ms windows each, pass 2 should process ~5
        # windows × ~1.5s × 30fps = ~225 frames.
        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)

        # 6 evenly-spaced peaks throughout the clip.
        timestamps_ms, wrists = _build_wrist_trajectory(
            n_samples=2400, peak_starts_at_ms=30_000, peak_interval_ms=30_000,
            n_peaks=6, seed=11,
        )

        monkeypatch.setattr(
            extraction, "_run_wrist_pass",
            lambda *_a, **_k: (timestamps_ms, wrists, 30.0, 300.0),
        )

        captured_windows = []

        def capture(video_path, sample_fps, max_seconds, candidate_windows=None):
            captured_windows.append(candidate_windows)
            return {
                "frames": [],
                "frame_count": 0,
                "schema_version": 3,
                "fps_sampled": sample_fps,
            }

        monkeypatch.setattr(extraction, "_extract_with_rtmpose", capture)

        extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )

        # Pass 2 got candidate windows; total frame coverage should be
        # well below 9000.
        assert len(captured_windows) == 1
        windows = captured_windows[0]
        assert windows is not None
        total_window_frames = sum(e - s for s, e, _p in windows)
        # Generous bound: even if windows were 2s each, 6 windows × 60
        # frames = 360 << 9000.
        assert total_window_frames < 9000 / 4, (
            f"two-pass should be ~5-10× narrower than single-pass; "
            f"got {total_window_frames} / 9000 frames"
        )

    def test_pass2_runs_only_once(self, monkeypatch):
        """Critical perf invariant: pass 2 is ONE call into the heavy
        extractor with all candidate windows batched, not one call per
        stroke. Sequential per-stroke calls would each pay the chunk
        pool spin-up + ORT session warm-up overhead."""
        import extraction

        _stub_video_probe(monkeypatch, fps=30.0, total_frames=9000)

        # 8 strokes — well above MIN_STROKES_FOR_TWO_PASS.
        timestamps_ms, wrists = _build_wrist_trajectory(
            n_samples=2400, peak_starts_at_ms=20_000, peak_interval_ms=5000,
            n_peaks=8, seed=29,
        )

        monkeypatch.setattr(
            extraction, "_run_wrist_pass",
            lambda *_a, **_k: (timestamps_ms, wrists, 30.0, 300.0),
        )

        call_count = {"n": 0}

        def counting_heavy(video_path, sample_fps, max_seconds, candidate_windows=None):
            call_count["n"] += 1
            return {
                "frames": [],
                "frame_count": 0,
                "schema_version": 3,
                "fps_sampled": sample_fps,
            }

        monkeypatch.setattr(extraction, "_extract_with_rtmpose", counting_heavy)

        extraction.extract_keypoints_from_video(
            "/fake.mp4", sample_fps=30, max_seconds=0
        )

        # Heavy extractor called exactly once with all 8 windows batched.
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# _extract_with_rtmpose: candidate_windows filter behavior
# ---------------------------------------------------------------------------


class TestExtractWithRtmposeCandidateWindows:
    """Verify the heavy extractor's frame-filtering behavior when
    candidate_windows is supplied. Reuses the stubbed-pose pattern
    from test_extract_backend_dispatch.py."""

    def test_skips_frames_outside_candidate_windows(self, monkeypatch):
        import extraction
        import pose_rtmpose

        # 30 frames at 30fps. Candidate windows: 5-10 and 20-25.
        # Should process ~12 of the 30 frames.
        def fake_yolo(_frame):
            return [(10.0, 10.0, 90.0, 90.0, 0.9)], 100, 100

        def fake_rtm(_frame, bbox):
            if bbox is None:
                return None
            return [
                {
                    "id": i, "name": f"l_{i}",
                    "x": 0.5, "y": 0.5, "z": 0.0,
                    "visibility": 0.9 if i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26) else 0.0,
                }
                for i in range(33)
            ]

        monkeypatch.setattr(pose_rtmpose, "_yolo_person_candidates", fake_yolo)
        monkeypatch.setattr(pose_rtmpose, "infer_rtmpose_for_bbox", fake_rtm)
        monkeypatch.setattr(pose_rtmpose, "_ensure_rtmpose", lambda: None)

        # Stub VideoCapture: 30 frames, 30fps.
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        frames_iter = iter(
            [(True, np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(30)]
            + [(False, None)]
        )
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0
        monkeypatch.setattr(extraction.cv2, "VideoCapture", lambda _path: fake_cap)
        monkeypatch.setattr(extraction, "_try_load_racket_detector", lambda: None)

        out = extraction._extract_with_rtmpose(
            "/fake.mp4",
            sample_fps=30,
            max_seconds=0,
            candidate_windows=[(5, 10, 7), (20, 25, 22)],
        )

        # Candidate windows total 6 + 6 = 12 frames. With sample_fps=30
        # and video_fps=30, every frame is sampled — so all 12 in-window
        # frames are processed.
        assert out["frame_count"] == 12

    def test_no_candidate_windows_processes_every_frame(self, monkeypatch):
        """Sanity check — the existing single-pass path is unchanged
        when candidate_windows=None."""
        import extraction
        import pose_rtmpose

        def fake_yolo(_frame):
            return [(10.0, 10.0, 90.0, 90.0, 0.9)], 100, 100

        def fake_rtm(_frame, bbox):
            if bbox is None:
                return None
            return [
                {
                    "id": i, "name": f"l_{i}",
                    "x": 0.5, "y": 0.5, "z": 0.0,
                    "visibility": 0.9 if i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26) else 0.0,
                }
                for i in range(33)
            ]

        monkeypatch.setattr(pose_rtmpose, "_yolo_person_candidates", fake_yolo)
        monkeypatch.setattr(pose_rtmpose, "infer_rtmpose_for_bbox", fake_rtm)
        monkeypatch.setattr(pose_rtmpose, "_ensure_rtmpose", lambda: None)

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        frames_iter = iter(
            [(True, np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(8)]
            + [(False, None)]
        )
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0
        monkeypatch.setattr(extraction.cv2, "VideoCapture", lambda _path: fake_cap)
        monkeypatch.setattr(extraction, "_try_load_racket_detector", lambda: None)

        out = extraction._extract_with_rtmpose(
            "/fake.mp4", sample_fps=30, max_seconds=0,
        )
        # All 8 frames should be processed.
        assert out["frame_count"] == 8
