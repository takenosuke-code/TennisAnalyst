'use client'

// Dev-only "replay a video file through the live pipeline" tool.
//
// Why this exists: /live can only be exercised on a tennis court, which
// makes the iterate cycle ~hours per round-trip. This page mounts a
// hidden <video> with a local file, derives a MediaStream via
// captureStream(), monkey-patches navigator.mediaDevices.getUserMedia
// to return that stream, and drives the same useLiveCapture +
// useLiveCoach pipeline that powers the production /live screen.
//
// All status/diagnostics render inline (no DevTools). Force-failure
// toggles for /api/extract and /api/live-coach let you exercise the
// fallback paths without unplugging the network. Audio is suppressed
// (the live coach's TTS adapter is muted via the store) so dev usage
// doesn't make the room talk.
//
// Out of scope: anything not directly observable from at-desk replay.
// We don't re-implement the upload / save flow on Stop — the resulting
// LiveSessionResult is just rendered inline. Same posture as
// app/dev/fmp4-spike/page.tsx.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  useLiveCapture,
  type LiveSessionResult,
  type PoseQuality,
} from '@/hooks/useLiveCapture'
import { useLiveCoach } from '@/hooks/useLiveCoach'
import { useLiveStore } from '@/store/live'
import type { PoseDetectStats } from '@/lib/browserPose'
import type { StreamedSwing } from '@/lib/liveSwingDetector'

type FetchToggles = {
  blockExtract: boolean
  blockLiveCoach: boolean
}

type DetectStatsSnapshot = PoseDetectStats & {
  capturedAtMs: number
}

type ReplayState =
  | 'idle'
  | 'loading-file'
  | 'starting'
  | 'replaying'
  | 'stopping'
  | 'done'
  | 'error'

type SwingLogEntry = {
  swingIndex: number
  startMs: number
  endMs: number
  framesCount: number
  receivedAtMs: number
  // 'server' if the lib re-anchored timestamps to start at 0 (Modal
  // batch path), 'fallback' otherwise. Same heuristic the live coach
  // uses for telemetry — see hooks/useLiveCoach.ts detectSwingSource.
  source: 'server' | 'fallback'
}

type CoachCueLogEntry = {
  id: string
  text: string
  sessionMs: number
  swingCount: number
  // Wall-clock time the cue landed in the transcript, used to derive
  // "time since the swing it was about" inline.
  receivedAtMs: number
  // Wall-clock of the most recent swing pushed into the coach BEFORE
  // this cue arrived — captured at the moment the transcript grew so
  // we can render "cue ↑ 3.4s after last swing" without the wall-clock
  // drifting under us.
  msSinceLatestSwing: number | null
}

const DETECT_STATS_BUFFER = 8

// ffmpeg.wasm is a ~30MB blob; lazy-load via the shared CDN so it
// doesn't ship in the main bundle. The "Transcode locally" button
// imports @ffmpeg/ffmpeg dynamically and pulls the core wasm from
// unpkg, which is what the official quickstart recommends. After
// first use the browser caches both, so subsequent transcodes are
// fast(er) — the encoding itself still takes ~30s/min of source.
const FFMPEG_CORE_BASE = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd'

// Try the browser's hardware-accelerated WebCodecs path (mediabunny)
// first. This is what `components/UploadZone.tsx` uses for the live
// upload flow on /analyze, so any clip that "works in /analyze" will
// transcode here too. If WebCodecs isn't supported (Safari < 17,
// older Chromium) or the codec isn't decodable by hardware, fall
// through to ffmpeg.wasm.
async function transcodeViaMediabunnyOrFfmpeg(
  input: File,
  onLog?: (line: string) => void,
): Promise<File> {
  try {
    const { canTranscode, transcodeToH264_720p } = await import('@/lib/videoTranscode')
    if (await canTranscode()) {
      onLog?.('[mediabunny] starting WebCodecs transcode (hardware-accelerated)…')
      const result = await transcodeToH264_720p(input, {
        onProgress: (f) => onLog?.(`[mediabunny] ${Math.round(f * 100)}%`),
      })
      onLog?.(
        `[mediabunny] success: ${result.originalBytes} → ${result.outputBytes} bytes`,
      )
      return result.file
    }
    onLog?.('[mediabunny] not supported on this browser; trying ffmpeg.wasm')
  } catch (err) {
    onLog?.(
      `[mediabunny] failed: ${err instanceof Error ? err.message : String(err)} — trying ffmpeg.wasm`,
    )
  }
  return transcodeToH264WithFfmpegWasm(input, onLog)
}

async function transcodeToH264WithFfmpegWasm(
  input: File,
  onLog?: (line: string) => void,
): Promise<File> {
  const [{ FFmpeg }, { fetchFile, toBlobURL }] = await Promise.all([
    import('@ffmpeg/ffmpeg'),
    import('@ffmpeg/util'),
  ])
  const ffmpeg = new FFmpeg()
  // Pipe ffmpeg's stderr (and stdout) into the page so the user can see
  // what's happening when a transcode silently produces bad output.
  // ffmpeg.wasm's default core ships without HEVC decoder on some
  // builds — when that's the case we'd otherwise produce a 0-byte
  // output and look like we succeeded.
  if (onLog) {
    ffmpeg.on('log', ({ message }: { message: string }) => onLog(message))
  }
  await ffmpeg.load({
    coreURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.js`, 'text/javascript'),
    wasmURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`, 'application/wasm'),
  })
  await ffmpeg.writeFile('input', await fetchFile(input))
  // Strip audio (-an) — Modal extracts pose from video frames only.
  // ultrafast preset because dev iteration; output isn't re-encoded.
  // Fragmented mp4 output. ffmpeg.wasm sometimes Abort()s after
  // encoding all frames but before finalizing the output's moov atom —
  // logged seen in production with H.264 inputs. Fragmented mp4 writes
  // self-contained moof+mdat fragments, so even if the wasm aborts
  // before the final moov flush, the output is still demuxable.
  // empty_moov+default_base_moof+frag_keyframe is the standard fmp4
  // flag combo. faststart guarantees the file plays without seeking
  // to the tail.
  const exitCode = await ffmpeg.exec([
    '-i', 'input',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart+empty_moov+default_base_moof+frag_keyframe',
    '-an',
    'output.mp4',
  ])
  if (exitCode !== 0) {
    throw new Error(
      `ffmpeg exit code ${exitCode} — most often this means the source codec isn't supported by the wasm build (HEVC is excluded from default @ffmpeg/core). Try cloudconvert.com instead, or use Safari.`,
    )
  }
  const out = await ffmpeg.readFile('output.mp4')
  const bytes = out instanceof Uint8Array ? out : new TextEncoder().encode(out)
  if (bytes.byteLength < 1024) {
    throw new Error(
      `ffmpeg produced ${bytes.byteLength} bytes — likely a decode failure. See the log panel for ffmpeg's actual stderr.`,
    )
  }
  return new File([bytes as BlobPart], `${input.name.replace(/\.[^.]+$/, '')}-h264.mp4`, {
    type: 'video/mp4',
  })
}

// Verify a File is actually playable by the browser before we hand
// it to handleReplay. Waits for `canplay` not just `loadedmetadata`
// — a corrupted file can have a parseable moov atom (loadedmetadata
// fires) but fail to decode any frames (canplay never fires). 6s
// budget — fragmented mp4 outputs need a beat longer than plain mp4
// to reach canplay.
function probePlayability(file: File): Promise<{ ok: true } | { ok: false; reason: string }> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const v = document.createElement('video')
    v.muted = true
    v.preload = 'auto'
    const cleanup = () => {
      v.removeEventListener('canplay', onOk)
      v.removeEventListener('error', onErr)
      URL.revokeObjectURL(url)
    }
    const onOk = () => {
      cleanup()
      resolve({ ok: true })
    }
    const onErr = () => {
      const code = v.error?.code
      const codeName: Record<number, string> = {
        3: 'MEDIA_ERR_DECODE',
        4: 'MEDIA_ERR_SRC_NOT_SUPPORTED',
      }
      cleanup()
      resolve({ ok: false, reason: code ? (codeName[code] ?? `code ${code}`) : 'unknown' })
    }
    v.addEventListener('canplay', onOk)
    v.addEventListener('error', onErr)
    v.src = url
    setTimeout(() => {
      cleanup()
      resolve({ ok: false, reason: 'timeout (>6s) — output may be present but undecodable' })
    }, 6000)
  })
}

export default function ReplayDevPage() {
  // -----------------------------------------------------------------
  // File picker + replay state
  // -----------------------------------------------------------------
  const [file, setFile] = useState<File | null>(null)
  const [fileObjectUrl, setFileObjectUrl] = useState<string | null>(null)
  const [replayState, setReplayState] = useState<ReplayState>('idle')
  const [replayError, setReplayError] = useState<string | null>(null)
  const [transcodeState, setTranscodeState] = useState<'idle' | 'running' | 'error'>('idle')
  const [transcodeError, setTranscodeError] = useState<string | null>(null)
  // Tail of ffmpeg's stderr so the user can see why a transcode failed
  // without opening DevTools. Most useful line is usually near the end.
  const [transcodeLog, setTranscodeLog] = useState<string[]>([])
  // Set true right after the transcode-button finishes; the effect
  // below auto-fires handleReplay once React has re-rendered with the
  // new file. Without this the user has to click Replay manually after
  // transcoding, and at least one user reported the manual click did
  // "nothing" — likely because handleReplay's useCallback closure
  // captured the OLD file reference until the next render.
  const [autoReplayPending, setAutoReplayPending] = useState(false)
  // Last time the Replay button click handler actually fired. Surfaced
  // inline as a diagnostic — if the user clicks Replay and this
  // timestamp doesn't update, the click isn't registering at all.
  const [lastReplayClickAt, setLastReplayClickAt] = useState<number | null>(null)
  const [playbackRate, setPlaybackRate] = useState<number>(1)
  const [toggles, setToggles] = useState<FetchToggles>({
    blockExtract: false,
    blockLiveCoach: false,
  })
  const togglesRef = useRef(toggles)
  togglesRef.current = toggles

  // Final session summary, rendered after the source clip ends + stop()
  // settles.
  const [sessionResult, setSessionResult] = useState<{
    durationMs: number
    swingCount: number
    keyframeCount: number
    blobBytes: number
    blobMimeType: string
  } | null>(null)

  // Pipeline status (mirrored from useLiveCapture's onPoseQuality).
  const [poseQuality, setPoseQuality] = useState<PoseQuality | null>(null)

  // Swings (most recent first in the UI).
  const [swings, setSwings] = useState<SwingLogEntry[]>([])
  const swingsRef = useRef<SwingLogEntry[]>([])

  // Last detect stats snapshots — polled at 500ms cadence while replaying,
  // mirroring LiveCapturePanel's debug poll.
  const [detectStats, setDetectStats] = useState<DetectStatsSnapshot[]>([])

  // Coaching cues, derived from useLiveStore.transcript. We mirror them
  // into a local array so we can stamp wall-clock at arrival time.
  const [coachCues, setCoachCues] = useState<CoachCueLogEntry[]>([])

  // -----------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------
  // Hidden source <video> — the file is loaded into this; its
  // captureStream() output is what we feed to useLiveCapture.
  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  // Visible preview <video> — what useLiveCapture.start() attaches the
  // file-derived stream to (and what it draws frames from for inference).
  const previewVideoRef = useRef<HTMLVideoElement | null>(null)
  // Ref into the captured stream so we can stop tracks on cleanup.
  const fileStreamRef = useRef<MediaStream | null>(null)
  // Restore-fn for the getUserMedia monkey-patch installed in handleReplay.
  // Held in a ref so every termination path (EOF, manual stop, start()
  // error, unmount) can reach it. Calling it nulls itself out, so a
  // double-call is a no-op. See the bug fix in the Phase E review: prior
  // versions only restored from the EOF + start()-error paths, leaking
  // the patch across pages whenever the user hit Stop or navigated away.
  const restoreGetUserMediaRef = useRef<(() => void) | null>(null)

  // -----------------------------------------------------------------
  // Live store wiring (reads + writes)
  // -----------------------------------------------------------------
  const transcript = useLiveStore((s) => s.transcript)
  const setTtsEnabled = useLiveStore((s) => s.setTtsEnabled)
  const resetSession = useLiveStore((s) => s.resetSession)
  // Track the last swing arrival wall-clock so cue rows can render
  // "cue 3.4s after the last swing".
  const lastSwingAtRef = useRef<number>(0)

  // Mute TTS for the duration of this dev page. The live coach reads
  // ttsEnabled and calls ttsQueue.mute() when it flips false. We restore
  // on unmount.
  useEffect(() => {
    const prev = useLiveStore.getState().ttsEnabled
    setTtsEnabled(false)
    return () => {
      setTtsEnabled(prev)
    }
  }, [setTtsEnabled])

  // Mirror transcript into local cue log with wall-clock timestamps.
  useEffect(() => {
    setCoachCues((prev) => {
      // Only append the genuinely-new tail of the transcript so we don't
      // re-stamp existing rows (which would corrupt msSinceLatestSwing).
      if (transcript.length <= prev.length) {
        // Likely a transcript reset — drop our local copy too.
        if (transcript.length === 0 && prev.length > 0) return []
        return prev
      }
      const newRows: CoachCueLogEntry[] = []
      const now = performance.now()
      for (let i = prev.length; i < transcript.length; i++) {
        const e = transcript[i]
        newRows.push({
          id: e.id,
          text: e.text,
          sessionMs: e.sessionMs,
          swingCount: e.swingCount,
          receivedAtMs: now,
          msSinceLatestSwing:
            lastSwingAtRef.current > 0 ? now - lastSwingAtRef.current : null,
        })
      }
      return [...prev, ...newRows]
    })
  }, [transcript])

  // -----------------------------------------------------------------
  // Coach + capture hooks
  // -----------------------------------------------------------------
  const coach = useLiveCoach()

  const { start, stop, abort, status, error, isRecording, getLastDetectStats } =
    useLiveCapture({
      onSwing: (swing) => {
        const now = performance.now()
        lastSwingAtRef.current = now
        const entry: SwingLogEntry = {
          swingIndex: swing.swingIndex,
          startMs: swing.startMs,
          endMs: swing.endMs,
          framesCount: swing.frames.length,
          receivedAtMs: now,
          source: detectSwingSource(swing),
        }
        swingsRef.current = [entry, ...swingsRef.current]
        setSwings(swingsRef.current.slice(0, 32))
        coach.pushSwing(swing)
      },
      onPoseQuality: (q) => setPoseQuality(q),
      onStatus: () => {
        // We don't gate on the live store's 'status' here; replayState
        // tracks our local lifecycle. Kept as a no-op listener so any
        // future hook-side state additions don't go unobserved.
      },
    })

  // -----------------------------------------------------------------
  // Force-failure: monkey-patched fetch
  // -----------------------------------------------------------------
  // We install once on mount and read the latest toggles via a ref so
  // toggling a checkbox doesn't reinstall the patch (which could race
  // with an in-flight request). Restore the original fetch on unmount.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const original = window.fetch.bind(window)
    const patched: typeof window.fetch = async (input, init) => {
      const t = togglesRef.current
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : input instanceof Request
              ? input.url
              : ''
      if (t.blockExtract && url.includes('/api/extract')) {
        return new Response(
          JSON.stringify({ error: 'forced-failure (replay dev page)' }),
          { status: 503, headers: { 'Content-Type': 'application/json' } },
        )
      }
      if (t.blockLiveCoach && url.includes('/api/live-coach')) {
        return new Response(
          JSON.stringify({ error: 'forced-failure (replay dev page)' }),
          { status: 503, headers: { 'Content-Type': 'application/json' } },
        )
      }
      return original(input, init)
    }
    window.fetch = patched
    return () => {
      window.fetch = original
    }
  }, [])

  // -----------------------------------------------------------------
  // ONNX detect-stats poller. Same pattern as LiveCapturePanel — 500ms
  // tick, lightweight, only runs while replaying.
  // -----------------------------------------------------------------
  useEffect(() => {
    if (replayState !== 'replaying') return
    const id = window.setInterval(() => {
      const s = getLastDetectStats()
      if (!s) return
      const snapshot: DetectStatsSnapshot = {
        ...s,
        capturedAtMs: performance.now(),
      }
      setDetectStats((prev) => {
        const next = [snapshot, ...prev]
        if (next.length > DETECT_STATS_BUFFER) next.length = DETECT_STATS_BUFFER
        return next
      })
    }, 500)
    return () => window.clearInterval(id)
  }, [replayState, getLastDetectStats])

  // -----------------------------------------------------------------
  // File handler
  // -----------------------------------------------------------------
  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0]
      if (!f) return
      // Revoke any prior URL.
      if (fileObjectUrl) {
        URL.revokeObjectURL(fileObjectUrl)
      }
      const url = URL.createObjectURL(f)
      setFile(f)
      setFileObjectUrl(url)
      setReplayState('idle')
      setReplayError(null)
      setSessionResult(null)
      setSwings([])
      swingsRef.current = []
      setDetectStats([])
      setCoachCues([])
      lastSwingAtRef.current = 0
      // Reset transcript so cue rows from prior replays don't bleed in.
      resetSession()
    },
    [fileObjectUrl, resetSession],
  )

  // -----------------------------------------------------------------
  // Replay start: monkey-patch getUserMedia, mount the file in the
  // hidden <video>, captureStream() it, and drive useLiveCapture.start().
  // -----------------------------------------------------------------
  const handleReplay = useCallback(async () => {
    setLastReplayClickAt(Date.now())
    if (!file || !fileObjectUrl) {
      setReplayError('Pick a file first')
      return
    }
    if (replayState === 'replaying' || replayState === 'starting') return

    setReplayError(null)
    setReplayState('starting')
    setSessionResult(null)
    setSwings([])
    swingsRef.current = []
    setDetectStats([])
    setCoachCues([])
    lastSwingAtRef.current = 0
    resetSession()

    const sourceEl = sourceVideoRef.current
    const previewEl = previewVideoRef.current
    if (!sourceEl || !previewEl) {
      setReplayState('error')
      setReplayError('video element refs not mounted')
      return
    }

    // Load the file into the hidden source element and wait for metadata
    // (videoWidth/videoHeight populated, captureStream() can return tracks).
    try {
      sourceEl.src = fileObjectUrl
      sourceEl.muted = true
      sourceEl.playsInline = true
      sourceEl.loop = false
      sourceEl.playbackRate = playbackRate
      await new Promise<void>((resolve, reject) => {
        const onReady = () => {
          sourceEl.removeEventListener('loadedmetadata', onReady)
          sourceEl.removeEventListener('error', onError)
          resolve()
        }
        const onError = () => {
          sourceEl.removeEventListener('loadedmetadata', onReady)
          sourceEl.removeEventListener('error', onError)
          // Most common failure on this code path is a codec the browser
          // can't decode (HEVC mp4 from an iPhone Camera roll on Chrome,
          // VP9 webm on older Safari, etc.). Surface the underlying
          // MediaError code + a codec-aware hint so the dev page tells
          // the user what to do instead of just "load error".
          const mediaErr = sourceEl.error
          const codeNames: Record<number, string> = {
            1: 'MEDIA_ERR_ABORTED',
            2: 'MEDIA_ERR_NETWORK',
            3: 'MEDIA_ERR_DECODE',
            4: 'MEDIA_ERR_SRC_NOT_SUPPORTED',
          }
          const codeName = mediaErr ? codeNames[mediaErr.code] ?? `code ${mediaErr.code}` : 'unknown'
          const hint =
            mediaErr?.code === 4
              ? ' — likely HEVC (iPhone default) on a non-Safari browser. Re-encode to H.264 via QuickTime/HandBrake, or use Safari.'
              : mediaErr?.code === 3
                ? ' — file decoded partially then errored. May be corrupted; try a different clip.'
                : ''
          const detail = mediaErr?.message ? `: ${mediaErr.message}` : ''
          reject(new Error(`source video load error (${codeName}${detail})${hint}`))
        }
        if (sourceEl.readyState >= 1) {
          resolve()
          return
        }
        sourceEl.addEventListener('loadedmetadata', onReady)
        sourceEl.addEventListener('error', onError)
      })
    } catch (err) {
      setReplayState('error')
      setReplayError(
        err instanceof Error ? err.message : 'failed loading source file',
      )
      return
    }

    // captureStream — Firefox uses mozCaptureStream; both return MediaStream.
    type CapturableVideo = HTMLVideoElement & {
      captureStream?: () => MediaStream
      mozCaptureStream?: () => MediaStream
    }
    const cap = sourceEl as CapturableVideo
    const captureFn = cap.captureStream ?? cap.mozCaptureStream
    if (typeof captureFn !== 'function') {
      setReplayState('error')
      setReplayError('captureStream() not supported on this browser')
      return
    }
    let fileStream: MediaStream
    try {
      fileStream = captureFn.call(sourceEl)
    } catch (err) {
      setReplayState('error')
      setReplayError(
        err instanceof Error ? err.message : 'captureStream() threw',
      )
      return
    }
    fileStreamRef.current = fileStream

    // Start playback BEFORE we hand the stream to useLiveCapture — without
    // this, captureStream's tracks can sit muted until the source plays,
    // and useLiveCapture's videoEl.play() then races against an empty
    // stream. We let the source video drive the timeline.
    try {
      await sourceEl.play()
    } catch (err) {
      setReplayState('error')
      setReplayError(
        err instanceof Error ? err.message : 'source video play() rejected',
      )
      return
    }

    // Monkey-patch getUserMedia for the duration of this replay. The
    // hook calls navigator.mediaDevices.getUserMedia({ video, audio })
    // exactly once during start(); we return our file-derived stream.
    // We restore the original in the EOF path / on stop.
    const md = navigator.mediaDevices as MediaDevices & {
      getUserMedia: typeof navigator.mediaDevices.getUserMedia
    }
    const originalGetUserMedia = md.getUserMedia.bind(md)
    md.getUserMedia = async () => fileStream
    // Park restore in a ref so every termination path (EOF, manual stop,
    // start()-error, unmount) can reach it. Self-nulling makes repeat
    // invocations safe.
    restoreGetUserMediaRef.current = () => {
      try {
        md.getUserMedia = originalGetUserMedia
      } catch {
        // ignore — defensive
      }
      restoreGetUserMediaRef.current = null
    }

    // EOF + error wiring: when the source clip ends, stop the live
    // capture pipeline. doStop reads the ref to restore getUserMedia.
    const onEnded = () => {
      sourceEl.removeEventListener('ended', onEnded)
      void doStop()
    }
    sourceEl.addEventListener('ended', onEnded)

    // Hand off to useLiveCapture. The hook will:
    //  - call navigator.mediaDevices.getUserMedia (returns our patched stream)
    //  - srcObject = stream into the previewEl we pass in
    //  - load the ONNX models
    //  - start MediaRecorder
    //  - drive runOneDetection on requestVideoFrameCallback ticks
    coach.markSessionStart(Date.now())
    coach.primeTts() // no-op since TTS is muted, but kept for parity

    try {
      await start(previewEl, {
        facingMode: 'environment',
        aspect: 'landscape',
      })
      setReplayState('replaying')
    } catch (err) {
      restoreGetUserMediaRef.current?.()
      sourceEl.removeEventListener('ended', onEnded)
      setReplayState('error')
      setReplayError(
        err instanceof Error ? err.message : 'useLiveCapture.start() threw',
      )
    }
  }, [
    file,
    fileObjectUrl,
    playbackRate,
    replayState,
    coach,
    start,
    resetSession,
  ])

  // After the transcode button replaces `file` + `fileObjectUrl`,
  // React re-renders and handleReplay's useCallback closes over the
  // new file. THIS effect (which depends on handleReplay) then
  // auto-fires it so the user doesn't have to click Replay manually.
  // The autoReplayPending guard ensures we only fire once per
  // transcode and only when state is clean ('idle', file present).
  useEffect(() => {
    if (!autoReplayPending) return
    if (replayState !== 'idle' || !file || !fileObjectUrl) return
    setAutoReplayPending(false)
    void handleReplay()
  }, [autoReplayPending, replayState, file, fileObjectUrl, handleReplay])

  // Centralised stop path. Used by the EOF handler and the manual Stop
  // button. Always restores getUserMedia (via ref) and stops file-stream
  // tracks.
  const doStop = useCallback(
    async () => {
      setReplayState('stopping')
      let result: LiveSessionResult | null = null
      try {
        result = await stop()
      } catch (err) {
        setReplayError(
          err instanceof Error ? err.message : 'stop() threw',
        )
      }
      // Stop the file-derived stream's tracks so the source video can
      // be safely teardown'd next.
      if (fileStreamRef.current) {
        for (const t of fileStreamRef.current.getTracks()) {
          try { t.stop() } catch { /* ignore */ }
        }
        fileStreamRef.current = null
      }
      const sourceEl = sourceVideoRef.current
      if (sourceEl) {
        try { sourceEl.pause() } catch { /* ignore */ }
      }
      // Restore the getUserMedia patch installed in handleReplay. Safe
      // to call regardless of which termination path got here first;
      // the ref nulls itself out after running.
      restoreGetUserMediaRef.current?.()
      if (result) {
        setSessionResult({
          durationMs: result.durationMs,
          swingCount: result.swings.length,
          keyframeCount: result.keypoints.frame_count,
          blobBytes: result.blob.size,
          blobMimeType: result.blobMimeType,
        })
      }
      setReplayState('done')
    },
    [stop],
  )

  const handleStopManual = useCallback(() => {
    if (replayState !== 'replaying') return
    void doStop()
  }, [replayState, doStop])

  // Cleanup on unmount: tear down stream, abort capture, revoke URL,
  // and critically — restore navigator.mediaDevices.getUserMedia so the
  // patch doesn't leak across pages (next /live visit would otherwise
  // get the stopped file stream instead of the real camera).
  useEffect(() => {
    return () => {
      try { abort() } catch { /* ignore */ }
      if (fileStreamRef.current) {
        for (const t of fileStreamRef.current.getTracks()) {
          try { t.stop() } catch { /* ignore */ }
        }
        fileStreamRef.current = null
      }
      restoreGetUserMediaRef.current?.()
      if (fileObjectUrl) {
        URL.revokeObjectURL(fileObjectUrl)
      }
    }
    // We deliberately don't include fileObjectUrl in deps — it's
    // recreated per-file-pick which already revokes the prior URL in
    // onFileChange. This effect is purely the unmount path.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [abort])

  // Update playbackRate on the source element live, even mid-replay.
  useEffect(() => {
    const el = sourceVideoRef.current
    if (!el) return
    el.playbackRate = playbackRate
  }, [playbackRate])

  // -----------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------
  const replayDisabled = !file || replayState === 'starting' || replayState === 'replaying' || replayState === 'stopping'

  const pillColor = useMemo(() => {
    switch (poseQuality) {
      case 'good': return 'text-emerald-400'
      case 'weak': return 'text-amber-300'
      case 'wrong-angle': return 'text-amber-300'
      case 'no-body': return 'text-rose-400'
      default: return 'text-white/60'
    }
  }, [poseQuality])

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-6 text-white">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold">Live-pipeline replay (dev)</h1>
        <p className="text-sm text-white/60">
          Pick a local mp4/webm and replay it through the same useLiveCapture
          + useLiveCoach pipeline that powers /live. Audio coaching is muted.
          All status is rendered inline.
        </p>
      </header>

      {/* File picker + replay controls */}
      <section className="rounded-md bg-white/5 p-3 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="file"
            accept="video/mp4,video/webm,video/*"
            onChange={onFileChange}
            className="text-xs"
          />
          {file ? (
            <span className="text-xs text-white/60 font-mono">
              {file.name} · {(file.size / (1024 * 1024)).toFixed(1)}MB
            </span>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleReplay}
            disabled={replayDisabled}
            className="rounded-md bg-emerald-600 px-3 py-2 text-sm font-medium disabled:opacity-40"
          >
            Replay through live pipeline
          </button>
          <button
            onClick={handleStopManual}
            disabled={replayState !== 'replaying'}
            className="rounded-md bg-rose-600 px-3 py-2 text-sm font-medium disabled:opacity-40"
          >
            Stop
          </button>
          <div className="flex items-center gap-2 text-xs">
            <label htmlFor="rate">speed</label>
            <input
              id="rate"
              type="range"
              min={0.5}
              max={2}
              step={0.25}
              value={playbackRate}
              onChange={(e) => setPlaybackRate(Number(e.target.value))}
            />
            <span className="font-mono">{playbackRate.toFixed(2)}×</span>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 text-xs">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={toggles.blockExtract}
              onChange={(e) =>
                setToggles((t) => ({ ...t, blockExtract: e.target.checked }))
              }
            />
            Force /api/extract → 503
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={toggles.blockLiveCoach}
              onChange={(e) =>
                setToggles((t) => ({ ...t, blockLiveCoach: e.target.checked }))
              }
            />
            Force /api/live-coach → 503
          </label>
        </div>

        <div className="text-xs space-y-1">
          <div>
            replayState: <span className="font-mono">{replayState}</span>
            {' · '}
            captureStatus: <span className="font-mono">{status}</span>
            {isRecording ? ' · recording' : null}
            {' · '}
            replayDisabled: <span className="font-mono">{String(replayDisabled)}</span>
          </div>
          {lastReplayClickAt ? (
            <div className="text-white/40">
              last Replay click: <span className="font-mono">{new Date(lastReplayClickAt).toLocaleTimeString()}</span>
              {' '}
              <span className="text-white/30">
                (if you clicked Replay and this didn&apos;t update, the click isn&apos;t registering)
              </span>
            </div>
          ) : null}
          {autoReplayPending ? (
            <div className="text-emerald-300">
              auto-replay pending — will fire on next render with the new file
            </div>
          ) : null}
          {replayError ? (
            <div className="text-rose-400">replay error: {replayError}</div>
          ) : null}
          {error ? (
            <div className="text-rose-400">capture error: {error}</div>
          ) : null}
          {/* Transcode escape hatch: when the source video failed with
              SRC_NOT_SUPPORTED (typically HEVC on a non-Safari browser),
              offer a one-click in-browser transcode to H.264 via
              ffmpeg.wasm. Cost is a ~30MB lazy-load + ~30s/min of source
              encode time. Output replaces the picked file so the user
              can just click Replay again. */}
          {file && replayError && /SRC_NOT_SUPPORTED/.test(replayError) ? (
            <div className="pt-2 space-y-1">
              <button
                type="button"
                disabled={transcodeState === 'running'}
                onClick={async () => {
                  setTranscodeState('running')
                  setTranscodeError(null)
                  setTranscodeLog([])
                  const logBuf: string[] = []
                  try {
                    const out = await transcodeViaMediabunnyOrFfmpeg(file, (line) => {
                      logBuf.push(line)
                      // Keep only the last 40 lines so the panel doesn't
                      // explode for long clips.
                      const tail = logBuf.slice(-40)
                      setTranscodeLog(tail)
                    })
                    // Validate: ffmpeg sometimes "succeeds" (exit 0) but
                    // produces an output the browser still can't decode
                    // — typically when the HEVC decoder in this wasm
                    // build returns garbage frames. Probe before swap.
                    const probe = await probePlayability(out)
                    if (!probe.ok) {
                      throw new Error(
                        `ffmpeg ran but the output isn't browser-playable (${probe.reason}). The wasm build likely couldn't decode this codec. Use cloudconvert.com or Safari.`,
                      )
                    }
                    if (fileObjectUrl) URL.revokeObjectURL(fileObjectUrl)
                    const newUrl = URL.createObjectURL(out)
                    setFile(out)
                    setFileObjectUrl(newUrl)
                    setReplayState('idle')
                    setReplayError(null)
                    setTranscodeState('idle')
                    setAutoReplayPending(true)
                  } catch (err) {
                    setTranscodeState('error')
                    setTranscodeError(err instanceof Error ? err.message : 'transcode failed')
                  }
                }}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs rounded px-3 py-1.5"
              >
                {transcodeState === 'running'
                  ? 'Transcoding (WebCodecs / ffmpeg.wasm fallback)…'
                  : 'Transcode to H.264 (same path /analyze uses)'}
              </button>
              {transcodeError ? (
                <div className="text-rose-400 text-xs">transcode error: {transcodeError}</div>
              ) : null}
              {transcodeLog.length > 0 ? (
                <details className="text-xs text-white/50">
                  <summary className="cursor-pointer">ffmpeg log (last {transcodeLog.length} lines)</summary>
                  <pre className="font-mono text-[10px] leading-tight max-h-40 overflow-y-auto bg-black/40 rounded px-2 py-1 mt-1 whitespace-pre-wrap">
                    {transcodeLog.join('\n')}
                  </pre>
                </details>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      {/* Visible preview video (the one useLiveCapture attaches to) */}
      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Preview (driven by useLiveCapture)</h2>
        <video
          ref={previewVideoRef}
          muted
          playsInline
          className="w-full rounded-md bg-black aspect-video"
        />
        <div className="text-xs text-white/60">
          tracking quality:{' '}
          <span className={`font-mono ${pillColor}`}>
            {poseQuality ?? '(no signal yet)'}
          </span>
        </div>
        {/* Hidden source video — file is loaded here, captureStream() feeds the pipeline. */}
        <video ref={sourceVideoRef} muted playsInline className="hidden" />
      </section>

      {/* Swings */}
      <section className="space-y-2">
        <h2 className="text-lg font-semibold">
          Swings detected ({swings.length})
        </h2>
        <div className="text-xs text-white/60">
          Most recent first. <span className="font-mono">source=server</span>{' '}
          means the Modal batch path successfully replaced on-device frames;
          <span className="font-mono"> fallback</span> means the on-device
          frames remained in place (per-swing extraction failed or was
          blocked by the toggle).
        </div>
        <ul className="space-y-1 text-xs font-mono">
          {swings.map((s) => (
            <li key={s.swingIndex} className="rounded bg-white/5 px-2 py-1">
              #{s.swingIndex} · {Math.round(s.startMs)}ms → {Math.round(s.endMs)}ms
              {' · '}{s.framesCount} frames
              {' · '}<span className={s.source === 'server' ? 'text-emerald-400' : 'text-amber-300'}>{s.source}</span>
            </li>
          ))}
          {swings.length === 0 ? (
            <li className="text-white/40">(no swings yet)</li>
          ) : null}
        </ul>
      </section>

      {/* Coaching transcript */}
      <section className="space-y-2">
        <h2 className="text-lg font-semibold">
          Coaching cues ({coachCues.length})
        </h2>
        <div className="text-xs text-white/60">
          Audio is muted. Each row shows the cue text, the session-time it
          references, and the wall-clock gap from the most recent swing
          when the cue arrived.
        </div>
        <ul className="space-y-2 text-xs">
          {coachCues.map((c) => (
            <li key={c.id} className="rounded bg-white/5 px-2 py-2">
              <div className="font-mono text-white/50">
                t={Math.round(c.sessionMs / 100) / 10}s · {c.swingCount} swings
                {c.msSinceLatestSwing != null
                  ? ` · ${(c.msSinceLatestSwing / 1000).toFixed(1)}s after last swing`
                  : ''}
              </div>
              <div>{c.text}</div>
            </li>
          ))}
          {coachCues.length === 0 ? (
            <li className="text-white/40">(no cues yet)</li>
          ) : null}
        </ul>
      </section>

      {/* Detect stats */}
      <section className="space-y-2">
        <h2 className="text-lg font-semibold">
          ONNX detect stats (last {DETECT_STATS_BUFFER})
        </h2>
        <div className="text-xs text-white/60">
          Refreshed at 2Hz. yolo cnf = top YOLO person score this tick;
          kp avg = mean RTMPose keypoint confidence (drives isFrameConfident);
          inf ms = ONNX inference latency.
        </div>
        <ul className="space-y-1 text-[11px] font-mono">
          {detectStats.map((s, i) => (
            <li
              key={`${s.capturedAtMs}-${i}`}
              className="rounded bg-white/5 px-2 py-1"
            >
              det={s.yoloDetections} · yolo cnf={s.yoloTopScore.toFixed(2)} · kp max={s.kpMaxConf.toFixed(2)} · kp avg={s.kpAvgConf.toFixed(2)} · {s.inferenceMs.toFixed(0)}ms · {s.provider ?? '?'}
            </li>
          ))}
          {detectStats.length === 0 ? (
            <li className="text-white/40">(no stats yet)</li>
          ) : null}
        </ul>
      </section>

      {/* Session result */}
      {sessionResult ? (
        <section className="space-y-2">
          <h2 className="text-lg font-semibold">LiveSessionResult</h2>
          <pre className="bg-black/40 p-2 rounded text-[11px] overflow-x-auto">
            {JSON.stringify(sessionResult, null, 2)}
          </pre>
        </section>
      ) : null}
    </div>
  )
}

// Mirrors the swing-source heuristic in hooks/useLiveCoach.ts. The Modal
// batch extractor re-anchors per-swing timestamps so frames[0].timestamp_ms
// === 0 ⇒ server-extracted; on-device fallback keeps session-relative
// timestamps which are effectively never exactly 0 except for the very
// first swing of the session (acceptable telemetry skew documented there).
function detectSwingSource(swing: StreamedSwing): 'server' | 'fallback' {
  const first = swing.frames[0]
  if (!first) return 'fallback'
  return first.timestamp_ms === 0 ? 'server' : 'fallback'
}
