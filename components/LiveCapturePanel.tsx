'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { upload } from '@vercel/blob/client'
import { useLiveCapture, type LiveSessionResult } from '@/hooks/useLiveCapture'
import { useLiveCoach } from '@/hooks/useLiveCoach'
import { useLiveStore } from '@/store/live'
import { usePoseStore } from '@/store'
import {
  clearOrphanedSession,
  getOrphanedSession,
  saveOrphanedSession,
  type OrphanedSession,
} from '@/lib/liveSessionRecovery'
import LiveSwingCounter from './LiveSwingCounter'

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

interface LiveCapturePanelProps {
  onSessionComplete?: (result: LiveSessionResult) => void
}

function extForMime(mimeType: string): string {
  if (mimeType.startsWith('video/mp4')) return 'mp4'
  if (mimeType.startsWith('video/webm')) return 'webm'
  if (mimeType.startsWith('video/quicktime')) return 'mov'
  return 'bin'
}

type ReviewPhase = 'review' | 'uploading' | 'error' | 'done'

export default function LiveCapturePanel({ onSessionComplete }: LiveCapturePanelProps) {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const reviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const [summary, setSummary] = useState<{ swings: number; durationMs: number } | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  // The recorded session, held until the user picks Save / Discard / Re-record.
  const [pendingResult, setPendingResult] = useState<LiveSessionResult | null>(null)
  // Object URL for the inline review player. Created when the user enters
  // review, revoked when they leave it (or on unmount).
  const [reviewVideoUrl, setReviewVideoUrl] = useState<string | null>(null)
  // Where in the post-stop flow we are: review (just stopped), uploading,
  // error (retry available), or done (handed off to /analyze).
  const [reviewPhase, setReviewPhase] = useState<ReviewPhase>('review')
  // "Waiting for last cue…" UI when awaitInFlight is blocking on an
  // in-flight coach request at Stop time.
  const [waitingForCue, setWaitingForCue] = useState(false)
  // Mount-time orphan detection. If the previous session never finished
  // uploading, offer to resume.
  const [orphan, setOrphan] = useState<OrphanedSession | null>(null)

  const shotType = useLiveStore((s) => s.shotType)
  const setShotType = useLiveStore((s) => s.setShotType)
  const swingCount = useLiveStore((s) => s.swingCount)
  const setSwingCount = useLiveStore((s) => s.setSwingCount)
  const status = useLiveStore((s) => s.status)
  const setStatus = useLiveStore((s) => s.setStatus)
  const setErrorMessage = useLiveStore((s) => s.setErrorMessage)
  const errorMessage = useLiveStore((s) => s.errorMessage)
  const setSessionStartedAtMs = useLiveStore((s) => s.setSessionStartedAtMs)
  const resetSession = useLiveStore((s) => s.resetSession)
  const transcript = useLiveStore((s) => s.transcript)

  const poseSetFramesData = usePoseStore((s) => s.setFramesData)
  const poseSetBlobUrl = usePoseStore((s) => s.setBlobUrl)
  const poseSetLocalVideoUrl = usePoseStore((s) => s.setLocalVideoUrl)
  const poseSetSessionId = usePoseStore((s) => s.setSessionId)
  const poseSetShotType = usePoseStore((s) => s.setShotType)

  const swingCountRef = useRef(0)
  const coach = useLiveCoach()

  // Keep the coach's shot type in sync with the user's selection.
  useEffect(() => {
    coach.setShotType(shotType)
  }, [coach, shotType])

  const { start, stop, abort, status: captureStatus, error: captureError, isRecording, pickedMimeType } = useLiveCapture({
    onSwing: (swing) => {
      swingCountRef.current++
      setSwingCount(swingCountRef.current)
      coach.pushSwing(swing)
    },
    onStatus: (s) => {
      if (s === 'recording') setStatus('recording')
      if (s === 'error') setStatus('error')
      if (s === 'stopping') setStatus('uploading')
      if (s === 'idle') setStatus('idle')
    },
  })

  // Surface capture errors into the store
  useEffect(() => {
    if (captureError) setErrorMessage(captureError)
  }, [captureError, setErrorMessage])

  const handleStart = useCallback(async () => {
    const videoEl = videoRef.current
    if (!videoEl) return
    // Must prime TTS from the user-gesture handler so iOS Safari unlocks
    // speechSynthesis before the first coaching utterance.
    coach.primeTts()
    setErrorMessage(null)
    setSummary(null)
    swingCountRef.current = 0
    setSwingCount(0)
    setStatus('preflight')
    const startedAt = Date.now()
    setSessionStartedAtMs(startedAt)
    coach.markSessionStart(startedAt)
    await start(videoEl)
  }, [coach, setErrorMessage, setSessionStartedAtMs, setStatus, setSwingCount, start])

  // Run the upload + save pipeline against an already-finalized recording.
  // Pulled out of handleStop so the Retry button (after a failure) and the
  // Resume-orphan path can re-fire it without re-recording.
  const runSaveFlow = useCallback(async (
    result: LiveSessionResult,
    saveShotType: string,
  ) => {
    setReviewPhase('uploading')
    setStatus('uploading')
    setErrorMessage(null)

    let blobUrl: string
    try {
      setUploadStatus('Uploading video…')
      const ext = extForMime(result.blobMimeType)
      const blobPath = `live/${Date.now()}-session.${ext}`
      const uploaded = await upload(blobPath, result.blob, {
        access: 'public',
        handleUploadUrl: '/api/upload',
        contentType: result.blobMimeType,
      })
      blobUrl = uploaded.url
    } catch (err) {
      // Persist the recording so the user can retry without losing it.
      void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
      setErrorMessage(err instanceof Error ? err.message : 'Upload failed')
      setUploadStatus(null)
      setStatus('error')
      setReviewPhase('error')
      return
    }

    const batchEventIds = transcript
      .map((e) => e.eventId)
      .filter((id): id is string => typeof id === 'string' && id.length > 0)

    try {
      setUploadStatus('Saving session…')
      const res = await fetch('/api/sessions/live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          blobUrl,
          shotType: saveShotType,
          keypointsJson: result.keypoints,
          swings: result.swings.map((s) => ({
            startFrame: s.startFrameIndex,
            endFrame: s.endFrameIndex,
            startMs: s.startMs,
            endMs: s.endMs,
          })),
          batchEventIds,
        }),
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => 'Save failed')
        void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
        setErrorMessage(`Save failed: ${msg}`)
        setUploadStatus(null)
        setStatus('error')
        setReviewPhase('error')
        return
      }
      const { sessionId } = (await res.json()) as { sessionId: string }

      // Success — clear any prior orphan (this session is safely persisted).
      void clearOrphanedSession()

      // Hydrate usePoseStore so /analyze picks this session up identically to
      // an uploaded clip.
      const objectUrl = URL.createObjectURL(result.blob)
      poseSetFramesData(result.keypoints.frames)
      poseSetBlobUrl(blobUrl)
      poseSetLocalVideoUrl(objectUrl)
      poseSetSessionId(sessionId)
      poseSetShotType(saveShotType)

      setUploadStatus(null)
      setStatus('complete')
      setReviewPhase('done')
      router.replace('/analyze')
    } catch (err) {
      void saveOrphanedSession(result.blob, result.keypoints, result.swings, saveShotType)
      setErrorMessage(err instanceof Error ? err.message : 'Save failed')
      setUploadStatus(null)
      setStatus('error')
      setReviewPhase('error')
    }
  }, [
    poseSetBlobUrl,
    poseSetFramesData,
    poseSetLocalVideoUrl,
    poseSetSessionId,
    poseSetShotType,
    router,
    setErrorMessage,
    setStatus,
    transcript,
  ])

  const handleStop = useCallback(async () => {
    setUploadStatus('Finalizing recording…')
    const result = await stop()
    if (!result) {
      setUploadStatus(null)
      setStatus('idle')
      return
    }

    setSummary({ swings: result.swings.length, durationMs: result.durationMs })
    onSessionComplete?.(result)

    // Wait for any in-flight coach call to settle so its analysis_events row
    // is written before /api/sessions/live backfills session_id. Capped at
    // 5s so a wedged request can't block the player at Stop time.
    if (coach.isRequestInFlight()) {
      setWaitingForCue(true)
      setUploadStatus('Waiting for last cue…')
      await coach.awaitInFlight(5_000)
      setWaitingForCue(false)
    }

    // Hold the recording in review state. The user picks Save / Discard /
    // Re-record next; only Save kicks off the upload + persist flow.
    setPendingResult(result)
    setReviewPhase('review')
    setUploadStatus(null)
    setStatus('idle')
  }, [coach, onSessionComplete, setStatus, stop])

  // Build (and tear down) the inline review player's object URL. We only
  // create one URL per pendingResult so the <video> element stays stable
  // across re-renders.
  useEffect(() => {
    if (!pendingResult) return
    const url = URL.createObjectURL(pendingResult.blob)
    setReviewVideoUrl(url)
    return () => {
      URL.revokeObjectURL(url)
      setReviewVideoUrl(null)
    }
  }, [pendingResult])

  // On mount, see if a previous session never finished uploading. If so,
  // surface the resume prompt — the recording itself is already in IDB.
  useEffect(() => {
    let cancelled = false
    void getOrphanedSession().then((found) => {
      if (cancelled) return
      if (found) setOrphan(found)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const handleSavePending = useCallback(async () => {
    if (!pendingResult) return
    await runSaveFlow(pendingResult, shotType)
  }, [pendingResult, runSaveFlow, shotType])

  const handleDiscardPending = useCallback(() => {
    setPendingResult(null)
    setSummary(null)
    setUploadStatus(null)
    setErrorMessage(null)
    setReviewPhase('review')
    setStatus('idle')
    swingCountRef.current = 0
    setSwingCount(0)
  }, [setErrorMessage, setStatus, setSwingCount])

  const handleRerecordPending = useCallback(() => {
    handleDiscardPending()
    coach.reset()
    resetSession()
  }, [coach, handleDiscardPending, resetSession])

  const handleRetrySave = useCallback(async () => {
    if (!pendingResult) return
    setErrorMessage(null)
    await runSaveFlow(pendingResult, shotType)
  }, [pendingResult, runSaveFlow, setErrorMessage, shotType])

  const handleResumeOrphan = useCallback(async () => {
    if (!orphan) return
    const result: LiveSessionResult = {
      blob: orphan.blob,
      blobMimeType: orphan.blob.type || 'video/webm',
      keypoints: orphan.keypoints,
      swings: orphan.swings,
      durationMs: 0,
    }
    setSummary({ swings: orphan.swings.length, durationMs: 0 })
    setPendingResult(result)
    setReviewPhase('review')
    setOrphan(null)
  }, [orphan])

  const handleDiscardOrphan = useCallback(() => {
    setOrphan(null)
    void clearOrphanedSession()
  }, [])

  // Kept for future "Hard reset" affordance (e.g. global error). The
  // post-stop flow uses the more targeted Discard / Re-record handlers
  // above instead — those preserve the streaming camera state.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _handleReset = useCallback(() => {
    abort()
    coach.reset()
    resetSession()
    swingCountRef.current = 0
    setSummary(null)
    setPendingResult(null)
    setReviewPhase('review')
  }, [abort, coach, resetSession])

  const busy = captureStatus === 'requesting-permissions' || captureStatus === 'initializing' || captureStatus === 'stopping'
  const displaySeconds = summary ? Math.max(1, Math.round(summary.durationMs / 1000)) : 0

  return (
    <div className="space-y-4">
      {/* Shot type picker — locked during a session */}
      <div className="flex gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            disabled={isRecording || busy}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
            } ${isRecording || busy ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Preview / idle card */}
      <div className="relative rounded-2xl border border-white/10 bg-black overflow-hidden aspect-video">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
          playsInline
          muted
        />
        {!isRecording && !busy ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gradient-to-br from-emerald-500/10 to-transparent">
            <div className="text-5xl">🎾</div>
            <p className="text-white font-medium">Lean your phone on anything side-on. Tap Start and play.</p>
            <p className="text-white/50 text-sm text-center max-w-sm px-4">
              Your bag, a bench, the fence — whatever you've got. We coach you every few swings and save the session for later review.
            </p>
          </div>
        ) : null}

        {isRecording ? (
          <div className="absolute top-3 left-3">
            <LiveSwingCounter swingCount={swingCount} />
          </div>
        ) : null}
      </div>

      {/* Resume prompt — appears on mount when an orphaned upload exists. */}
      {orphan && !pendingResult && !isRecording ? (
        <div
          data-testid="resume-orphan-prompt"
          className="rounded-2xl border border-amber-400/30 bg-amber-500/10 px-5 py-4 space-y-3"
        >
          <p className="text-white font-medium">
            We saved a session you didn&apos;t finish uploading. Resume?
          </p>
          <p className="text-white/60 text-sm">
            {orphan.swings.length} {orphan.swings.length === 1 ? 'swing' : 'swings'} detected.
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleResumeOrphan}
              className="bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl px-4 py-2 text-sm transition-all"
            >
              Resume
            </button>
            <button
              onClick={handleDiscardOrphan}
              className="bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-2 text-sm transition-all"
            >
              Discard
            </button>
          </div>
        </div>
      ) : null}

      {/* Controls — hidden once we're in the post-stop review flow so the
          Stop button can't fire a second time on a recording that's
          already finished. */}
      {!pendingResult ? (
        <div className="flex flex-col sm:flex-row gap-3">
          {!isRecording ? (
            <button
              onClick={handleStart}
              disabled={busy}
              className="flex-1 bg-emerald-500 hover:bg-emerald-400 disabled:bg-white/10 disabled:text-white/40 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
            >
              {busy ? 'Starting…' : 'Start'}
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex-1 bg-red-500 hover:bg-red-400 text-white font-bold rounded-2xl px-6 py-4 text-lg transition-all"
            >
              Stop
            </button>
          )}
        </div>
      ) : null}

      {/* Pre-upload review screen. The user just hit Stop; before we burn
          the upload roundtrip + Vercel Blob storage we let them inspect
          the recording and pick Save / Discard / Re-record. */}
      {pendingResult && reviewPhase === 'review' ? (
        <div
          data-testid="review-screen"
          className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 space-y-4"
        >
          <p className="text-white font-semibold text-base">Review your session</p>
          {reviewVideoUrl ? (
            <video
              ref={reviewVideoRef}
              data-testid="review-video"
              src={reviewVideoUrl}
              controls
              playsInline
              className="w-full rounded-xl bg-black"
            />
          ) : null}
          <div className="flex flex-wrap gap-4 text-sm text-white/70">
            <span>
              <span className="text-white font-semibold">{summary?.swings ?? 0}</span>{' '}
              {summary?.swings === 1 ? 'swing' : 'swings'}
            </span>
            <span>
              <span className="text-white font-semibold">{displaySeconds}s</span> duration
            </span>
          </div>
          <div className="flex flex-col sm:flex-row gap-2">
            <button
              onClick={handleSavePending}
              className="flex-1 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl px-4 py-3 transition-all"
            >
              Save
            </button>
            <button
              onClick={handleDiscardPending}
              className="flex-1 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-3 transition-all"
            >
              Discard
            </button>
            <button
              onClick={handleRerecordPending}
              className="flex-1 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl px-4 py-3 transition-all"
            >
              Re-record
            </button>
          </div>
        </div>
      ) : null}

      {errorMessage ? (
        <div
          data-testid="error-box"
          className="rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 text-sm px-4 py-3 space-y-3"
        >
          <p>{errorMessage}</p>
          {pendingResult && reviewPhase === 'error' ? (
            <div className="flex gap-2">
              <button
                onClick={handleRetrySave}
                className="bg-red-500 hover:bg-red-400 text-white font-semibold rounded-lg px-3 py-1.5 text-xs transition-all"
              >
                Retry
              </button>
              <button
                onClick={handleDiscardPending}
                className="bg-white/10 hover:bg-white/20 text-white font-medium rounded-lg px-3 py-1.5 text-xs transition-all"
              >
                Discard
              </button>
            </div>
          ) : null}
        </div>
      ) : null}

      {pickedMimeType && isRecording ? (
        <p className="text-white/40 text-xs">Recording {pickedMimeType}</p>
      ) : null}

      {uploadStatus ? (
        <div
          data-testid="upload-status"
          className="rounded-xl border border-white/10 bg-white/[0.04] text-white text-sm px-4 py-3"
        >
          {waitingForCue ? 'Waiting for last cue…' : uploadStatus}
        </div>
      ) : null}

      {/* Hand-off summary — shown only after the upload+save chain has
          completed and we're navigating to /analyze. */}
      {summary && reviewPhase === 'done' && !uploadStatus ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] px-5 py-4 space-y-1">
          <p className="text-white font-medium">
            {summary.swings} {summary.swings === 1 ? 'swing' : 'swings'} detected in {displaySeconds}s
          </p>
          <p className="text-white/50 text-sm">
            Opening the session in Analyze…
          </p>
        </div>
      ) : null}

      <p className="text-white/40 text-xs leading-relaxed">
        Status: <span className="text-white/60">{status}</span>
        {' · '}
        Any side-on setup works — prop the phone on your bag, a bench, the fence. Just keep your full body in frame. Front and back-of-court angles are too noisy to coach from.
      </p>
    </div>
  )
}
