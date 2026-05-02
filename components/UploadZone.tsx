'use client'

import { useRef, useState, useEffect } from 'react'
import { upload as vercelBlobUpload } from '@vercel/blob/client'
import { uploadVideoResumable, buildObjectPath, type UploadHandle } from '@/lib/supabaseUpload'
import { canTranscode, shouldTranscode, sniffVideoSize, transcodeToH264_720p } from '@/lib/videoTranscode'
import { usePoseStore } from '@/store'
import { usePoseExtractor } from '@/hooks/usePoseExtractor'
import { createPoseDetector } from '@/lib/browserPose'
import { extractPoseViaRailway, RailwayExtractError } from '@/lib/poseExtractionRailway'
import { hashFileInWorker } from '@/lib/contentHash'
import type { ExtractResult } from '@/lib/poseExtraction'
import type { PoseFrame } from '@/lib/supabase'

// Gate the server-side (Railway) extraction path behind an env flag so a
// rollout can be toggled without redeploying code. When unset / false the
// upload flow keeps using in-browser MediaPipe exactly as before.
const USE_RAILWAY_EXTRACT =
  process.env.NEXT_PUBLIC_USE_RAILWAY_EXTRACT === 'true'

// Storage backend toggle. When true: tus → Supabase Storage (resumable,
// survives tab close). When false / unset: legacy Vercel Blob multipart
// (the path that worked before). Default OFF so a deploy without the
// Supabase Storage migration applied (lib/db/011_videos_storage_bucket.sql)
// doesn't 404 every upload. Flip to true once the bucket + RLS are in place.
const USE_SUPABASE_TUS =
  process.env.NEXT_PUBLIC_USE_SUPABASE_TUS === 'true'

// Vercel Blob path mirrors the prior `safeBlobFilename` helper. Only
// used when the tus path is gated off.
function legacySafeBlobFilename(name: string): string {
  return (
    name
      .split(/[\\/]/)
      .pop()!
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .slice(0, 100) || 'upload'
  )
}

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

interface UploadZoneProps {
  onComplete?: (blobUrl: string, frames: PoseFrame[]) => void
}

export default function UploadZone({ onComplete }: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [shotType, setShotType] = useState<string>('forehand')
  const [statusMsg, setStatusMsg] = useState('')
  const [overallProgress, setOverallProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  // Whether the MediaPipe model is currently downloading. Drives the
  // indeterminate-spinner UI so a 15-20s model load doesn't read as a
  // stuck progress bar (root cause of the "stuck at 25%" bug class).
  const [modelLoading, setModelLoading] = useState(false)
  // Last successfully-uploaded file, kept so the retry button can re-run
  // pose extraction without re-uploading to Supabase Storage (saves the
  // user ~30s on every retry).
  const [lastFile, setLastFile] = useState<File | null>(null)
  const [showRetry, setShowRetry] = useState(false)
  // Active upload handle so Cancel can abort the in-flight tus upload
  // (otherwise the chunk PATCH loop keeps running and consumes
  // bandwidth even after the user backs out).
  const uploadHandleRef = useRef<UploadHandle | null>(null)

  const {
    setFramesData,
    setBlobUrl,
    setLocalVideoUrl,
    setProcessing,
    isProcessing,
    setShotType: persistShotType,
    setExtractorBackend,
    setFallbackReason,
    setSessionId,
  } = usePoseStore()

  const { extract, progress: extractProgress, error: extractError, isProcessing: extracting, abort } = usePoseExtractor()

  // Reset only the processing flag when UploadZone mounts fresh (e.g. after back-button).
  // Don't clear framesData/blobUrl/localVideoUrl - those are needed by VideoCanvas.
  useEffect(() => {
    if (isProcessing) {
      setProcessing(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Remap 0..100 extractor progress into the 25..90 band of our overall bar.
  // Bands: transcode 0..10, upload 10..25, extraction 25..90, save 90..100.
  // tus resumes don't restart the bar (each chunk PATCH commits its own
  // offset), so the upload band fills monotonically.
  useEffect(() => {
    if (!extracting) return
    setOverallProgress(25 + Math.round((extractProgress / 100) * 65))
  }, [extractProgress, extracting])

  // Surface extractor errors and arm the retry button so the user can
  // re-run extraction without re-uploading the file.
  useEffect(() => {
    if (extractError) {
      setStatusMsg(extractError)
      setShowRetry(true)
    }
  }, [extractError])

  const processVideo = async (file: File, opts: { skipUpload?: boolean; cachedBlobUrl?: string } = {}) => {
    setBusy(true)
    setProcessing(true)
    setShowRetry(false)
    setOverallProgress(0)

    // 0. Hardware-accelerated transcode (4K60 → 720p30 H.264) before
    //    upload, so the bytes we send over cellular are 5-10× smaller
    //    than what the camera roll has. Skipped for files that are
    //    already small enough (1080p / under 80 MB) and for browsers
    //    where WebCodecs isn't available — both fall through to
    //    uploading the original.
    let uploadFile = file
    if (!opts.skipUpload) {
      try {
        const dims = await sniffVideoSize(file)
        if (shouldTranscode(file, dims?.width) && (await canTranscode())) {
          setStatusMsg('Optimizing video for upload...')
          const result = await transcodeToH264_720p(file, {
            // Transcode owns the 0..10 band of the overall bar.
            onProgress: (frac) => setOverallProgress(Math.round(frac * 10)),
          })
          uploadFile = result.file
          const mb = (n: number) => (n / 1024 / 1024).toFixed(0)
          console.info(
            `[UploadZone] transcoded ${mb(result.originalBytes)} MB → ${mb(result.outputBytes)} MB`,
          )
        }
      } catch (err) {
        // Transcode failed (HEVC decode bug, encoder OOM, codec
        // mismatch, etc). Fall through and upload the original — the
        // worst case is a slower upload, not a failed one.
        console.warn('[UploadZone] transcode failed, uploading original:', err)
      }
    }
    setStatusMsg('Uploading video...')

    // 0b. Kick off the SHA-256 content hash in a Web Worker, in
    //     parallel with the upload below. The hash is over the ORIGINAL
    //     `file` (NOT the post-transcode `uploadFile`). WebCodecs
    //     transcode is non-deterministic across hardware/browser
    //     versions — same source clip on two devices produces different
    //     post-transcode bytes, so hashing post-transcode would defeat
    //     cross-device cache hits. Hashing the source file means two
    //     users uploading the same source clip share the same cache
    //     key. Transcode-param drift is bounded by `model_version`
    //     (bump it when transcode changes) so the cache busts cleanly.
    //     The Promise resolves while the upload is still in flight; we
    //     await it just before /api/extract so the server can short-
    //     circuit on a cache hit. A hashing failure (CSP, missing
    //     subtle.crypto, OOM) is non-fatal: we continue with sha256
    //     unset and the server falls through to a real extraction.
    const sha256Promise: Promise<string | null> = hashFileInWorker(file)
      .catch((err) => {
        console.warn('[UploadZone] sha256 hash failed, cache disabled for this upload:', err)
        return null
      })

    // 1. Upload to Supabase Storage via the tus resumable protocol.
    //    tus persists progress to localStorage so a closed tab, locked
    //    phone, or dropped connection mid-upload resumes from the last
    //    completed 6 MB chunk — the actual UX win over the prior
    //    Vercel-Blob multipart path which lost state on tab close.
    //
    // Retry path: skipUpload=true reuses the cached blob URL from the
    // previous attempt so a failed extraction doesn't re-upload the
    // (potentially large) file. Saves ~30s per retry.
    let blobUrl: string
    if (opts.skipUpload && opts.cachedBlobUrl) {
      blobUrl = opts.cachedBlobUrl
      setBlobUrl(blobUrl)
    } else if (USE_SUPABASE_TUS) {
      // Resumable path — survives tab close, but requires the videos
      // bucket + RLS from migration 011 to exist on Supabase.
      try {
        const objectPath = buildObjectPath(uploadFile.name)
        // Park the publicUrl in a closure since the Promise resolves
        // when start()'s called (handle is ready), not when the upload
        // finishes. We need a separate promise that flips on onSuccess.
        let resolveUploaded: (url: string) => void
        let rejectUploaded: (err: Error) => void
        const uploaded = new Promise<string>((res, rej) => {
          resolveUploaded = res
          rejectUploaded = rej
        })
        const handle = await uploadVideoResumable(uploadFile, objectPath, {
          // Map 0..1 progress into the 10..25 band of the overall bar
          // (transcode owns 0..10). Extraction picks up at 25..90 via
          // the extracting effect below.
          onProgress: (frac) => setOverallProgress(10 + Math.round(frac * 15)),
          onSuccess: (url) => resolveUploaded(url),
          onError: (err) => rejectUploaded(err),
        })
        uploadHandleRef.current = handle
        blobUrl = await uploaded
        uploadHandleRef.current = null
        setBlobUrl(blobUrl)
      } catch (err) {
        console.error('[UploadZone] Supabase tus upload failed:', err)
        setStatusMsg('Upload failed. Please try again.')
        setShowRetry(true)
        setProcessing(false)
        setBusy(false)
        uploadHandleRef.current = null
        return
      }
    } else {
      // Legacy Vercel Blob path. No resumability across tabs, but
      // unblocks testing while the Supabase Storage migration is
      // pending. The transcode step above still runs first, so the
      // upload is the smaller post-transcode payload either way.
      try {
        const blobPath = `videos/${Date.now()}-${legacySafeBlobFilename(uploadFile.name)}`
        const blob = await vercelBlobUpload(blobPath, uploadFile, {
          access: 'public',
          handleUploadUrl: '/api/upload',
          contentType: uploadFile.type,
        })
        blobUrl = blob.url
        setBlobUrl(blobUrl)
      } catch (err) {
        console.error('[UploadZone] Vercel Blob upload failed:', err)
        setStatusMsg('Upload failed. Please try again.')
        setShowRetry(true)
        setProcessing(false)
        setBusy(false)
        return
      }
    }

    // Cache the file + blob URL on the component so the retry button
    // can call back into processVideo without re-uploading.
    // Cache the *transcoded* file (or the original on the skip-transcode
    // path) so retry skips the transcode + upload steps and only re-runs
    // extraction.
    setLastFile(uploadFile)

    let result: ExtractResult | null = null
    let usedRailway = false
    let pendingSessionId: string | null = null
    let railwayFailReason: string | null = null

    // Railway path: create the session row FIRST (so Railway has something
    // to write back to), then ask Railway to extract, then poll. On any
    // failure we fall through to the browser MediaPipe path below — the
    // user should never see an upload die because Railway had a bad day.
    if (USE_RAILWAY_EXTRACT) {
      setStatusMsg('Queuing server-side extraction...')
      setOverallProgress(20)
      try {
        const pendingRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ blobUrl, shotType }),
        })
        if (!pendingRes.ok) {
          // Surface the underlying error detail (e.g. Supabase RLS message,
          // 'Missing service key' env failure) so the chip shows *why*,
          // not just the HTTP status. Falls back to the raw body text if
          // the response isn't JSON.
          let detail = ''
          try {
            const errBody = await pendingRes.json()
            detail = errBody?.detail || errBody?.error || ''
          } catch {
            try {
              detail = await pendingRes.text()
            } catch {
              detail = ''
            }
          }
          throw new Error(
            `create-pending failed: ${pendingRes.status}${detail ? ` (${detail})` : ''}`,
          )
        }
        const pendingBody = await pendingRes.json()
        pendingSessionId = pendingBody.sessionId
        // Mirror the new session id into usePoseStore so downstream
        // surfaces that need it (e.g. /api/baselines/from-swing's
        // server-side trim lookup) can read it from the store. Without
        // this, "Save as baseline" surfaced "Save unavailable: missing
        // session id" because nothing else in the upload pipeline
        // wrote sessionId back into the store.
        if (pendingSessionId) setSessionId(pendingSessionId)

        setStatusMsg('Analyzing pose on the server...')
        // Hand the sha256 (or null) to the Railway extractor so the
        // /api/extract route can short-circuit on cache hit without
        // talking to Railway. The hash usually lands well before the
        // upload finishes — we just await whatever's resolved.
        const sha256 = (await sha256Promise) ?? undefined
        result = await extractPoseViaRailway({
          sessionId: pendingSessionId!,
          blobUrl,
          sha256,
          onProgress: (pct) => setOverallProgress(20 + Math.round((pct / 100) * 70)),
        })
        usedRailway = true
      } catch (err) {
        // Any failure falls back to in-browser extraction silently.
        // Log the reason so we can tell 'Railway not configured' (normal
        // during staging) from 'Railway errored out' (needs attention).
        // Concatenate reason + the underlying Error.message so the
        // diagnostic chip shows BOTH the category ('error-status') and
        // the actual Railway error text ('OOM', 'CUDA out of memory',
        // 'YOLO failed: ...'). The category alone is rarely actionable.
        if (err instanceof RailwayExtractError) {
          console.info(
            '[UploadZone] Railway path unavailable, falling back to browser:',
            err.reason, err.message,
          )
          railwayFailReason =
            err.message && err.message !== err.reason
              ? `${err.reason}: ${err.message}`
              : err.reason
        } else {
          console.error('[UploadZone] Railway path errored, falling back to browser:', err)
          railwayFailReason = err instanceof Error ? err.message : 'unknown'
        }
        result = null
      }
    }

    // Browser MediaPipe fallback (or primary path when the feature flag
    // is off). Only runs when Railway didn't produce a result —
    // Railway-success users skip the 16MB MediaPipe WASM download
    // entirely. Model load only happens here (lazily) so the first-paint
    // experience for Railway users stays fast.
    if (!result) {
      setOverallProgress(15)
      setStatusMsg('Loading pose model...')
      // Explicit model-load step. First-time users wait ~5-15s while
      // YOLO11n + RTMPose-m weights download (then cache via the
      // loadModel layer). Indeterminate spinner reads as "actively
      // loading", not "stuck progress bar". We instantiate + dispose a
      // detector here purely to warm the cache; the real extraction
      // call below creates its own detector against the now-warm cache.
      setModelLoading(true)
      try {
        const warmup = await createPoseDetector()
        warmup.dispose()
      } catch {
        setStatusMsg('Failed to load pose model. Check your connection and tap Retry.')
        setShowRetry(true)
        setModelLoading(false)
        setProcessing(false)
        setBusy(false)
        return
      }
      setModelLoading(false)

      setOverallProgress(25)
      setStatusMsg('Analyzing pose from video...')
      result = await extract(uploadFile)
      if (!result) {
        // Hook handled status/error; just release the processing lock.
        // showRetry is already set by the extractError effect.
        setProcessing(false)
        setBusy(false)
        return
      }
      // If we ended up here AFTER attempting Railway, the user's actual
      // pose data is browser-mediapipe but they were *expecting* server
      // extraction — distinguish so the diagnostic chip can warn (red)
      // rather than just informing (yellow).
      if (USE_RAILWAY_EXTRACT) {
        result = { ...result, extractorBackend: 'rtmpose-browser-fallback' }
      }
    }

    setOverallProgress(95)
    setStatusMsg('Saving analysis...')

    const keypointsJson = {
      fps_sampled: result.fps,
      frame_count: result.frames.length,
      frames: result.frames,
    }

    // When Railway did the work, it already wrote the keypoints to the
    // pending session row. We only POST /api/sessions for the browser
    // path (or on Railway fallback, to upsert the browser output into
    // the pending row).
    if (!usedRailway) {
      try {
        const sessionId =
          pendingSessionId ?? usePoseStore.getState().sessionId
        const sessRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, blobUrl, shotType, keypointsJson }),
        })
        if (!sessRes.ok) {
          console.error('Failed to save session:', sessRes.status, await sessRes.text())
          setStatusMsg('Warning: analysis complete but failed to save session.')
        } else {
          // Browser-path session id has to be captured from the
          // response — pure-browser flow has no pendingSessionId
          // because it skipped the Railway pending-row create. Without
          // this write, usePoseStore.sessionId stays null and the
          // "Save as baseline" CTA reports a missing-session error.
          try {
            const sessBody = await sessRes.json()
            const newSessionId =
              typeof sessBody?.sessionId === 'string' ? sessBody.sessionId : null
            if (newSessionId) setSessionId(newSessionId)
          } catch {
            // Non-JSON response is unexpected here but recoverable —
            // the warning surface above covered the failure case.
          }
        }
      } catch (err) {
        console.error('Failed to save session:', err)
        setStatusMsg('Warning: analysis complete but failed to save session.')
      }
    }

    // Hand the object URL off to the store for playback (store revokes old on replace).
    // On the Railway path we don't have a local object URL; create one from
    // the original File so playback still works without re-downloading
    // from Vercel Blob.
    // Local playback uses the transcoded file when we transcoded — saves
    // a re-fetch from Supabase for instant scrubbing, and the file is
    // already the smaller version the user will see in comparison too.
    const playbackUrl = usedRailway ? URL.createObjectURL(uploadFile) : result.objectUrl
    setLocalVideoUrl(playbackUrl)
    setFramesData(result.frames)
    setExtractorBackend(result.extractorBackend)
    setFallbackReason(
      result.extractorBackend === 'rtmpose-browser-fallback' ? railwayFailReason : null,
    )
    persistShotType(shotType)
    setOverallProgress(100)
    setStatusMsg(
      `Done! Analyzed ${result.frames.length} frames${usedRailway ? ' (server)' : ''}.`,
    )
    setProcessing(false)
    setBusy(false)
    onComplete?.(blobUrl, result.frames)
  }

  const handleFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      setStatusMsg('Please upload a video file.')
      return
    }
    processVideo(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleRetry = () => {
    if (!lastFile) return
    // Reuse the cached blob URL so we don't re-upload a 100MB clip
    // every time the user retries an extraction failure.
    const cached = usePoseStore.getState().blobUrl ?? undefined
    processVideo(lastFile, { skipUpload: !!cached, cachedBlobUrl: cached ?? undefined })
  }

  const handleCancel = () => {
    // Abort the in-flight tus upload first if there is one — otherwise
    // the chunk-PATCH loop keeps spinning until the connection times
    // out, burning cellular data while the user has already moved on.
    uploadHandleRef.current?.abort()
    uploadHandleRef.current = null
    abort()
    setStatusMsg('Cancelled.')
    setShowRetry(true)
    setProcessing(false)
    setBusy(false)
    setModelLoading(false)
  }

  const processing = busy || extracting

  return (
    <div className="space-y-4 text-ink">
      {/* Shot type selector */}
      <div className="flex gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-clay text-cream'
                : 'bg-ink/5 text-ink/70 hover:bg-ink/10 hover:text-ink'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Drop zone — sits inside a cream-bg parent on the analyze
          page, so all colors are ink-based for contrast. Hard corners
          per the redesign; clay accents replace the old emerald. */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !processing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed p-12 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-clay bg-clay/10'
            : 'border-ink/20 hover:border-ink/40 hover:bg-ink/5'
        } ${processing ? 'cursor-not-allowed opacity-80' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
          }}
        />

        {processing ? (
          <div className="space-y-4">
            <div className="text-4xl">🎾</div>
            <p className="text-ink font-medium">{statusMsg}</p>
            {modelLoading ? (
              // Indeterminate animated bar during the MediaPipe model
              // download. Honest signal: "actively loading" rather than
              // "stuck at 25%". The pulse stripe slides infinitely.
              <div className="w-full bg-ink/10 h-2 overflow-hidden relative">
                <div
                  className="absolute inset-y-0 w-1/3 bg-clay"
                  style={{
                    animation: 'upload-zone-pulse 1.4s ease-in-out infinite',
                  }}
                />
                <style>{`
                  @keyframes upload-zone-pulse {
                    0%, 100% { left: 0%; opacity: 0.5; }
                    50% { left: 66.6%; opacity: 1; }
                  }
                `}</style>
              </div>
            ) : (
              <div className="w-full bg-ink/10 h-2 overflow-hidden">
                <div
                  className="h-full bg-clay transition-all duration-300"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
            )}
            <p className="text-ink/60 text-sm">
              {modelLoading ? 'Loading Model (One-Time, ~10s)' : `${overallProgress}%`}
            </p>
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleCancel()
              }}
              className="text-xs text-ink/60 hover:text-ink underline transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-5xl">🎬</div>
            <p className="text-ink text-lg font-medium">
              Drop Your Swing Video Here
            </p>
            <p className="text-ink/60 text-sm">
              MP4, MOV, WebM · Max 200MB · Select Shot Type Above First
            </p>
            {statusMsg && (
              <p className={`${/fail|error|please|cancel/i.test(statusMsg) ? 'text-red-600' : 'text-clay'} text-sm font-medium`}>{statusMsg}</p>
            )}
            {showRetry && lastFile && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleRetry()
                }}
                className="px-4 py-2 rounded-full bg-clay hover:bg-[#c4633f] text-cream text-sm font-medium transition-colors"
              >
                Retry Analysis
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
