'use client'

// Phase E SPIKE TEST — fragmented-MP4 MediaRecorder probe.
//
// Purpose: gate Phase E (Modal-during-live) by confirming on real phones
// that:
//   1. MediaRecorder can produce fMP4 chunks via
//      `mimeType: 'video/mp4;codecs=avc1'` with a small `timeslice`.
//   2. Each individual chunk (or a 3-chunk concat) plays back as a
//      standalone `video/mp4` blob — i.e. the moov atom is actually
//      fragmented into each chunk, not stuck in chunk #0 only.
//   3. /api/extract accepts a short (~3s) clip and queues a Modal job
//      against it without rejecting on min-duration.
//
// All output is rendered inline so the user can run this on a phone
// without DevTools. Dev-only: not linked from the main nav.
//
// Out of scope (intentionally): polling Supabase for the queued job's
// keypoints. The fresh sessionId we send doesn't have a row in
// user_sessions, so the Railway background task will silently no-op the
// write-back. The synchronous { status: 'queued' } response from
// /api/extract is the entire signal we need: if Railway rejects on
// min-duration, the proxy returns 502 and we render the error.
// Verifying the keypoints round-trip end-to-end is E2/E3's job once the
// real session row is in the loop.

import { useEffect, useRef, useState } from 'react'
import { upload } from '@vercel/blob/client'

type Chunk = {
  index: number
  blob: Blob
  size: number
  mimeType: string
  timestamp: number
}

type ChunkPlayResult = {
  chunkIndex: number
  state: 'pending' | 'playing' | 'ok' | 'error'
  message?: string
}

const CANDIDATE_MIMES = [
  'video/mp4;codecs=avc1',
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
]

function extForMime(mimeType: string): string {
  if (mimeType.startsWith('video/mp4')) return 'mp4'
  if (mimeType.startsWith('video/webm')) return 'webm'
  return 'bin'
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n}B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`
  return `${(n / (1024 * 1024)).toFixed(2)}MB`
}

export default function FMP4SpikePage() {
  const [userAgent, setUserAgent] = useState('(loading)')
  const [supportedMimes, setSupportedMimes] = useState<Record<string, boolean>>({})
  const [activeMime, setActiveMime] = useState<string | null>(null)
  const [recState, setRecState] = useState<'idle' | 'requesting' | 'recording' | 'stopped' | 'error'>('idle')
  const [recError, setRecError] = useState<string | null>(null)
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [chunkPlay, setChunkPlay] = useState<Record<number, ChunkPlayResult>>({})
  const [uploadState, setUploadState] = useState<{
    state: 'idle' | 'uploading' | 'extracting' | 'done' | 'error'
    elapsedMs?: number
    response?: unknown
    error?: string
    blobUrl?: string
    sessionId?: string
  }>({ state: 'idle' })

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunkIndexRef = useRef(0)
  const chunksRef = useRef<Chunk[]>([])
  const previewVideoRef = useRef<HTMLVideoElement | null>(null)
  const playbackVideoRef = useRef<HTMLVideoElement | null>(null)
  const lastPlayUrlRef = useRef<string | null>(null)

  // Probe MediaRecorder mimeType support + log userAgent on mount.
  useEffect(() => {
    setUserAgent(typeof navigator !== 'undefined' ? navigator.userAgent : '(no navigator)')
    const result: Record<string, boolean> = {}
    if (typeof MediaRecorder !== 'undefined') {
      for (const m of CANDIDATE_MIMES) {
        try {
          result[m] = MediaRecorder.isTypeSupported(m)
        } catch {
          result[m] = false
        }
      }
    }
    setSupportedMimes(result)
  }, [])

  // Cleanup any object URLs + stream on unmount.
  useEffect(() => {
    return () => {
      if (lastPlayUrlRef.current) {
        URL.revokeObjectURL(lastPlayUrlRef.current)
        lastPlayUrlRef.current = null
      }
      if (streamRef.current) {
        for (const t of streamRef.current.getTracks()) t.stop()
        streamRef.current = null
      }
    }
  }, [])

  async function startRecording() {
    setRecError(null)
    setChunks([])
    setChunkPlay({})
    chunksRef.current = []
    chunkIndexRef.current = 0
    setRecState('requesting')

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      })
    } catch (err) {
      setRecError(err instanceof Error ? err.message : 'getUserMedia failed')
      setRecState('error')
      return
    }
    streamRef.current = stream

    if (previewVideoRef.current) {
      previewVideoRef.current.srcObject = stream
      void previewVideoRef.current.play().catch(() => {
        // Autoplay can fail on iOS without a touch — not fatal for the spike.
      })
    }

    // Pick the first supported mime in priority order. Surface the choice.
    let chosen: string | null = null
    for (const m of CANDIDATE_MIMES) {
      if (supportedMimes[m]) {
        chosen = m
        break
      }
    }
    if (!chosen) {
      setRecError('No supported MediaRecorder mimeType on this device')
      setRecState('error')
      return
    }
    setActiveMime(chosen)

    let recorder: MediaRecorder
    try {
      recorder = new MediaRecorder(stream, { mimeType: chosen })
    } catch (err) {
      setRecError(err instanceof Error ? err.message : 'MediaRecorder ctor failed')
      setRecState('error')
      return
    }
    recorderRef.current = recorder

    recorder.ondataavailable = (e) => {
      if (!e.data || e.data.size === 0) return
      const chunk: Chunk = {
        index: chunkIndexRef.current++,
        blob: e.data,
        size: e.data.size,
        mimeType: e.data.type || chosen!,
        timestamp: Date.now(),
      }
      chunksRef.current = [...chunksRef.current, chunk]
      setChunks(chunksRef.current)
    }
    recorder.onerror = (e) => {
      const err = (e as unknown as { error?: Error }).error
      setRecError(err?.message ?? 'MediaRecorder error event')
      setRecState('error')
    }
    recorder.onstop = () => {
      setRecState('stopped')
    }

    recorder.start(1000) // 1s timeslice → fMP4 fragments every ~1s
    setRecState('recording')
  }

  function stopRecording() {
    const r = recorderRef.current
    if (r && r.state !== 'inactive') {
      r.stop()
    }
    if (streamRef.current) {
      for (const t of streamRef.current.getTracks()) t.stop()
      streamRef.current = null
    }
  }

  async function tryPlayChunk(chunk: Chunk) {
    setChunkPlay((prev) => ({
      ...prev,
      [chunk.index]: { chunkIndex: chunk.index, state: 'playing' },
    }))

    // Wrap the chunk in a fresh Blob with `video/mp4` so the <video>
    // element doesn't get fed the raw MediaRecorder MIME (which can
    // include codec parameters the element rejects).
    const playMime = chunk.mimeType.startsWith('video/mp4') ? 'video/mp4' : 'video/webm'
    const standalone = new Blob([chunk.blob], { type: playMime })
    if (lastPlayUrlRef.current) {
      URL.revokeObjectURL(lastPlayUrlRef.current)
    }
    const url = URL.createObjectURL(standalone)
    lastPlayUrlRef.current = url

    const v = playbackVideoRef.current
    if (!v) {
      setChunkPlay((prev) => ({
        ...prev,
        [chunk.index]: {
          chunkIndex: chunk.index,
          state: 'error',
          message: 'No <video> element',
        },
      }))
      return
    }
    v.src = url
    v.muted = true
    v.playsInline = true

    let settled = false
    const onPlaying = () => {
      if (settled) return
      settled = true
      cleanup()
      setChunkPlay((prev) => ({
        ...prev,
        [chunk.index]: {
          chunkIndex: chunk.index,
          state: 'ok',
          message: `playing (${playMime})`,
        },
      }))
    }
    const onError = () => {
      if (settled) return
      settled = true
      cleanup()
      const mediaErr = v.error
      setChunkPlay((prev) => ({
        ...prev,
        [chunk.index]: {
          chunkIndex: chunk.index,
          state: 'error',
          message: mediaErr ? `MediaError code=${mediaErr.code}` : 'play() rejected',
        },
      }))
    }
    function cleanup() {
      v?.removeEventListener('playing', onPlaying)
      v?.removeEventListener('error', onError)
    }
    v.addEventListener('playing', onPlaying)
    v.addEventListener('error', onError)

    try {
      await v.play()
      // Some browsers fire 'playing' synchronously, but if we got past
      // play() without throwing we still wait for 'playing' to confirm
      // decode actually started. 4s budget — short clips usually fire
      // within ~100ms.
      setTimeout(() => {
        if (!settled) {
          settled = true
          cleanup()
          setChunkPlay((prev) => ({
            ...prev,
            [chunk.index]: {
              chunkIndex: chunk.index,
              state: 'error',
              message: 'play() resolved but no "playing" event in 4s — likely undecodable standalone',
            },
          }))
        }
      }, 4000)
    } catch (err) {
      if (settled) return
      settled = true
      cleanup()
      setChunkPlay((prev) => ({
        ...prev,
        [chunk.index]: {
          chunkIndex: chunk.index,
          state: 'error',
          message: err instanceof Error ? err.message : 'play() threw',
        },
      }))
    }
  }

  async function uploadLast3() {
    if (chunks.length === 0) {
      setUploadState({ state: 'error', error: 'No chunks recorded yet' })
      return
    }
    const last3 = chunks.slice(-3)
    const containerMime = activeMime?.startsWith('video/mp4') ? 'video/mp4' : 'video/webm'
    const bundle = new Blob(
      last3.map((c) => c.blob),
      { type: containerMime },
    )
    const sessionId = crypto.randomUUID()
    const startTs = performance.now()
    setUploadState({ state: 'uploading', sessionId })

    let blobUrl: string
    try {
      const ext = extForMime(containerMime)
      const blobPath = `live-spike/${Date.now()}-spike.${ext}`
      const uploaded = await upload(blobPath, bundle, {
        access: 'public',
        handleUploadUrl: '/api/upload',
        contentType: containerMime,
      })
      blobUrl = uploaded.url
    } catch (err) {
      setUploadState({
        state: 'error',
        error: `Upload failed: ${err instanceof Error ? err.message : String(err)}`,
        elapsedMs: Math.round(performance.now() - startTs),
        sessionId,
      })
      return
    }

    setUploadState((prev) => ({ ...prev, state: 'extracting', blobUrl, sessionId }))

    try {
      const res = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blobUrl, sessionId }),
      })
      const text = await res.text()
      let parsed: unknown
      try {
        parsed = JSON.parse(text)
      } catch {
        parsed = { raw: text }
      }
      const elapsed = Math.round(performance.now() - startTs)
      if (!res.ok) {
        setUploadState({
          state: 'error',
          error: `extract returned ${res.status}`,
          response: parsed,
          elapsedMs: elapsed,
          blobUrl,
          sessionId,
        })
        return
      }
      setUploadState({
        state: 'done',
        response: parsed,
        elapsedMs: elapsed,
        blobUrl,
        sessionId,
      })
    } catch (err) {
      setUploadState({
        state: 'error',
        error: err instanceof Error ? err.message : 'extract fetch threw',
        elapsedMs: Math.round(performance.now() - startTs),
        blobUrl,
        sessionId,
      })
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-6 text-white">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold">fMP4 MediaRecorder spike</h1>
        <p className="text-sm text-white/60">
          Phase E gate test. Verifies fragmented-MP4 chunks are individually
          playable and that /api/extract accepts a short clip. All output is
          rendered on-page — no DevTools needed.
        </p>
      </header>

      <section className="rounded-md bg-white/5 p-3 text-xs space-y-1">
        <div>
          <span className="text-white/50">userAgent: </span>
          <span className="font-mono break-all">{userAgent}</span>
        </div>
        <div className="pt-1">
          <span className="text-white/50">MediaRecorder mimeType support:</span>
        </div>
        <ul className="font-mono text-[11px]">
          {CANDIDATE_MIMES.map((m) => (
            <li key={m}>
              {supportedMimes[m] ? 'YES' : 'no '} — {m}
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={startRecording}
            disabled={recState === 'recording' || recState === 'requesting'}
            className="rounded-md bg-emerald-600 px-3 py-2 text-sm font-medium disabled:opacity-40"
          >
            Start recording
          </button>
          <button
            onClick={stopRecording}
            disabled={recState !== 'recording'}
            className="rounded-md bg-rose-600 px-3 py-2 text-sm font-medium disabled:opacity-40"
          >
            Stop
          </button>
        </div>
        <div className="text-xs text-white/70">
          state: <span className="font-mono">{recState}</span>
          {activeMime ? (
            <>
              {' · active mime: '}
              <span className="font-mono">{activeMime}</span>
            </>
          ) : null}
          {recError ? (
            <div className="text-rose-400 mt-1">error: {recError}</div>
          ) : null}
        </div>
        <video
          ref={previewVideoRef}
          muted
          playsInline
          className="w-full rounded-md bg-black aspect-video"
        />
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Chunks ({chunks.length})</h2>
        <div className="text-xs text-white/60">
          MediaRecorder emits a chunk every ~1s. Each row tries to play the
          chunk as a standalone Blob — green = the chunk has its own moov
          atom (fMP4 success), red = it depends on the prefix (classic MP4
          behavior).
        </div>
        <video
          ref={playbackVideoRef}
          muted
          playsInline
          controls
          className="w-full rounded-md bg-black aspect-video"
        />
        <ul className="space-y-1 text-xs">
          {chunks.map((c) => {
            const r = chunkPlay[c.index]
            return (
              <li
                key={c.index}
                className="flex items-center justify-between gap-2 rounded bg-white/5 px-2 py-1"
              >
                <span className="font-mono">
                  #{c.index} · {fmtBytes(c.size)} · {c.mimeType || '(no type)'}
                </span>
                <span className="flex items-center gap-2">
                  {r ? (
                    <span
                      className={
                        r.state === 'ok'
                          ? 'text-emerald-400'
                          : r.state === 'error'
                            ? 'text-rose-400'
                            : 'text-amber-300'
                      }
                    >
                      {r.state}
                      {r.message ? `: ${r.message}` : ''}
                    </span>
                  ) : null}
                  <button
                    onClick={() => tryPlayChunk(c)}
                    className="rounded bg-white/10 px-2 py-1 text-[11px]"
                  >
                    Play this chunk alone
                  </button>
                </span>
              </li>
            )
          })}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold">Upload last 3s to /api/extract</h2>
        <div className="text-xs text-white/60">
          Bundles the most recent 3 chunks into a single Blob, uploads via
          @vercel/blob/client, then POSTs to /api/extract with a fresh uuid
          sessionId. Only the synchronous queue ack is rendered — the
          background extraction writes back to a non-existent session row
          and silently no-ops, which is fine for the gate test. Look for
          status=queued (good) or 502/503 (rejected — investigate
          /api/extract logs).
        </div>
        <button
          onClick={uploadLast3}
          disabled={chunks.length === 0 || uploadState.state === 'uploading' || uploadState.state === 'extracting'}
          className="rounded-md bg-sky-600 px-3 py-2 text-sm font-medium disabled:opacity-40"
        >
          Upload last 3s to /api/extract
        </button>
        <div className="text-xs space-y-1">
          <div>
            state: <span className="font-mono">{uploadState.state}</span>
            {uploadState.elapsedMs != null ? ` · ${uploadState.elapsedMs}ms total` : ''}
          </div>
          {uploadState.sessionId ? (
            <div>
              sessionId: <span className="font-mono break-all">{uploadState.sessionId}</span>
            </div>
          ) : null}
          {uploadState.blobUrl ? (
            <div>
              blobUrl: <span className="font-mono break-all">{uploadState.blobUrl}</span>
            </div>
          ) : null}
          {uploadState.error ? (
            <div className="text-rose-400">error: {uploadState.error}</div>
          ) : null}
          {uploadState.response ? (
            <pre className="bg-black/40 p-2 rounded text-[11px] overflow-x-auto">
              {JSON.stringify(uploadState.response, null, 2)}
            </pre>
          ) : null}
        </div>
      </section>
    </div>
  )
}
