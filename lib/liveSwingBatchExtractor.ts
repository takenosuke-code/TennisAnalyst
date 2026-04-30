'use client'

// Phase E batch extractor: takes the per-batch swing windows the live
// coach already buffers, splices them into ONE sub-clip, uploads it to
// Vercel Blob, and asks Modal (via /api/extract) to run RTMPose on it.
// The flat per-batch keypoint array that comes back is split per swing
// using the offsets we tracked during splice, then re-anchored so each
// swing's frames start at timestamp_ms = 0 / frame_index = 0.
//
// This module is the production library code that hooks/useLiveCapture
// (worker E3) and hooks/useLiveCoach (worker E4) compose with. It's
// intentionally framework-agnostic — no React, no zustand — so the
// integration workers can wire it up to whatever buffering scheme they
// land on.
//
// Phase E6 update — mp4box demux + remux for iOS Safari compatibility.
// Real-device testing (`app/dev/fmp4-spike`) on iPhone Safari showed
// chunks #1+ from MediaRecorder fail standalone playback ("operation not
// supported"). iOS produces a non-fragmented mp4 prefix (moov in chunk
// #0 only), so the previous "concat raw chunks per swing window" path
// produced malformed mp4 bytes that Modal would reject. Modal's reject
// fell back to E3's on-device frames, so /live still worked, but Phase
// E benefits never materialised on iPhones.
//
// Fix: concatenate ALL chunks into one ArrayBuffer (this prefix IS a
// valid mp4 thanks to chunk #0's moov), parse it with mp4box.js,
// extract the video samples whose presentation time falls in any swing
// window, and emit a fresh fragmented mp4 containing only those
// samples. Audio is dropped — Modal extracts pose from video frames
// only, so audio bytes would just inflate the upload.
//
// Memory/perf budget: mp4box parses 30s of typical phone-recording
// (~5-10MB) in <200ms on a recent phone; 5min of recording (~50-80MB)
// in ~1-2s. Both are run on a worker tick, not in the swing-detect hot
// path, so the cost is invisible to the camera preview. Don't try to
// "optimize" without measuring on a real device — premature
// optimization here has historically broken iOS playback semantics.
//
// Browser compat: mp4box.js is pure JS, ESM, ~50KB minified. Works on
// iOS Safari 17+, Chrome (any), Firefox 110+. The dynamic import below
// keeps it out of the SSR bundle (lib/liveSwingBatchExtractor.ts is
// imported by hooks that could in theory be tree-shaken into a server
// component).

import { upload } from '@vercel/blob/client'
import type { PoseFrame } from '@/lib/supabase'

// =====================================================================
// Public types — E3/E4 import these. Keep the shape stable.
// =====================================================================

export interface BatchExtractSwingInput {
  /** Monotonic per session. Used to key the per-swing split-back result. */
  swingIndex: number
  /** Start of the swing window in ms within the recording (media clock). */
  startMs: number
  /** End of the swing window in ms within the recording (media clock). */
  endMs: number
}

export interface BatchExtractRequest {
  swings: BatchExtractSwingInput[]
  /** MediaRecorder fragments accumulated so far (in chronological order). */
  blobChunks: Blob[]
  blobMimeType: string
  /** Wall-clock or media-clock anchor of chunks[0] (ms). */
  blobStartMs: number
}

export interface BatchExtractPerSwingResult {
  swingIndex: number
  /** Frames re-anchored so timestamp_ms = 0 at the swing's startMs. */
  frames: PoseFrame[]
  /** Null on success; reason string on per-swing failure. */
  failureReason: string | null
}

export interface BatchExtractResult {
  perSwing: BatchExtractPerSwingResult[]
  /** Wall-clock duration for the whole batch operation (ms). */
  totalDurationMs: number
}

export interface BatchExtractOptions {
  abortSignal?: AbortSignal
  onProgress?: (pct: number) => void
}

// =====================================================================
// Internal types
// =====================================================================

/**
 * The result of splicing 1+ swing windows out of the chunk buffer into
 * a single concatenated Blob. We keep per-swing offsets *within the
 * spliced clip* so we can split keypoints back per swing after Modal
 * returns a flat frame array for the whole sub-clip.
 */
interface SplicedSubClip {
  blob: Blob
  /**
   * One entry per swing, parallel to the input swing array. Offsets are
   * timestamps within the spliced sub-clip (0-based), NOT the original
   * recording. swingStartMs is the original recording-relative start of
   * the swing — needed so we can re-anchor the output frames back to
   * timestamp_ms = 0 at the swing's start.
   */
  perSwingOffsets: Array<{
    swingIndex: number
    /** Where this swing starts within the spliced sub-clip (ms). */
    spliceStartMs: number
    /** Where this swing ends within the spliced sub-clip (ms). */
    spliceEndMs: number
    /** Original recording-relative start (ms) — used for the re-anchor. */
    originalStartMs: number
  }>
}

const KNOWN_VIDEO_EXT: Record<string, string> = {
  'video/mp4': 'mp4',
  'video/webm': 'webm',
  'video/quicktime': 'mov',
}

function extForMime(mimeType: string): string {
  const base = mimeType.split(';')[0]?.trim().toLowerCase() ?? ''
  return KNOWN_VIDEO_EXT[base] ?? 'bin'
}

/**
 * Cheap check that the buffer's first 12 bytes look like an mp4 ftyp
 * box: bytes [4..8) === 'ftyp'. The size prefix at [0..4) and the brand
 * at [8..12) are not checked — we only need to know whether mp4box has
 * a chance of parsing this as ISO BMFF. If not, fall back to the dumb
 * raw-concat path so unit-test smoke fixtures (which pass tiny Uint8Arrays
 * that don't claim to be mp4) keep working.
 */
function looksLikeMp4(buf: ArrayBuffer): boolean {
  if (buf.byteLength < 12) return false
  const view = new Uint8Array(buf, 4, 4)
  // 'f' 't' 'y' 'p' = 0x66 0x74 0x79 0x70
  return view[0] === 0x66 && view[1] === 0x74 && view[2] === 0x79 && view[3] === 0x70
}

// =====================================================================
// Splice helper — fallback (raw concat) path
// =====================================================================
//
// Used when (a) chunks aren't recognisable as mp4 (unit-test fixtures
// pass synthetic 1-byte Blobs), or (b) the mp4box demux/remux path
// throws unexpectedly. The fallback is the legacy behaviour: walk the
// chunks under a uniform-duration assumption, emit the intersecting
// chunks raw, and produce per-swing offsets via running counter. Modal
// will reject the resulting bytes when chunks aren't actually
// fragmented, but E3's per-swing failure path catches that and falls
// back to on-device frames — so /live keeps working even on the
// fallback.

function fallbackRawConcatSplice(
  chunks: Blob[],
  mimeType: string,
  swings: BatchExtractSwingInput[],
  blobStartMs: number,
): SplicedSubClip {
  const lastEnd = Math.max(...swings.map((s) => s.endMs))
  const totalDurationMs = Math.max(1, lastEnd - blobStartMs)
  const perChunkMs = totalDurationMs / chunks.length

  const chunkRanges: Array<{ start: number; end: number }> = chunks.map((_, i) => ({
    start: blobStartMs + i * perChunkMs,
    end: blobStartMs + (i + 1) * perChunkMs,
  }))

  const splicedBlobs: Blob[] = []
  const perSwingOffsets: SplicedSubClip['perSwingOffsets'] = []
  let runningSpliceMs = 0

  for (const swing of swings) {
    const swingStart = swing.startMs
    const swingEnd = swing.endMs
    const spliceStartMs = runningSpliceMs

    for (let i = 0; i < chunks.length; i++) {
      const range = chunkRanges[i]
      if (!range) continue
      const intersects = range.end > swingStart && range.start < swingEnd
      if (intersects) {
        splicedBlobs.push(chunks[i] as Blob)
      }
    }

    const swingDurationMs = Math.max(0, swingEnd - swingStart)
    runningSpliceMs += swingDurationMs

    perSwingOffsets.push({
      swingIndex: swing.swingIndex,
      spliceStartMs,
      spliceEndMs: spliceStartMs + swingDurationMs,
      originalStartMs: swingStart,
    })
  }

  return {
    blob: new Blob(splicedBlobs, { type: mimeType }),
    perSwingOffsets,
  }
}

// =====================================================================
// Splice helper — mp4box demux/remux (production path)
// =====================================================================

/**
 * Splice the swing windows from `chunks` into ONE concatenated mp4 sub-clip
 * Blob using mp4box.js to demux + remux. This is the production path that
 * works on iOS Safari (where MediaRecorder produces classic, not fragmented,
 * mp4 — so naive concat-by-time produces malformed bytes).
 *
 * Algorithm:
 *   1. Concatenate ALL provided chunks into one Blob and read as
 *      ArrayBuffer. This prefix is a valid mp4 because chunk #0
 *      contains the moov.
 *   2. Parse with mp4box.appendBuffer + start. Pick the first video
 *      track.
 *   3. For each video sample, compute its presentation time
 *      (cts / timescale * 1000) in ms. Mark samples whose time falls
 *      in ANY requested swing window.
 *   4. Group selected samples by swing → contiguous runs (a swing
 *      window picks consecutive samples by construction, so each swing
 *      is one run).
 *   5. Mutate sample.dts/cts so each swing's samples start at the
 *      desired contiguous output time (spliceStartMs in track-timescale
 *      units) — preserves inter-sample spacing within the swing.
 *   6. Build the init segment via initializeSegmentation().
 *   7. For each swing's run, call createFragment(track_id, runStart,
 *      runEnd, sharedStream) to append a moof+mdat. mp4box's createMoof
 *      uses sample[0].dts to set tfdt.baseMediaDecodeTime, so our
 *      mutation in step 5 anchors the fragment correctly.
 *   8. Concat init buffer + fragments stream → output Blob.
 *
 * Audio decision: dropped. Modal's pose extractor reads video frames
 * only via OpenCV. Including audio just inflates upload bytes for no
 * downstream benefit. setSegmentOptions is called only on the video
 * track, and the moov rebuilt by initializeSegmentation only contains
 * fragmented tracks — audio falls out naturally.
 */
async function mp4boxSplice(
  chunks: Blob[],
  mimeType: string,
  swings: BatchExtractSwingInput[],
): Promise<SplicedSubClip> {
  // Dynamic import keeps mp4box.js out of any server bundles. The lib
  // file declares 'use client', but Next can still tree-shake imports
  // into server graphs if a server component touches anything from
  // this module. Lazy-loading mp4box on first call is cheap (~50KB
  // gzipped, browser-only path).
  const MP4Box = await import('mp4box')

  const fullBlob = new Blob(chunks, { type: mimeType })
  const fullBuf = await fullBlob.arrayBuffer()

  // ISOFile expects an MP4BoxBuffer (ArrayBuffer subclass with a
  // fileStart property). MP4BoxBuffer.fromArrayBuffer wraps an
  // existing ArrayBuffer with fileStart=0 in one call.
  const mp4Buffer = MP4Box.MP4BoxBuffer.fromArrayBuffer(fullBuf, 0)

  const file = MP4Box.createFile()

  // mp4box parses lazily as buffers are appended. Capture errors so we
  // can fall back gracefully.
  let parseError: string | null = null
  file.onError = (_module, message) => {
    parseError = message
  }

  // Append the full buffer in one shot, then flush so any final boxes
  // are processed.
  file.appendBuffer(mp4Buffer)
  file.flush()

  if (parseError) {
    throw new Error(`mp4box parse error: ${parseError}`)
  }

  const info = file.getInfo()
  if (!info || !info.videoTracks || info.videoTracks.length === 0) {
    throw new Error('mp4box: no video tracks found')
  }

  const videoTrack = info.videoTracks[0]
  if (!videoTrack) throw new Error('mp4box: video track missing')

  const trackId = videoTrack.id
  const timescale = videoTrack.timescale

  const trakInternal = (file as unknown as {
    getTrackById: (id: number) => { samples: Array<{ dts: number; cts: number; duration: number; size: number }> } | undefined
  }).getTrackById(trackId)
  if (!trakInternal) throw new Error('mp4box: getTrackById returned nothing')

  const allSamples = trakInternal.samples
  if (!allSamples || allSamples.length === 0) {
    throw new Error('mp4box: video track has no samples')
  }

  // Step 3 — find sample indices in each swing window.
  //
  // Swings are evaluated in input order. perSwingOffsets is built in
  // parallel: each swing's spliceStartMs is the running counter of
  // previously-packed durations. spliceEndMs is computed from the
  // selected sample count + their durations rather than from the raw
  // (endMs - startMs) window, so split-back math matches what mp4box
  // actually emitted.
  const perSwingOffsets: SplicedSubClip['perSwingOffsets'] = []
  const swingSampleRanges: Array<{ first: number; last: number }> = []
  let runningSpliceMs = 0

  for (const swing of swings) {
    const startMs = swing.startMs
    const endMs = swing.endMs

    let firstIdx = -1
    let lastIdx = -1
    for (let i = 0; i < allSamples.length; i++) {
      const s = allSamples[i]
      if (!s) continue
      const cts_ms = (s.cts / timescale) * 1000
      if (cts_ms >= startMs && cts_ms < endMs) {
        if (firstIdx === -1) firstIdx = i
        lastIdx = i
      } else if (firstIdx !== -1) {
        // Past the window — samples within a window are contiguous, so
        // we can stop walking.
        break
      }
    }

    if (firstIdx === -1 || lastIdx === -1) {
      // No samples in this swing's window. Record an empty offset so
      // splitFramesPerSwing can flag the swing as 'no-frames-in-range'.
      perSwingOffsets.push({
        swingIndex: swing.swingIndex,
        spliceStartMs: runningSpliceMs,
        spliceEndMs: runningSpliceMs,
        originalStartMs: swing.startMs,
      })
      swingSampleRanges.push({ first: -1, last: -1 })
      continue
    }

    // Sum sample durations in [firstIdx, lastIdx] for the actual
    // emitted window length in track-timescale units.
    let durationTimescale = 0
    for (let i = firstIdx; i <= lastIdx; i++) {
      durationTimescale += allSamples[i]?.duration ?? 0
    }
    const durationMs = (durationTimescale / timescale) * 1000

    perSwingOffsets.push({
      swingIndex: swing.swingIndex,
      spliceStartMs: runningSpliceMs,
      spliceEndMs: runningSpliceMs + durationMs,
      originalStartMs: swing.startMs,
    })
    swingSampleRanges.push({ first: firstIdx, last: lastIdx })
    runningSpliceMs += durationMs
  }

  // If no swing matched any sample, return an empty blob so the caller
  // routes everything to the per-swing 'empty-subclip' fallback.
  const anyMatched = swingSampleRanges.some((r) => r.first !== -1)
  if (!anyMatched) {
    return { blob: new Blob([], { type: mimeType }), perSwingOffsets }
  }

  // Step 5 — mutate selected sample dts/cts so swings pack contiguously
  // in the output timeline. We snapshot the originals first so we can
  // restore them if downstream code (or future revisions) re-reads the
  // file post-remux. This is defensive — currently nothing else touches
  // `file` after createFragment returns.
  const originalDtsCts: Array<{ idx: number; dts: number; cts: number }> = []
  for (let s = 0; s < swings.length; s++) {
    const range = swingSampleRanges[s]
    const offset = perSwingOffsets[s]
    if (!range || range.first === -1 || !offset) continue
    const baseDts = Math.round((offset.spliceStartMs * timescale) / 1000)
    const firstSample = allSamples[range.first]
    if (!firstSample) continue
    const dtsShift = baseDts - firstSample.dts
    for (let i = range.first; i <= range.last; i++) {
      const sample = allSamples[i]
      if (!sample) continue
      originalDtsCts.push({ idx: i, dts: sample.dts, cts: sample.cts })
      sample.dts += dtsShift
      sample.cts += dtsShift
    }
  }

  // Step 6 — set segmentation options for the video track only (drops
  // audio) and build the init segment.
  const fileWithSegment = file as unknown as {
    setSegmentOptions: (id: number, user: unknown, opts: { nbSamples?: number }) => void
    initializeSegmentation: () => Array<{ id: number; user: unknown; buffer: ArrayBuffer }>
    createFragment: (track_id: number, sampleStart: number, sampleEnd: number, existingStream: unknown) => unknown
    onSegment: (id: number, user: unknown, buffer: ArrayBuffer, nextSample: number, last: boolean) => void
  }

  // No-op onSegment — we drive segmentation manually via createFragment.
  fileWithSegment.onSegment = () => undefined
  fileWithSegment.setSegmentOptions(trackId, null, { nbSamples: 1_000_000 })

  const initSegments = fileWithSegment.initializeSegmentation()
  const initSeg = initSegments.find((s) => s.id === trackId)
  if (!initSeg) throw new Error('mp4box: initializeSegmentation returned no init for video track')

  // Step 7 — emit one fragment per swing's sample run. mp4box's
  // DataStream is dynamically resizing; we share a single stream across
  // all createFragment calls so the moof+mdat boxes are appended in
  // order.
  const DataStreamCtor = (MP4Box as unknown as { DataStream: new () => { buffer: ArrayBuffer } }).DataStream
  const stream = new DataStreamCtor()

  for (let s = 0; s < swings.length; s++) {
    const range = swingSampleRanges[s]
    if (!range || range.first === -1) continue
    // mp4box uses 1-based sample numbers in createFragment; sample[0]
    // in the array is sample number 1 in the trun's accounting. But
    // looking at the implementation (createFragment loops for (i=sampleStart;
    // i<=sampleEnd; i++) and calls getSample(trak, i) which indexes
    // trak.samples[i] directly), the function uses 0-based array
    // indices. So we pass the array indices as-is.
    fileWithSegment.createFragment(trackId, range.first, range.last, stream)
  }

  // Step 8 — concatenate init segment + fragments stream into a final
  // Blob. mp4box's stream.buffer is the underlying ArrayBuffer holding
  // all written bytes; init is a separate ArrayBuffer.
  const outBlob = new Blob([initSeg.buffer, stream.buffer], { type: 'video/mp4' })

  // Restore the originals (defensive — see snapshot above).
  for (const orig of originalDtsCts) {
    const sample = allSamples[orig.idx]
    if (!sample) continue
    sample.dts = orig.dts
    sample.cts = orig.cts
  }

  return { blob: outBlob, perSwingOffsets }
}

// =====================================================================
// Splice helper — public entry point
// =====================================================================

/**
 * Splice the swing windows from `chunks` into ONE concatenated sub-clip
 * Blob. Tracks per-swing offsets within the spliced clip so split-back
 * works after Modal returns a flat keypoint array.
 *
 * Two implementation paths:
 *   - mp4box demux + remux (production). Handles classic-mp4 input
 *     from iOS Safari MediaRecorder by parsing the source file,
 *     selecting samples by presentation time, and emitting a fresh
 *     fragmented mp4 with the selected samples packed contiguously.
 *   - Raw concat fallback. Used when input bytes don't look like mp4
 *     (unit-test smoke fixtures) or when mp4box throws. Walks chunks
 *     under a uniform-duration assumption and emits the intersecting
 *     bytes raw. Not robust on real recordings; included to keep the
 *     existing test suite passing as a smoke test.
 */
export async function sliceToSubClip(
  chunks: Blob[],
  mimeType: string,
  swings: BatchExtractSwingInput[],
  blobStartMs: number,
): Promise<SplicedSubClip> {
  if (chunks.length === 0 || swings.length === 0) {
    return {
      blob: new Blob([], { type: mimeType }),
      perSwingOffsets: [],
    }
  }

  // Cheap mp4 sniff. Synthetic test Blobs (Uint8Array([1])) trip this
  // and route to the fallback. Real MediaRecorder output starts with
  // an ftyp box at offset 4.
  const fullBlob = new Blob(chunks, { type: mimeType })
  let fullBuf: ArrayBuffer
  try {
    fullBuf = await fullBlob.arrayBuffer()
  } catch {
    return fallbackRawConcatSplice(chunks, mimeType, swings, blobStartMs)
  }

  if (!looksLikeMp4(fullBuf)) {
    return fallbackRawConcatSplice(chunks, mimeType, swings, blobStartMs)
  }

  try {
    return await mp4boxSplice(chunks, mimeType, swings)
  } catch (err) {
    // mp4box can throw on malformed inputs we can't anticipate. Falling
    // back to raw concat is strictly better than aborting the swing —
    // E3's per-swing failure path catches Modal rejection downstream.
    if (typeof console !== 'undefined') {
      console.warn('[liveSwingBatchExtractor] mp4box splice failed, falling back to raw concat:', err)
    }
    return fallbackRawConcatSplice(chunks, mimeType, swings, blobStartMs)
  }
}

// =====================================================================
// Split-back helper
// =====================================================================

/**
 * Given a flat array of frames returned by Modal for the spliced
 * sub-clip, slice per-swing arrays out by the spliceStartMs/spliceEndMs
 * offsets and re-anchor each swing's frames so timestamp_ms = 0 at the
 * swing's start. Re-indexes frame_index from 0 within the swing.
 *
 * Per-swing failure: a swing that yields zero frames in its range gets
 * `failureReason = 'no-frames-in-range'` and `frames: []`. Other swings
 * are unaffected.
 */
export function splitFramesPerSwing(
  flatFrames: PoseFrame[],
  perSwingOffsets: SplicedSubClip['perSwingOffsets'],
): BatchExtractPerSwingResult[] {
  return perSwingOffsets.map((offset) => {
    // Inclusive on start, exclusive on end — the canonical half-open
    // interval. A frame whose timestamp lands exactly on the next
    // swing's spliceStartMs belongs to the next swing.
    const inRange = flatFrames.filter(
      (f) => f.timestamp_ms >= offset.spliceStartMs && f.timestamp_ms < offset.spliceEndMs,
    )
    if (inRange.length === 0) {
      return {
        swingIndex: offset.swingIndex,
        frames: [],
        failureReason: 'no-frames-in-range',
      }
    }
    const reAnchored = inRange.map((f, i) => ({
      ...f,
      frame_index: i,
      // Re-anchor to swing-relative timing so consumers don't have to
      // know about the splice. timestamp_ms is the offset from the
      // swing's start (0-based).
      timestamp_ms: f.timestamp_ms - offset.spliceStartMs,
    }))
    return {
      swingIndex: offset.swingIndex,
      frames: reAnchored,
      failureReason: null,
    }
  })
}

// =====================================================================
// Modal caller
// =====================================================================

const POLL_INTERVAL_MS = 1000
// 10 minutes — same budget as poseExtractionRailway.ts. A batch of 4
// swings on a cold Modal container can need 60-120s; the 10min cap
// gives 5x headroom.
const EXTRACTION_TIMEOUT_MS = 600_000

class BatchExtractError extends Error {
  constructor(message: string, public reason: string) {
    super(message)
    this.name = 'BatchExtractError'
  }
}

/**
 * POST the spliced-blob URL to /api/extract and poll until Modal
 * returns the keypoints. Returns the flat frame array.
 *
 * TODO(E3): This is a thinner caller than `extractPoseViaRailway`
 * because that helper expects a sessionId tied to a real DB row that
 * Modal writes back into. The batch path doesn't have one — we want
 * the keypoints in-memory, not persisted to a session row. If E3 wires
 * the batch path through a session row anyway, replace this with a
 * direct call to `extractPoseViaRailway`. Until then this duplicates
 * the polling shape — keep the two in sync if the polling protocol
 * changes.
 */
async function extractKeypointsForBatch(
  blobUrl: string,
  signal: AbortSignal | undefined,
): Promise<PoseFrame[]> {
  if (signal?.aborted) throw new BatchExtractError('aborted before start', 'aborted')

  const queueRes = await fetch('/api/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blobUrl, mode: 'batch-ephemeral' }),
    signal,
  })

  if (queueRes.status === 503) {
    throw new BatchExtractError('Modal not configured', 'not-configured')
  }
  if (!queueRes.ok) {
    throw new BatchExtractError(`Queue rejected: HTTP ${queueRes.status}`, 'queue-failed')
  }

  // Two response shapes are possible from /api/extract:
  //   1. Synchronous: { frames: [...] } — Modal already finished.
  //   2. Async: { jobId: '...' } — poll a status endpoint.
  // The current /api/extract path is fire-and-poll keyed on sessionId,
  // but the batch-ephemeral mode (TODO above) is expected to return
  // frames inline once it lands. For now we accept either shape.
  const body = (await queueRes.json()) as {
    frames?: PoseFrame[]
    jobId?: string
    status?: string
    keypoints_json?: { frames: PoseFrame[] }
  }
  if (Array.isArray(body.frames)) return body.frames
  if (body.keypoints_json?.frames) return body.keypoints_json.frames

  if (!body.jobId) {
    throw new BatchExtractError('Queue accepted but returned no frames or jobId', 'queue-failed')
  }

  // Poll loop.
  const deadline = Date.now() + EXTRACTION_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new BatchExtractError('aborted during poll', 'aborted')

    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, POLL_INTERVAL_MS)
      signal?.addEventListener(
        'abort',
        () => {
          clearTimeout(t)
          reject(new BatchExtractError('aborted during poll sleep', 'aborted'))
        },
        { once: true },
      )
    })

    const pollRes = await fetch(`/api/extract?jobId=${encodeURIComponent(body.jobId)}`, { signal })
    if (!pollRes.ok) continue
    const pollBody = (await pollRes.json()) as {
      status?: string
      frames?: PoseFrame[]
      keypoints_json?: { frames: PoseFrame[] }
      error_message?: string
    }
    if (pollBody.status === 'complete') {
      const frames = pollBody.frames ?? pollBody.keypoints_json?.frames
      if (frames) return frames
      throw new BatchExtractError('Complete but no frames in body', 'error-status')
    }
    if (pollBody.status === 'error') {
      throw new BatchExtractError(pollBody.error_message ?? 'Modal error', 'error-status')
    }
  }
  throw new BatchExtractError('timeout waiting for Modal', 'timeout')
}

// =====================================================================
// Public entry point
// =====================================================================

/**
 * Splice the swing windows out of the in-progress recording, upload
 * the spliced clip to Vercel Blob, ask Modal to extract keypoints, and
 * return per-swing keypoint arrays re-anchored to swing-local time.
 *
 * Failure semantics:
 *   - A failure of the WHOLE batch (upload error, Modal hard-fail) is
 *     surfaced as every swing failing with the same reason. The caller
 *     can then fall back to on-device angles for the whole batch.
 *   - A per-swing failure (zero frames in that swing's range, or Modal
 *     reports an error for that range) leaves other swings successful.
 */
export async function extractBatchedSwingKeypoints(
  req: BatchExtractRequest,
  opts?: BatchExtractOptions,
): Promise<BatchExtractResult> {
  const startedAt = Date.now()
  const { abortSignal, onProgress } = opts ?? {}

  // Splice up front so we have the offset table even if everything
  // downstream fails — the caller can still report which swings were
  // attempted.
  const subClip = await sliceToSubClip(req.blobChunks, req.blobMimeType, req.swings, req.blobStartMs)

  if (subClip.blob.size === 0 || subClip.perSwingOffsets.length === 0) {
    return {
      perSwing: req.swings.map((s) => ({
        swingIndex: s.swingIndex,
        frames: [],
        failureReason: 'empty-subclip',
      })),
      totalDurationMs: Date.now() - startedAt,
    }
  }

  onProgress?.(10)

  // Upload to Vercel Blob via the same `/api/upload` flow LiveCapturePanel uses.
  let blobUrl: string
  try {
    const ext = extForMime(req.blobMimeType)
    const blobPath = `live/swings/${Date.now()}-batch.${ext}`
    const uploaded = await upload(blobPath, subClip.blob, {
      access: 'public',
      handleUploadUrl: '/api/upload',
      contentType: req.blobMimeType,
    })
    blobUrl = uploaded.url
  } catch (err) {
    const reason = err instanceof Error ? `upload-failed: ${err.message}` : 'upload-failed'
    return {
      perSwing: subClip.perSwingOffsets.map((o) => ({
        swingIndex: o.swingIndex,
        frames: [],
        failureReason: reason,
      })),
      totalDurationMs: Date.now() - startedAt,
    }
  }

  onProgress?.(40)

  // Hit Modal.
  let flatFrames: PoseFrame[]
  try {
    flatFrames = await extractKeypointsForBatch(blobUrl, abortSignal)
  } catch (err) {
    const reason =
      err instanceof BatchExtractError
        ? `modal-${err.reason}`
        : err instanceof Error
          ? `modal-error: ${err.message}`
          : 'modal-error'
    return {
      perSwing: subClip.perSwingOffsets.map((o) => ({
        swingIndex: o.swingIndex,
        frames: [],
        failureReason: reason,
      })),
      totalDurationMs: Date.now() - startedAt,
    }
  }

  onProgress?.(90)

  const perSwing = splitFramesPerSwing(flatFrames, subClip.perSwingOffsets)

  onProgress?.(100)

  return {
    perSwing,
    totalDurationMs: Date.now() - startedAt,
  }
}

// Exported for tests; not part of the public API contract.
export { BatchExtractError }
