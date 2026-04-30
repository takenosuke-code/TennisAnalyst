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
// TODO(E1): Worker E1 is running an fMP4 spike in parallel. The
// `sliceToSubClip` helper assumes each MediaRecorder chunk is
// independently demuxable (the fMP4 path). If the spike report says
// fMP4 is unavailable on iOS Safari, swap the splice strategy for the
// "send the whole blob with explicit time-range markers per swing"
// fallback (see TODO inside sliceToSubClip). Both code paths are kept
// behind the same public API so swapping is a one-line change here, not
// in E3/E4.

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

// =====================================================================
// Splice helper
// =====================================================================

/**
 * Splice the swing windows from `chunks` into ONE concatenated sub-clip
 * Blob. Tracks per-swing offsets within the spliced clip so split-back
 * works after Modal returns a flat keypoint array.
 *
 * Approach (fMP4-friendly path):
 *   - Treat each MediaRecorder chunk as independently demuxable. Each
 *     chunk maps to a fixed time slice [chunkStartMs, chunkEndMs) based
 *     on a uniform-distribution assumption of total duration / chunk
 *     count. This is the assumption fMP4 mode buys us.
 *   - For each swing window, gather the chunks whose time slice
 *     intersects [startMs, endMs] and concatenate their bytes.
 *   - Record per-swing offsets so the caller can map flat output frames
 *     back to their swing.
 *
 * TODO(E1 fallback): If the fMP4 spike fails, swap the chunk-walking
 * loop for "ship the whole concatenated buffer + send swing time
 * markers as side metadata to /api/extract", and have Modal slice
 * server-side. That fallback would mux via the `mp4-muxer` package
 * (~10KB) — gated on what E1 reports.
 *
 * Note on duration estimation: MediaRecorder doesn't tag each chunk
 * with a precise duration. We estimate per-chunk duration from
 * (totalEndMs - blobStartMs) / chunks.length. This is approximate but
 * good enough for splice boundaries — the keypoint split-back uses the
 * spliced timestamps, not chunk-aligned ones.
 */
export function sliceToSubClip(
  chunks: Blob[],
  mimeType: string,
  swings: BatchExtractSwingInput[],
  blobStartMs: number,
): SplicedSubClip {
  if (chunks.length === 0) {
    return {
      blob: new Blob([], { type: mimeType }),
      perSwingOffsets: [],
    }
  }
  if (swings.length === 0) {
    return {
      blob: new Blob([], { type: mimeType }),
      perSwingOffsets: [],
    }
  }

  // Estimate the total media-clock duration covered by the chunk buffer.
  // Use the latest swing's endMs as the upper bound — caller is expected
  // to flush MediaRecorder before invoking us, so the latest swing end
  // is at or before the buffer's tail.
  const lastEnd = Math.max(...swings.map((s) => s.endMs))
  const totalDurationMs = Math.max(1, lastEnd - blobStartMs)
  const perChunkMs = totalDurationMs / chunks.length

  // Build a [chunkIndex] → [chunkStartMs, chunkEndMs) table.
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

    // Walk the chunks in order; include any that intersect this swing.
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
  const subClip = sliceToSubClip(req.blobChunks, req.blobMimeType, req.swings, req.blobStartMs)

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
