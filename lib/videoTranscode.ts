'use client'

import {
  Conversion,
  Input,
  Output,
  BlobSource,
  BufferTarget,
  Mp4OutputFormat,
  ALL_FORMATS,
  QUALITY_MEDIUM,
} from 'mediabunny'

/*
 * Hardware-accelerated client-side video transcode for the analyze
 * upload flow. The reason this exists: a 4K60 phone clip is ~350 MB
 * and takes 10+ minutes to upload over cellular; transcoded to 720p30
 * H.264 it's ~50 MB and uploads in under a minute. The compute is
 * routed to the phone's video-encoder silicon (VideoToolbox on iOS,
 * Mediacodec on Android) via WebCodecs, so the CPU/battery cost is
 * tiny compared to a software transcode (ffmpeg.wasm) that would
 * burn a regular core.
 *
 * Mediabunny is the library doing demux → decode → resize → encode →
 * mux. Replaces the now-deprecated mp4-muxer and exposes a
 * `Conversion` helper that's the canonical short-form for the whole
 * pipeline. https://github.com/Vanilagy/mediabunny
 *
 * This module is the simple "transcode-to-File then upload" path.
 * Worker-based streaming would be marginally smoother on lower-end
 * phones; the encoder work is inherently async so the main thread
 * stays mostly unblocked even without one. If we hit jank on
 * iPhone < 13 we'll move to a Worker.
 */

// Target dimensions / bitrate. 1280×720 × 30fps × ~2.5 Mbps lands a
// 60s clip around 18-20 MB after H.264 — enough headroom to upload
// over cellular in under a minute.
const TARGET_WIDTH = 1280
const TARGET_FRAMERATE = 30
// Threshold: anything wider than this triggers transcode. Smaller
// inputs already upload quickly, no point spending battery
// re-encoding them.
const TRANSCODE_TRIGGER_WIDTH = 1281
// Or any file bigger than this regardless of dimensions — covers the
// "phone shot at 1080p but bitrate is high" case.
const TRANSCODE_TRIGGER_BYTES = 80 * 1024 * 1024

export interface TranscodeOptions {
  // 0..1 fraction of completed work. Caller maps onto its progress UI.
  onProgress?: (fraction: number) => void
}

export interface TranscodeResult {
  // The transcoded File, ready to upload via tus. Same File API as the
  // original so callers can swap one in for the other without changes.
  file: File
  // Original byte size for telemetry / debugging. Surfaces as a
  // "compressed 350 MB → 18 MB" status line.
  originalBytes: number
  outputBytes: number
}

/**
 * Decide whether a file should be transcoded. Returns false for files
 * already small enough to upload directly — saves the user
 * 10-15 seconds of unnecessary encoding when their phone is already
 * recording at 1080p30.
 */
export function shouldTranscode(file: File, sniffedWidth?: number): boolean {
  if (file.size > TRANSCODE_TRIGGER_BYTES) return true
  if (sniffedWidth && sniffedWidth >= TRANSCODE_TRIGGER_WIDTH) return true
  return false
}

/**
 * Read the source video's intrinsic dimensions without decoding the
 * whole file — used to gate the transcode/skip decision before
 * spending hardware on encoding. Falls back to undefined on any
 * decode failure (caller treats that as "don't know, skip transcode
 * unless the file is huge").
 */
export async function sniffVideoSize(file: File): Promise<{ width: number; height: number } | undefined> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.muted = true
    let settled = false
    const done = (result: { width: number; height: number } | undefined) => {
      if (settled) return
      settled = true
      URL.revokeObjectURL(url)
      v.removeAttribute('src')
      try { v.load() } catch { /* noop */ }
      resolve(result)
    }
    v.onloadedmetadata = () => {
      if (v.videoWidth > 0 && v.videoHeight > 0) {
        done({ width: v.videoWidth, height: v.videoHeight })
      } else {
        done(undefined)
      }
    }
    v.onerror = () => done(undefined)
    // 5s ceiling — HEVC iPhone clips can hang here on Safari < 17.
    setTimeout(() => done(undefined), 5000)
    v.src = url
  })
}

/**
 * Feature-detect whether this browser can hardware-encode H.264 via
 * WebCodecs. We use Baseline profile 3.0 (`avc1.42E01E`) because
 * VideoToolbox on iPhone is hardware-accelerated for it universally.
 * Returns false on Safari < 17, older Chromium, or anywhere the
 * `VideoEncoder` global isn't defined — caller falls through to
 * uploading the original file unchanged.
 */
export async function canTranscode(): Promise<boolean> {
  if (typeof window === 'undefined') return false
  if (typeof (globalThis as { VideoEncoder?: unknown }).VideoEncoder === 'undefined') return false
  try {
    // Chrome resolves with `{ supported: false }`, Safari rejects on
    // unsupported configs — handle both.
    const probe = await (globalThis as { VideoEncoder: { isConfigSupported: (c: unknown) => Promise<{ supported?: boolean }> } })
      .VideoEncoder.isConfigSupported({
        codec: 'avc1.42E01E',
        width: TARGET_WIDTH,
        height: 720,
        bitrate: 2_500_000,
        framerate: TARGET_FRAMERATE,
      })
    return probe.supported === true
  } catch {
    return false
  }
}

/**
 * Transcode `file` to 1280×720 H.264 at 30fps. Returns a fresh File
 * whose .name preserves the original (with .mp4 extension swapped in)
 * so downstream filename-derived paths don't break.
 *
 * Throws on encoder/decoder failure; caller is expected to catch and
 * fall through to uploading the original.
 */
export async function transcodeToH264_720p(
  file: File,
  options: TranscodeOptions = {},
): Promise<TranscodeResult> {
  const input = new Input({
    source: new BlobSource(file),
    formats: ALL_FORMATS,
  })
  const target = new BufferTarget()
  const output = new Output({
    format: new Mp4OutputFormat({
      // fastStart puts the moov atom at the front, letting consumers
      // start streaming as soon as the first chunks land. Costs us a
      // tiny bit of memory during finalize; worth it for playback UX.
      fastStart: 'in-memory',
    }),
    target,
  })

  const conversion = await Conversion.init({
    input,
    output,
    video: {
      width: TARGET_WIDTH,
      // height auto-derived from aspect ratio when only width is set
      fit: 'contain',
      frameRate: TARGET_FRAMERATE,
      codec: 'avc',
      // QUALITY_MEDIUM resolves to ~2.5 Mbps at 720p — the sweet
      // spot for visible-quality vs upload size.
      bitrate: QUALITY_MEDIUM,
    },
  })

  if (!conversion.isValid) {
    throw new Error(
      `transcode invalid: ${conversion.discardedTracks.map((t) => t.reason).join(', ')}`,
    )
  }

  conversion.onProgress = (p) => options.onProgress?.(p)
  await conversion.execute()

  const buffer = target.buffer
  if (!buffer) throw new Error('transcode finished but produced no buffer')

  // Preserve the source filename but force .mp4 so iOS/Android pickers
  // and the Supabase content_type allowlist accept the result.
  const baseName = file.name.replace(/\.[^.]+$/, '')
  const transcoded = new File([buffer], `${baseName}.mp4`, { type: 'video/mp4' })

  return {
    file: transcoded,
    originalBytes: file.size,
    outputBytes: transcoded.size,
  }
}
