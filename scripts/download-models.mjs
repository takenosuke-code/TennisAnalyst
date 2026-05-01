#!/usr/bin/env node
// Fetches the two ONNX model files the browser pose stack needs (Phase 5)
// and drops them at public/models/. Runs as part of `postinstall` after
// scripts/copy-ffmpeg-core.mjs, and as the standalone `npm run
// download-models` command.
//
// Models:
//   - yolo11n.onnx     : copied from railway-service/models/yolo11n.onnx
//                        (already in-repo for Railway cold-start reliability)
//   - rtmpose-m.onnx   : downloaded + unzipped from openmmlab.com, same URL
//                        rtmlib pins server-side (see
//                        railway-service/pose_rtmpose.py:79).
//
// Idempotent: if the destination file already exists with a sane size,
// skip with an "OK" log line. Network errors do NOT block the install —
// devs can keep working; visiting /live without the models will surface
// a runtime error from lib/modelLoader.ts that's much easier to debug
// than a failed install.
//
// No third-party deps. Zip extraction is done in-script with a minimal
// central-directory parser + node:zlib's raw inflate. The OpenMMLab
// archive is a stock zip with one .onnx file inside, so a 100-line
// parser is plenty.

import { Buffer } from 'node:buffer'
import {
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { inflateRawSync } from 'node:zlib'

const MODELS_DIR = resolve('public/models')
// Vercel persists `node_modules/` between builds, so anything we drop in
// node_modules/.cache survives. We use that as a hop for the rtmpose
// download — first build does the cross-Pacific fetch, subsequent
// builds just copy the file. Keeps `next build` off the openmmlab.com
// critical path on every deploy.
const CACHE_DIR = resolve('node_modules/.cache/tennis-models')

// Source URL for the RTMPose-m body7 256x192 ONNX. Pinned to match
// railway-service/pose_rtmpose.py so live + analyze stay in lockstep on
// the exact same weights. If you bump this URL on the Python side,
// bump it here too.
const RTMPOSE_ZIP_URL =
  'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/' +
  'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'

// Local source for yolo11n.onnx. Already committed to the repo for
// Railway cold-start reliability; we just copy it into public/models/.
const YOLO_LOCAL_SRC = resolve('railway-service/models/yolo11n.onnx')

// Expected sizes (bytes), with a generous tolerance to allow for
// re-encoding / minor revisions. Used to catch a partial / truncated
// download and to early-exit when the destination already looks healthy.
const EXPECTED = {
  'yolo11n.onnx': { bytes: 10_631_335, tolerance: 0.05 },
  'rtmpose-m.onnx': { bytes: 50_400_000, tolerance: 0.15 }, // ~50 MB; vendor occasionally re-mints
}

const NET_TIMEOUT_MS = 5 * 60 * 1000 // 5 min cap; the rtmpose zip is ~50 MB

/** @param {number} bytes */
function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

/** @param {string} name @param {number} actual */
function sizeIsHealthy(name, actual) {
  const exp = EXPECTED[name]
  if (!exp) return actual > 1024 // generic sanity check
  const lo = exp.bytes * (1 - exp.tolerance)
  const hi = exp.bytes * (1 + exp.tolerance)
  return actual >= lo && actual <= hi
}

/** @param {string} name */
function destPathFor(name) {
  return join(MODELS_DIR, name)
}

/** @param {string} name */
function destLooksGood(name) {
  const p = destPathFor(name)
  if (!existsSync(p)) return false
  try {
    const s = statSync(p)
    return sizeIsHealthy(name, s.size)
  } catch {
    return false
  }
}

function ensureModelsDir() {
  mkdirSync(MODELS_DIR, { recursive: true })
}

function copyYolo() {
  const dst = destPathFor('yolo11n.onnx')
  if (destLooksGood('yolo11n.onnx')) {
    const sz = statSync(dst).size
    console.log(`[models] yolo11n.onnx: ${fmtSize(sz)} OK (cached)`)
    return
  }
  if (!existsSync(YOLO_LOCAL_SRC)) {
    console.warn(
      `[models] yolo11n.onnx: source missing at ${YOLO_LOCAL_SRC}.\n` +
        '         Run from a fresh clone where railway-service/models/yolo11n.onnx is present.\n' +
        '         /live will fail to load until this file is in public/models/.',
    )
    return
  }
  copyFileSync(YOLO_LOCAL_SRC, dst)
  const sz = statSync(dst).size
  if (!sizeIsHealthy('yolo11n.onnx', sz)) {
    console.warn(
      `[models] yolo11n.onnx: copied ${fmtSize(sz)} but expected ~${fmtSize(EXPECTED['yolo11n.onnx'].bytes)}.\n` +
        '         File may be corrupt; check the source.',
    )
    return
  }
  console.log(`[models] yolo11n.onnx: ${fmtSize(sz)} OK (copied)`)
}

/**
 * Download the URL to `dest`, streaming so we can render a progress bar.
 * Throws on non-2xx. Cleans up partial files on error.
 *
 * @param {string} url
 * @param {string} dest
 * @param {(loaded: number, total: number) => void} onProgress
 */
async function downloadStreamed(url, dest, onProgress) {
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(new Error('timeout')), NET_TIMEOUT_MS)
  let resp
  try {
    resp = await fetch(url, { signal: ac.signal, redirect: 'follow' })
  } finally {
    // Note: we want the timer to keep ticking through the download — clear
    // it once the body is fully consumed below.
  }
  if (!resp.ok) {
    clearTimeout(timer)
    throw new Error(`HTTP ${resp.status} ${resp.statusText} for ${url}`)
  }
  const total = Number(resp.headers.get('content-length')) || 0
  const tmp = `${dest}.partial`
  let loaded = 0
  let lastPct = -1

  // Convert the WHATWG ReadableStream to a Node stream and pipe to disk
  // while reporting progress.
  const nodeStream = Readable.fromWeb(resp.body)
  const sink = createWriteStream(tmp)
  nodeStream.on('data', (chunk) => {
    loaded += chunk.length
    if (total > 0) {
      const pct = Math.floor((loaded / total) * 100)
      if (pct !== lastPct && pct % 5 === 0) {
        onProgress(loaded, total)
        lastPct = pct
      }
    }
  })
  try {
    await pipeline(nodeStream, sink)
  } finally {
    clearTimeout(timer)
  }
  // Final progress tick so callers see "100%".
  onProgress(loaded, total || loaded)
  renameSync(tmp, dest)
}

/**
 * Minimal ZIP reader. Parses the End-of-Central-Directory record at the
 * tail of the file, walks the central directory to find each entry, and
 * inflates the deflated data with zlib.inflateRawSync. Sufficient for
 * the OpenMMLab rtmpose archive (single .onnx file, store/deflate, no
 * encryption, no zip64 — the zip is ~50 MB).
 *
 * Returns an array of `{ name, data }`.
 *
 * @param {Buffer} buf
 */
function readZipEntries(buf) {
  // EOCD signature: 0x06054b50, found by scanning backward from EOF.
  const EOCD_SIG = 0x06054b50
  const CDFH_SIG = 0x02014b50
  const LFH_SIG = 0x04034b50

  let eocdOffset = -1
  // Min EOCD size is 22 bytes; comment can extend it up to 65557 bytes.
  const scanStart = Math.max(0, buf.length - 65557 - 22)
  for (let i = buf.length - 22; i >= scanStart; i--) {
    if (buf.readUInt32LE(i) === EOCD_SIG) {
      eocdOffset = i
      break
    }
  }
  if (eocdOffset < 0) throw new Error('zip: EOCD not found')

  const totalEntries = buf.readUInt16LE(eocdOffset + 10)
  const cdSize = buf.readUInt32LE(eocdOffset + 12)
  const cdOffset = buf.readUInt32LE(eocdOffset + 16)
  if (cdOffset === 0xffffffff || cdSize === 0xffffffff) {
    throw new Error('zip: zip64 not supported')
  }

  const entries = []
  let cur = cdOffset
  for (let i = 0; i < totalEntries; i++) {
    if (buf.readUInt32LE(cur) !== CDFH_SIG) {
      throw new Error(`zip: bad central directory header at ${cur}`)
    }
    const compMethod = buf.readUInt16LE(cur + 10)
    const compSize = buf.readUInt32LE(cur + 20)
    const fileNameLen = buf.readUInt16LE(cur + 28)
    const extraLen = buf.readUInt16LE(cur + 30)
    const commentLen = buf.readUInt16LE(cur + 32)
    const localHeaderOffset = buf.readUInt32LE(cur + 42)
    const name = buf.slice(cur + 46, cur + 46 + fileNameLen).toString('utf8')

    // Walk to the local file header to find the actual data offset
    if (buf.readUInt32LE(localHeaderOffset) !== LFH_SIG) {
      throw new Error(`zip: bad local file header at ${localHeaderOffset}`)
    }
    const lfhFileNameLen = buf.readUInt16LE(localHeaderOffset + 26)
    const lfhExtraLen = buf.readUInt16LE(localHeaderOffset + 28)
    const dataStart =
      localHeaderOffset + 30 + lfhFileNameLen + lfhExtraLen
    const compressed = buf.slice(dataStart, dataStart + compSize)

    let data
    if (compMethod === 0) {
      // stored
      data = compressed
    } else if (compMethod === 8) {
      // deflate (raw — no zlib header)
      data = inflateRawSync(compressed)
    } else {
      throw new Error(`zip: unsupported compression method ${compMethod}`)
    }
    entries.push({ name, data })
    cur += 46 + fileNameLen + extraLen + commentLen
  }
  return entries
}

async function downloadRtmpose() {
  const dst = destPathFor('rtmpose-m.onnx')
  if (destLooksGood('rtmpose-m.onnx')) {
    const sz = statSync(dst).size
    console.log(`[models] rtmpose-m.onnx: ${fmtSize(sz)} OK (cached)`)
    return
  }

  // Look for a copy in the persistent build cache (Vercel keeps
  // node_modules/ between builds). If found and healthy, skip the
  // openmmlab.com download entirely.
  const cached = join(CACHE_DIR, 'rtmpose-m.onnx')
  if (existsSync(cached)) {
    try {
      const sz = statSync(cached).size
      if (sizeIsHealthy('rtmpose-m.onnx', sz)) {
        copyFileSync(cached, dst)
        console.log(
          `[models] rtmpose-m.onnx: ${fmtSize(sz)} OK (build-cache hit)`,
        )
        return
      }
    } catch {
      /* fall through to network download */
    }
  }

  const tmpZip = join(MODELS_DIR, 'rtmpose-m.zip.partial')

  console.log(`[models] rtmpose-m.onnx: downloading…`)
  try {
    await downloadStreamed(RTMPOSE_ZIP_URL, tmpZip, (loaded, total) => {
      const pct = total > 0 ? Math.floor((loaded / total) * 100) : 0
      const sizeStr = total > 0 ? `${fmtSize(loaded)} / ${fmtSize(total)}` : fmtSize(loaded)
      process.stdout.write(`\r[models] rtmpose-m.onnx: downloading… ${pct}% (${sizeStr})    `)
    })
    process.stdout.write('\n')
  } catch (err) {
    process.stdout.write('\n')
    if (existsSync(tmpZip)) {
      try { unlinkSync(tmpZip) } catch { /* ignore */ }
    }
    const msg = err instanceof Error ? err.message : String(err)
    console.warn(
      `[models] rtmpose-m.onnx: download failed (${msg}).\n` +
        '         /live will not work until this file lands in public/models/.\n' +
        '         Re-run `npm run download-models` once the network is available.',
    )
    return
  }

  // Read the zip and find the .onnx entry
  let entries
  try {
    const zipBuf = readFileSync(tmpZip)
    entries = readZipEntries(zipBuf)
  } catch (err) {
    if (existsSync(tmpZip)) {
      try { unlinkSync(tmpZip) } catch { /* ignore */ }
    }
    const msg = err instanceof Error ? err.message : String(err)
    console.warn(`[models] rtmpose-m.onnx: zip parse failed (${msg}). Skipping.`)
    return
  }

  const onnxEntry = entries.find((e) => e.name.toLowerCase().endsWith('.onnx'))
  if (!onnxEntry) {
    try { unlinkSync(tmpZip) } catch { /* ignore */ }
    console.warn(
      `[models] rtmpose-m.onnx: no .onnx entry in archive. Found: ${entries.map((e) => e.name).join(', ')}`,
    )
    return
  }

  writeFileSync(dst, onnxEntry.data)
  try { unlinkSync(tmpZip) } catch { /* ignore */ }

  const sz = statSync(dst).size
  if (!sizeIsHealthy('rtmpose-m.onnx', sz)) {
    console.warn(
      `[models] rtmpose-m.onnx: extracted ${fmtSize(sz)} but expected ~${fmtSize(EXPECTED['rtmpose-m.onnx'].bytes)}.\n` +
        '         File may be corrupt; delete and re-run.',
    )
    return
  }

  // Drop a copy in the persistent build cache so the next Vercel build
  // skips the download. Best-effort — failures here aren't fatal.
  try {
    mkdirSync(CACHE_DIR, { recursive: true })
    copyFileSync(dst, cached)
  } catch {
    /* ignore — we already have the file in public/models for this build */
  }

  console.log(`[models] rtmpose-m.onnx: ${fmtSize(sz)} OK (downloaded + extracted)`)
}

async function main() {
  ensureModelsDir()
  copyYolo()
  await downloadRtmpose()
}

main().catch((err) => {
  // Never block install. Surface as a warning; runtime error in
  // /live will be the next cue if the models really aren't there.
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err)
  console.warn(`[models] unexpected error: ${msg}\n[models] continuing without models.`)
  process.exit(0)
})
