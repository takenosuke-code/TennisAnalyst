// SHA-256 content hashing for the pose-cache flow.
//
// Two layers:
//
// 1. `sha256Hex(buf)` — pure async helper that runs subtle.crypto and
//    returns the lowercase hex digest. Works in any context with
//    crypto.subtle (browsers, Web Workers, Node 19+, jsdom). Tests hit
//    this directly so they don't need a real Worker.
//
// 2. `hashFileInWorker(file)` — wraps the Web Worker (
//    contentHash.worker.ts) so the main thread doesn't block while
//    digesting a 200 MB clip. Falls back to in-process `sha256Hex`
//    when Worker isn't available (SSR, very old browsers, jsdom).
//
// We only need hex out — not bytes, not base64 — so the helper returns
// a string and callers don't need to think about encoding.

/**
 * SHA-256 the input and return the lowercase hex digest. Works in any
 * environment with `crypto.subtle` (modern browsers, Web Workers,
 * Node 19+).
 *
 * Throws if `crypto.subtle.digest` is unavailable. Callers should
 * surface a clear "your browser doesn't support secure hashing" error
 * rather than swallowing — silent fallback to no-cache would mean
 * every upload re-runs Modal extraction, which is exactly what this
 * module exists to prevent.
 */
export async function sha256Hex(
  buf: ArrayBuffer | Uint8Array,
): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle?.digest) {
    throw new Error('sha256Hex: crypto.subtle.digest is not available')
  }
  // subtle.digest takes a BufferSource (ArrayBuffer | TypedArray |
  // DataView). Node Buffer (subclass of Uint8Array but bound to a
  // SharedArrayBuffer-ish backing in Node) and the FileReader-result
  // type can both fail the strict instanceof check that
  // SubtleCrypto.digest performs in jsdom. Normalize to a fresh
  // Uint8Array view backed by a real ArrayBuffer to dodge that.
  // Always copy into a fresh ArrayBuffer-backed Uint8Array so the
  // result is exactly what SubtleCrypto.digest expects.
  let data: Uint8Array<ArrayBuffer>
  if (buf instanceof ArrayBuffer) {
    data = new Uint8Array(buf.slice(0))
  } else {
    const fresh = new ArrayBuffer(buf.byteLength)
    const view = new Uint8Array(fresh)
    view.set(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength))
    data = view
  }
  const hashBuf = await crypto.subtle.digest('SHA-256', data)
  // Manual hex encoding — TextDecoder won't work (binary, not text) and
  // Buffer doesn't exist in browsers. 32 bytes -> 64 hex chars.
  const bytes = new Uint8Array(hashBuf)
  let hex = ''
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0')
  }
  return hex
}

/**
 * Read a Blob's full contents as an ArrayBuffer. Wraps Blob.arrayBuffer()
 * with a FileReader fallback for environments where it's missing
 * (jsdom Blob, very old browsers). The FileReader path is functionally
 * identical — both return the entire buffer once the read completes.
 */
function readBlobAsArrayBuffer(blob: Blob): Promise<ArrayBuffer> {
  if (typeof (blob as Blob & { arrayBuffer?: () => Promise<ArrayBuffer> }).arrayBuffer === 'function') {
    return blob.arrayBuffer()
  }
  // FileReader fallback — present in jsdom and every browser.
  return new Promise((resolve, reject) => {
    if (typeof FileReader === 'undefined') {
      reject(new Error('readBlobAsArrayBuffer: no Blob.arrayBuffer or FileReader'))
      return
    }
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error ?? new Error('FileReader error'))
    reader.readAsArrayBuffer(blob)
  })
}

/**
 * SHA-256 a File or Blob in a Web Worker so the main thread stays
 * responsive during a multi-hundred-MB digest. The Worker reads the
 * Blob via `arrayBuffer()` inside the worker context, hashes it, and
 * posts the hex string back.
 *
 * Falls back to running `sha256Hex` on the main thread when:
 *   - `Worker` is undefined (SSR / jsdom)
 *   - Worker construction throws (CSP / sandbox / older browser)
 *
 * Both paths return the same string, so callers don't branch.
 *
 * Usage:
 *   const sha = await hashFileInWorker(file)
 *   // -> "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
 */
export function hashFileInWorker(file: Blob): Promise<string> {
  if (typeof Worker === 'undefined') {
    return readBlobAsArrayBuffer(file).then(sha256Hex)
  }

  return new Promise<string>((resolve, reject) => {
    let worker: Worker
    try {
      // Standard webpack/Turbopack pattern for ES-module workers in
      // Next.js — the bundler picks up the URL and emits a separate
      // worker chunk at build time. Kept inside the Promise so a
      // construction failure (CSP, etc) falls through to the main-
      // thread fallback below.
      worker = new Worker(
        new URL('./contentHash.worker.ts', import.meta.url),
        { type: 'module' },
      )
    } catch (err) {
      // Construction failed — fall back to main thread.
      console.warn(
        '[contentHash] Worker construction failed, hashing on main thread:',
        err,
      )
      readBlobAsArrayBuffer(file).then(sha256Hex).then(resolve, reject)
      return
    }

    worker.onmessage = (e: MessageEvent<{ ok: true; hex: string } | { ok: false; error: string }>) => {
      worker.terminate()
      const data = e.data
      if (data.ok) {
        resolve(data.hex)
      } else {
        reject(new Error(data.error))
      }
    }

    worker.onerror = (e) => {
      worker.terminate()
      reject(new Error(`contentHash worker error: ${e.message || 'unknown'}`))
    }

    // Transferring the Blob is cheap — Blobs are reference-counted
    // across the postMessage boundary, no copy required.
    worker.postMessage({ blob: file })
  })
}
