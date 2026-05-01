// Web Worker: SHA-256 a File/Blob without blocking the main thread.
//
// Wired in by lib/contentHash.ts via:
//   new Worker(new URL('./contentHash.worker.ts', import.meta.url),
//              { type: 'module' })
//
// Protocol (kept tiny, no shared deps):
//   in:  { blob: Blob }
//   out: { ok: true,  hex: string }   on success
//        { ok: false, error: string } on failure (e.g. crypto.subtle missing)
//
// The worker imports the same `sha256Hex` helper main-thread code uses,
// so test parity (browser sha256 == Node crypto sha256) is enforced
// against a single implementation.

import { sha256Hex } from './contentHash'

type InMsg = { blob: Blob }
type OutMsg = { ok: true; hex: string } | { ok: false; error: string }

// `self` in a module worker is DedicatedWorkerGlobalScope. The repo's
// tsconfig only ships the DOM lib (not WebWorker), so we type the
// scope as a Window-compatible target with the postMessage signature
// we need. This narrows just enough for type-safety without having
// to add WebWorker to lib globally.
type WorkerScope = {
  addEventListener(
    type: 'message',
    listener: (e: MessageEvent<InMsg>) => void,
  ): void
  postMessage(msg: OutMsg): void
}
const ctx = self as unknown as WorkerScope

ctx.addEventListener('message', async (e: MessageEvent<InMsg>) => {
  const { blob } = e.data ?? {}
  if (!blob || typeof blob.arrayBuffer !== 'function') {
    const out: OutMsg = { ok: false, error: 'contentHash worker: payload is not a Blob' }
    ctx.postMessage(out)
    return
  }
  try {
    // arrayBuffer() reads the entire Blob into memory in the worker
    // context. For very large files (>500 MB) this could OOM the
    // worker; current upload cap is 200 MB (UploadZone copy) so we're
    // well under. If we ever raise that cap, switch to the streaming
    // approach (read chunks via Blob.slice() and feed an incremental
    // SHA-256 — the spec doesn't expose one, so we'd need a wasm
    // crate or a JS implementation).
    const buf = await blob.arrayBuffer()
    const hex = await sha256Hex(buf)
    const out: OutMsg = { ok: true, hex }
    ctx.postMessage(out)
  } catch (err) {
    const out: OutMsg = {
      ok: false,
      error: err instanceof Error ? err.message : 'contentHash worker: unknown error',
    }
    ctx.postMessage(out)
  }
})

// Module workers need an explicit empty export so TS treats this file
// as a module. The `import` above already does that — the explicit
// statement here is purely a defensive marker for future authors.
export {}
