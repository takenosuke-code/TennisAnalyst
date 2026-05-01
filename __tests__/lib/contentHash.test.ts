import { describe, it, expect, vi } from 'vitest'
import { createHash, randomBytes } from 'node:crypto'

import { sha256Hex, hashFileInWorker } from '@/lib/contentHash'

// ---------------------------------------------------------------------------
// Why these tests look the way they do
// ---------------------------------------------------------------------------
//
// jsdom (the vitest environment for this repo) ships crypto.subtle, so
// `sha256Hex` runs natively. It does NOT ship a real Web Worker — the
// global `Worker` is defined as a stub that throws or no-ops. That's
// fine: our `hashFileInWorker` falls back to running `sha256Hex` on
// the main thread when Worker is unavailable, and that fallback is
// what we exercise here. The Worker-execution path is covered
// indirectly via parity assertion: Worker and main-thread call into
// the same `sha256Hex`, so verifying the digest helper against
// Node's `crypto.createHash('sha256')` covers both.

function nodeSha256Hex(buf: ArrayBuffer | Uint8Array): string {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf)
  return createHash('sha256').update(u8).digest('hex')
}

describe('sha256Hex (pure helper)', () => {
  it('matches Node crypto for [0,1,2,3]', async () => {
    const input = new Uint8Array([0, 1, 2, 3])
    const browserHex = await sha256Hex(input)
    expect(browserHex).toBe(nodeSha256Hex(input))
    // Sanity: 64 hex chars.
    expect(browserHex).toMatch(/^[0-9a-f]{64}$/)
  })

  it('matches Node crypto for empty input', async () => {
    const empty = new Uint8Array(0)
    const browserHex = await sha256Hex(empty)
    expect(browserHex).toBe(nodeSha256Hex(empty))
    // Well-known SHA-256 of empty string.
    expect(browserHex).toBe(
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    )
  })

  it('matches Node crypto for the ASCII string "tennis-analyzer"', async () => {
    const input = new TextEncoder().encode('tennis-analyzer')
    const browserHex = await sha256Hex(input)
    expect(browserHex).toBe(nodeSha256Hex(input))
  })

  it('matches Node crypto for a 1 MB random buffer', async () => {
    // Use Node's randomBytes so the comparison is against a known
    // reference rather than re-hashing our own output.
    const bytes = randomBytes(1024 * 1024)
    const u8 = new Uint8Array(bytes)
    const browserHex = await sha256Hex(u8)
    expect(browserHex).toBe(nodeSha256Hex(u8))
  })

  it('is deterministic — calling twice on the same buffer yields the same hash', async () => {
    const input = new Uint8Array([7, 7, 7, 42, 9, 9, 9])
    const a = await sha256Hex(input)
    const b = await sha256Hex(input)
    expect(a).toBe(b)
  })

  it('accepts an ArrayBuffer directly (not just Uint8Array)', async () => {
    const u8 = new Uint8Array([1, 2, 3, 4, 5])
    const fromU8 = await sha256Hex(u8)
    const fromBuf = await sha256Hex(u8.buffer)
    expect(fromU8).toBe(fromBuf)
    expect(fromU8).toBe(nodeSha256Hex(u8))
  })

  it('throws a clear error when crypto.subtle.digest is unavailable', async () => {
    // Stash + remove subtle so we exercise the guard path. Restoring is
    // critical — leaving subtle blanked here would break every later
    // test in the file.
    const original = globalThis.crypto
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: {},
    })
    try {
      await expect(sha256Hex(new Uint8Array([1]))).rejects.toThrow(
        /crypto\.subtle\.digest/,
      )
    } finally {
      Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: original,
      })
    }
  })
})

describe('hashFileInWorker', () => {
  // jsdom doesn't ship a real Worker; `typeof Worker === 'undefined'`,
  // so hashFileInWorker takes the main-thread fallback. That's a
  // legitimate code path (older browsers, CSP-locked sites) and
  // verifying it here means production clients that can't construct
  // a Worker still get a correct hash.
  it('returns the same hash as sha256Hex for a known Blob (jsdom fallback)', async () => {
    const bytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])
    const blob = new Blob([bytes])
    const fromWorker = await hashFileInWorker(blob)
    const direct = await sha256Hex(bytes)
    expect(fromWorker).toBe(direct)
    expect(fromWorker).toBe(nodeSha256Hex(bytes))
  })

  it('is deterministic on identical Blob content', async () => {
    const bytes = new Uint8Array([0xde, 0xad, 0xbe, 0xef])
    const a = await hashFileInWorker(new Blob([bytes]))
    const b = await hashFileInWorker(new Blob([bytes]))
    expect(a).toBe(b)
  })

  it('two Blobs with different content hash to different values', async () => {
    const a = await hashFileInWorker(new Blob([new Uint8Array([0])]))
    const b = await hashFileInWorker(new Blob([new Uint8Array([1])]))
    expect(a).not.toBe(b)
  })

  it('handles a 1MB random Blob and matches Node parity', async () => {
    const buf = randomBytes(1024 * 1024)
    const u8 = new Uint8Array(buf)
    const blob = new Blob([u8])
    const fromWorker = await hashFileInWorker(blob)
    expect(fromWorker).toBe(nodeSha256Hex(u8))
  })

  it('does not call console.warn on the happy fallback path', async () => {
    // The fallback path (typeof Worker === 'undefined') is silent; the
    // warn would only fire on Worker construction failure. This guards
    // against an accidental log on every upload.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      await hashFileInWorker(new Blob([new Uint8Array([1, 2, 3])]))
      expect(warn).not.toHaveBeenCalled()
    } finally {
      warn.mockRestore()
    }
  })
})
