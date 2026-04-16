import { describe, it, expect, beforeEach } from 'vitest'

// mediapipe.ts uses module-level mutable state (_highWaterMark), and it has
// a 'use client' directive. We need to re-import with fresh state per test
// group. Vitest's module reset helps here.

// We cannot easily reset module state between tests without vi.resetModules().
// Instead, we accept that timestamps are monotonically increasing across all
// tests within this file (which is actually what the code guarantees).

describe('mediapipe timestamp management', () => {
  // We need dynamic import to get fresh module state
  let getMonotonicTimestamp: (relativeMs: number) => number
  let recordTimestamp: (ts: number) => void

  beforeEach(async () => {
    // Use dynamic imports to get the functions.
    // Note: Module state persists between tests because vitest caches modules.
    // We test the monotonic property which holds regardless of module state.
    const mod = await import('@/lib/mediapipe')
    getMonotonicTimestamp = mod.getMonotonicTimestamp
    recordTimestamp = mod.recordTimestamp
  })

  it('getMonotonicTimestamp returns increasing values for sequential calls', () => {
    const ts1 = getMonotonicTimestamp(0)
    const ts2 = getMonotonicTimestamp(0)
    const ts3 = getMonotonicTimestamp(0)

    // Each call updates the high-water mark, so subsequent calls produce higher values
    expect(ts2).toBeGreaterThan(ts1)
    expect(ts3).toBeGreaterThan(ts2)
  })

  it('getMonotonicTimestamp returns higher values for higher relative offsets', () => {
    const ts1 = getMonotonicTimestamp(0)
    const ts2 = getMonotonicTimestamp(100)
    const ts3 = getMonotonicTimestamp(200)

    expect(ts2).toBeGreaterThan(ts1)
    expect(ts3).toBeGreaterThan(ts2)
  })

  it('recordTimestamp updates the high-water mark', () => {
    // Record a large timestamp
    recordTimestamp(50000)

    // Now getMonotonicTimestamp(0) should return > 50000
    const ts = getMonotonicTimestamp(0)
    expect(ts).toBeGreaterThan(50000)
  })

  it('after recording ts=5000, getMonotonicTimestamp(0) returns > 5000', () => {
    recordTimestamp(100000) // use a value higher than any prior test state
    const ts = getMonotonicTimestamp(0)
    expect(ts).toBeGreaterThan(100000)
  })

  it('recordTimestamp ignores lower values (does not decrease high-water mark)', () => {
    recordTimestamp(200000)
    const tsAfterHigh = getMonotonicTimestamp(0)

    // Now record a lower value
    recordTimestamp(100)
    const tsAfterLow = getMonotonicTimestamp(0)

    // tsAfterLow should still be greater than tsAfterHigh because
    // the high-water mark was not decreased by recordTimestamp(100)
    expect(tsAfterLow).toBeGreaterThan(tsAfterHigh)
  })

  it('getMonotonicTimestamp with relativeMs adds to high-water mark', () => {
    recordTimestamp(300000)
    // ts = _highWaterMark + 1 + relativeMs
    const ts = getMonotonicTimestamp(500)
    expect(ts).toBe(300000 + 1 + 500)
  })
})
