import { describe, it, expect, vi, beforeEach } from 'vitest'

import type { KeypointsJson } from '@/lib/supabase'

// ---------------------------------------------------------------------------
// Mock @/lib/supabase the same way analyze-stream.test.ts does.
//
// The pose-cache module calls supabaseAdmin.from('pose_cache')
//   .select(...).eq(...).maybeSingle()        // get
//   .upsert(payload, { onConflict, ignoreDuplicates })  // set
//
// Each chain stage is its own vi.fn() so individual tests can stub the
// terminal behavior (data/error pair) without rewriting the whole tree.
// ---------------------------------------------------------------------------

// vi.hoisted runs before any vi.mock factory, so the chain mocks can
// be referenced inside the factory. Without it the factory closes over
// uninitialized vars and the import explodes.
const { mockMaybeSingle, mockEq, mockSelect, mockUpsert, mockFrom } = vi.hoisted(() => {
  const mockMaybeSingle = vi.fn()
  const mockEq = vi.fn(() => ({ maybeSingle: mockMaybeSingle }))
  const mockSelect = vi.fn(() => ({ eq: mockEq }))
  const mockUpsert = vi.fn()
  const mockFrom = vi.fn(() => ({
    select: mockSelect,
    upsert: mockUpsert,
  }))
  return { mockMaybeSingle, mockEq, mockSelect, mockUpsert, mockFrom }
})

vi.mock('@/lib/supabase', () => ({
  supabase: { from: mockFrom },
  supabaseAdmin: { from: mockFrom },
}))

// Imported AFTER the mock so the module-under-test picks up the
// mocked supabaseAdmin.
import { getCachedPose, setCachedPose } from '@/lib/poseCache'

const SAMPLE_POSE: KeypointsJson = {
  fps_sampled: 30,
  frame_count: 1,
  schema_version: 3,
  frames: [
    {
      frame_index: 0,
      timestamp_ms: 0,
      landmarks: [],
      joint_angles: { right_elbow: 90 },
    },
  ],
}

const SHA = 'a'.repeat(64) // 64-char lowercase hex placeholder

describe('poseCache', () => {
  beforeEach(() => {
    mockMaybeSingle.mockReset()
    mockEq.mockClear()
    mockSelect.mockClear()
    mockUpsert.mockReset()
    mockFrom.mockClear()
    // Re-prime chain proxies because vi.fn() reset clears the
    // implementation, not the .mock results.
    mockEq.mockImplementation(() => ({ maybeSingle: mockMaybeSingle }))
    mockSelect.mockImplementation(() => ({ eq: mockEq }))
    mockFrom.mockImplementation(() => ({
      select: mockSelect,
      upsert: mockUpsert,
    }))
  })

  // -------------------------------------------------------------------------
  // getCachedPose
  // -------------------------------------------------------------------------
  describe('getCachedPose', () => {
    it('returns the stored poseJson on cache hit', async () => {
      mockMaybeSingle.mockResolvedValueOnce({
        data: { pose_json: SAMPLE_POSE, model_version: 'rtmpose-modal' },
        error: null,
      })
      const out = await getCachedPose(SHA)
      expect(out).toEqual(SAMPLE_POSE)
      // Hit the right table + column.
      expect(mockFrom).toHaveBeenCalledWith('pose_cache')
      expect(mockSelect).toHaveBeenCalledWith('pose_json, model_version')
      expect(mockEq).toHaveBeenCalledWith('sha256', SHA)
    })

    it('returns null on cache miss', async () => {
      mockMaybeSingle.mockResolvedValueOnce({ data: null, error: null })
      const out = await getCachedPose(SHA)
      expect(out).toBeNull()
    })

    it('returns null when sha256 is empty', async () => {
      const out = await getCachedPose('')
      expect(out).toBeNull()
      // Should not even hit the DB.
      expect(mockFrom).not.toHaveBeenCalled()
    })

    it('treats a model_version mismatch as a miss', async () => {
      mockMaybeSingle.mockResolvedValueOnce({
        data: { pose_json: SAMPLE_POSE, model_version: 'rtmpose-m-old' },
        error: null,
      })
      const out = await getCachedPose(SHA, 'rtmpose-l-new')
      expect(out).toBeNull()
    })

    it('returns the row when the model_version matches', async () => {
      mockMaybeSingle.mockResolvedValueOnce({
        data: { pose_json: SAMPLE_POSE, model_version: 'rtmpose-l-new' },
        error: null,
      })
      const out = await getCachedPose(SHA, 'rtmpose-l-new')
      expect(out).toEqual(SAMPLE_POSE)
    })

    it('returns the row when no expected model_version is provided (any version OK)', async () => {
      mockMaybeSingle.mockResolvedValueOnce({
        data: { pose_json: SAMPLE_POSE, model_version: 'whatever' },
        error: null,
      })
      const out = await getCachedPose(SHA)
      expect(out).toEqual(SAMPLE_POSE)
    })

    it('returns null + logs on DB error rather than throwing', async () => {
      // Don't want a transient DB error to take down the upload path —
      // the worst-case is we re-extract.
      mockMaybeSingle.mockResolvedValueOnce({
        data: null,
        error: { message: 'transient outage' },
      })
      const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        const out = await getCachedPose(SHA)
        expect(out).toBeNull()
        expect(errSpy).toHaveBeenCalled()
      } finally {
        errSpy.mockRestore()
      }
    })
  })

  // -------------------------------------------------------------------------
  // setCachedPose
  // -------------------------------------------------------------------------
  describe('setCachedPose', () => {
    it('writes via real upsert (onConflict=sha256, NO ignoreDuplicates)', async () => {
      // We deliberately use a real upsert (not ON CONFLICT DO NOTHING)
      // so a row written under an old model_version is overwritten
      // when re-extracted under a new one. The absence of
      // `ignoreDuplicates: true` is the load-bearing detail.
      mockUpsert.mockResolvedValueOnce({ error: null })
      await setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-modal')
      expect(mockFrom).toHaveBeenCalledWith('pose_cache')
      expect(mockUpsert).toHaveBeenCalledTimes(1)
      const [payload, options] = mockUpsert.mock.calls[0]
      expect(payload).toEqual({
        sha256: SHA,
        pose_json: SAMPLE_POSE,
        model_version: 'rtmpose-modal',
      })
      expect(options).toEqual({ onConflict: 'sha256' })
      // Explicit guard: a future "fix" that re-introduces
      // ignoreDuplicates would silently re-break the model-upgrade
      // path, so we pin its absence.
      expect(options).not.toHaveProperty('ignoreDuplicates')
    })

    it('throws on missing sha256', async () => {
      await expect(setCachedPose('', SAMPLE_POSE, 'v')).rejects.toThrow(
        /sha256 is required/,
      )
      expect(mockUpsert).not.toHaveBeenCalled()
    })

    it('throws on missing modelVersion', async () => {
      await expect(setCachedPose(SHA, SAMPLE_POSE, '')).rejects.toThrow(
        /modelVersion is required/,
      )
      expect(mockUpsert).not.toHaveBeenCalled()
    })

    it('throws on real DB error so the caller can choose to swallow', async () => {
      mockUpsert.mockResolvedValueOnce({
        error: { message: 'permission denied' },
      })
      await expect(setCachedPose(SHA, SAMPLE_POSE, 'v')).rejects.toThrow(
        /setCachedPose failed: permission denied/,
      )
    })

    it('concurrent setCachedPose calls for same sha256: both resolve cleanly', async () => {
      // Postgres ON CONFLICT (sha256) DO UPDATE is race-safe: two
      // concurrent writers for the same (sha256, model_version) pair
      // hold byte-equivalent payloads, so whichever lands second just
      // overwrites with identical data. We assert both calls resolve
      // cleanly — neither is allowed to throw.
      mockUpsert.mockResolvedValue({ error: null })
      const a = setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-modal')
      const b = setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-modal')
      await expect(Promise.all([a, b])).resolves.toBeDefined()
      expect(mockUpsert).toHaveBeenCalledTimes(2)
      // Both calls used the real-upsert options (no ignoreDuplicates).
      for (const [, opts] of mockUpsert.mock.calls) {
        expect(opts).toEqual({ onConflict: 'sha256' })
      }
    })

    it('model_version upgrade (v1 then v2) hands the v2 payload to upsert — caller relies on real-upsert semantics so the new row OVERWRITES the old', async () => {
      // Regression test for the cache-bypass-after-model-upgrade bug.
      // Earlier code used ON CONFLICT DO NOTHING here, which made the
      // second call a silent no-op — the v1 row stuck around forever
      // and the read-side gate (which compares model_version) treated
      // it as a permanent miss for any client expecting v2. After the
      // fix, setCachedPose passes onConflict='sha256' WITHOUT
      // ignoreDuplicates, which postgrest-js maps to a real
      // INSERT ... ON CONFLICT DO UPDATE. We can't observe the DB
      // mutation under a mock, but we can pin two invariants:
      //   1. The v2 call carries the v2 payload + model_version.
      //   2. The options object is the real-upsert form, not
      //      ignoreDuplicates: true (which would re-break the bug).
      mockUpsert.mockResolvedValue({ error: null })

      const POSE_V1: KeypointsJson = {
        fps_sampled: 30,
        frame_count: 1,
        schema_version: 3,
        frames: [
          {
            frame_index: 0,
            timestamp_ms: 0,
            landmarks: [],
            joint_angles: { right_elbow: 90 },
          },
        ],
      }
      const POSE_V2: KeypointsJson = {
        fps_sampled: 30,
        frame_count: 1,
        schema_version: 3,
        frames: [
          {
            frame_index: 0,
            timestamp_ms: 0,
            landmarks: [],
            // Different content than v1 so an accidental "first writer
            // wins" would surface as a payload mismatch.
            joint_angles: { right_elbow: 105 },
          },
        ],
      }

      await setCachedPose(SHA, POSE_V1, 'v1')
      await setCachedPose(SHA, POSE_V2, 'v2')

      expect(mockUpsert).toHaveBeenCalledTimes(2)

      const [v1Payload, v1Opts] = mockUpsert.mock.calls[0]
      const [v2Payload, v2Opts] = mockUpsert.mock.calls[1]

      // First call: v1 payload + v1 model_version.
      expect(v1Payload).toMatchObject({
        sha256: SHA,
        pose_json: POSE_V1,
        model_version: 'v1',
      })

      // Second call MUST hand v2 payload + v2 model_version to
      // upsert — and MUST do so with options that map to a real
      // upsert (not DO NOTHING). Together those are the contract the
      // read-side gate depends on for cache freshness.
      expect(v2Payload).toMatchObject({
        sha256: SHA,
        pose_json: POSE_V2,
        model_version: 'v2',
      })

      // Real-upsert options: onConflict only, no ignoreDuplicates.
      expect(v1Opts).toEqual({ onConflict: 'sha256' })
      expect(v2Opts).toEqual({ onConflict: 'sha256' })
      expect(v1Opts).not.toHaveProperty('ignoreDuplicates')
      expect(v2Opts).not.toHaveProperty('ignoreDuplicates')
    })
  })
})
