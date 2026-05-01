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
    it('writes via upsert with onConflict=sha256 + ignoreDuplicates', async () => {
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
      expect(options).toEqual({
        onConflict: 'sha256',
        ignoreDuplicates: true,
      })
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

    it('concurrent setCachedPose calls for same sha256: both resolve, no error from the duplicate', async () => {
      // Postgres ON CONFLICT (sha256) DO NOTHING returns no error and
      // no inserted row when the second writer hits a duplicate. We
      // assert both calls resolve cleanly — the second one is a
      // silent no-op and must not throw.
      mockUpsert.mockResolvedValue({ error: null })
      const a = setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-modal')
      const b = setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-modal')
      await expect(Promise.all([a, b])).resolves.toBeDefined()
      expect(mockUpsert).toHaveBeenCalledTimes(2)
      // Both calls used the conflict-tolerant options.
      for (const [, opts] of mockUpsert.mock.calls) {
        expect(opts).toEqual({
          onConflict: 'sha256',
          ignoreDuplicates: true,
        })
      }
    })

    it('different model_versions for the same sha256: BOTH attempts use the same PK so the second is a no-op', async () => {
      // Documented semantics: PK is sha256 only. The cache stores ONE
      // row per content hash, model_version is a tag. The "latest
      // wins" knob is owned by the read-side gate (getCachedPose
      // expectedModelVersion arg) — old rows become misses on
      // version mismatch and get rewritten on the next extraction
      // when we explicitly UPSERT (not in this minimal contract).
      // For setCachedPose: ON CONFLICT DO NOTHING means the row that
      // landed first sticks. This test pins that behavior.
      mockUpsert.mockResolvedValue({ error: null })
      await setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-m')
      await setCachedPose(SHA, SAMPLE_POSE, 'rtmpose-l')
      // Both calls hit upsert with their respective model_versions —
      // the database-side ON CONFLICT decides which one persists.
      expect(mockUpsert).toHaveBeenCalledTimes(2)
      expect(mockUpsert.mock.calls[0][0]).toMatchObject({
        model_version: 'rtmpose-m',
      })
      expect(mockUpsert.mock.calls[1][0]).toMatchObject({
        model_version: 'rtmpose-l',
      })
    })
  })
})
