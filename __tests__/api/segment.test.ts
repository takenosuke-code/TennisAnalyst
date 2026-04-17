import { describe, it, expect, vi, beforeEach } from 'vitest'
import { NextRequest } from 'next/server'

// Mock Anthropic
const mockCreate = vi.fn()
vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    messages = { create: mockCreate }
  },
}))

// Mock supabase
const mockInsert = vi.fn()
const mockUpdate = vi.fn()
const mockEq = vi.fn()
const mockFrom = vi.fn()

vi.mock('@/lib/supabase', () => ({
  supabase: new Proxy({}, {
    get: (_t: object, prop: string) => {
      if (prop === 'from') return mockFrom
      return undefined
    },
  }),
}))

function setupMocks() {
  mockInsert.mockReturnValue({ error: null })
  mockUpdate.mockReturnValue({ eq: mockEq })
  mockEq.mockReturnValue({ error: null })

  mockFrom.mockImplementation((table: string) => {
    if (table === 'video_segments') return { insert: mockInsert }
    if (table === 'user_sessions') return { update: mockUpdate }
    return {}
  })
}

function makeFakeKeypoints(frameCount: number) {
  return {
    fps_sampled: 30,
    frame_count: frameCount,
    frames: Array.from({ length: frameCount }, (_, i) => ({
      frame_index: i,
      timestamp_ms: (i / 30) * 1000,
      landmarks: [],
      joint_angles: {
        right_elbow: 120 + Math.sin(i / 10) * 30,
        left_elbow: 130,
        right_shoulder: 90,
        left_shoulder: 90,
        right_knee: 150,
        left_knee: 150,
        hip_rotation: 40 + Math.sin(i / 15) * 20,
        trunk_rotation: 80 + Math.sin(i / 12) * 40,
      },
    })),
  }
}

describe('POST /api/segment', () => {
  beforeEach(() => {
    vi.resetModules()
    mockFrom.mockClear()
    mockInsert.mockClear()
    mockUpdate.mockClear()
    mockEq.mockClear()
    mockCreate.mockClear()
    setupMocks()
  })

  it('returns 400 when sessionId is missing', async () => {
    const { POST } = await import('@/app/api/segment/route')
    const req = new NextRequest('http://localhost/api/segment', {
      method: 'POST',
      body: JSON.stringify({ keypointsJson: makeFakeKeypoints(100) }),
    })
    const res = await POST(req)
    expect(res.status).toBe(400)
  })

  it('returns 400 when keypointsJson has no frames', async () => {
    const { POST } = await import('@/app/api/segment/route')
    const req = new NextRequest('http://localhost/api/segment', {
      method: 'POST',
      body: JSON.stringify({
        sessionId: 'sess-1',
        keypointsJson: { fps_sampled: 30, frame_count: 0, frames: [] },
      }),
    })
    const res = await POST(req)
    expect(res.status).toBe(400)
  })

  it('calls Claude and inserts detected segments', async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{
        type: 'text',
        text: JSON.stringify([
          { start_frame: 0, end_frame: 30, shot_type: 'forehand', confidence: 0.9 },
          { start_frame: 31, end_frame: 50, shot_type: 'idle', confidence: 0.8 },
          { start_frame: 51, end_frame: 90, shot_type: 'backhand', confidence: 0.85 },
        ]),
      }],
    })

    const { POST } = await import('@/app/api/segment/route')
    const req = new NextRequest('http://localhost/api/segment', {
      method: 'POST',
      body: JSON.stringify({
        sessionId: 'sess-1',
        keypointsJson: makeFakeKeypoints(100),
      }),
    })
    const res = await POST(req)
    expect(res.status).toBe(200)

    const json = await res.json()
    expect(json.total).toBe(3)
    expect(json.shots).toBe(2) // forehand + backhand, not idle
    expect(json.segments).toHaveLength(3)
    expect(json.segments[0].shot_type).toBe('forehand')
    expect(json.segments[2].shot_type).toBe('backhand')

    // Verify insert was called
    expect(mockInsert).toHaveBeenCalledTimes(1)
  })

  it('returns 422 when Claude detects no shots', async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: 'text', text: '[]' }],
    })

    const { POST } = await import('@/app/api/segment/route')
    const req = new NextRequest('http://localhost/api/segment', {
      method: 'POST',
      body: JSON.stringify({
        sessionId: 'sess-1',
        keypointsJson: makeFakeKeypoints(100),
      }),
    })
    const res = await POST(req)
    expect(res.status).toBe(422)
  })

  it('filters out invalid segment types from Claude response', async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{
        type: 'text',
        text: JSON.stringify([
          { start_frame: 0, end_frame: 30, shot_type: 'forehand', confidence: 0.9 },
          { start_frame: 31, end_frame: 50, shot_type: 'dropshot', confidence: 0.5 },
        ]),
      }],
    })

    const { POST } = await import('@/app/api/segment/route')
    const req = new NextRequest('http://localhost/api/segment', {
      method: 'POST',
      body: JSON.stringify({
        sessionId: 'sess-1',
        keypointsJson: makeFakeKeypoints(100),
      }),
    })
    const res = await POST(req)
    expect(res.status).toBe(200)

    const json = await res.json()
    // Only forehand should survive; 'dropshot' is not a valid type
    expect(json.total).toBe(1)
    expect(json.segments[0].shot_type).toBe('forehand')
  })
})
