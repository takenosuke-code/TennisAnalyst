import { describe, it, expect } from 'vitest'
import { getProVideoUrl } from '@/lib/proVideoUrl'
import type { ProSwing } from '@/lib/supabase'

function makeSwing(overrides: Partial<ProSwing> = {}): ProSwing {
  return {
    id: 'swing-1',
    pro_id: 'pro-1',
    shot_type: 'forehand',
    video_url: 'https://example.com/video.mp4',
    thumbnail_url: null,
    keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
    fps: 30,
    frame_count: null,
    duration_ms: null,
    phase_labels: {},
    metadata: {},
    created_at: '2025-01-01',
    ...overrides,
  }
}

describe('getProVideoUrl', () => {
  it('returns null for null swing', () => {
    expect(getProVideoUrl(null)).toBeNull()
  })

  it('YouTube URL returns null (not playable in video element)', () => {
    const swing = makeSwing({
      shot_type: 'forehand',
      video_url: 'https://www.youtube.com/watch?v=abc123',
    })
    expect(getProVideoUrl(swing)).toBeNull()
  })

  it('YouTube short URL returns null', () => {
    const swing = makeSwing({
      shot_type: 'serve',
      video_url: 'https://youtu.be/short123',
    })
    expect(getProVideoUrl(swing)).toBeNull()
  })

  it('normal (non-YouTube) URL passes through unchanged', () => {
    const swing = makeSwing({
      video_url: 'https://cdn.example.com/pro/federer_forehand.mp4',
    })
    expect(getProVideoUrl(swing)).toBe('https://cdn.example.com/pro/federer_forehand.mp4')
  })

  it('youtu.be short URL is recognized as YouTube', () => {
    const swing = makeSwing({
      shot_type: 'forehand',
      video_url: 'https://youtu.be/abc123',
    })
    expect(getProVideoUrl(swing)).toBeNull()
  })

  it('returns null when video_url is null', () => {
    const swing = makeSwing({ video_url: null as unknown as string })
    expect(getProVideoUrl(swing)).toBeNull()
  })
})
