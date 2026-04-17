import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase } from '@/lib/supabase'
import type { KeypointsJson, PoseFrame } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

// Sample every Nth frame to keep token count manageable
const SAMPLE_INTERVAL = 5
// Max frames to send per Claude request
const MAX_SAMPLED_FRAMES = 600

interface DetectedSegment {
  start_frame: number
  end_frame: number
  shot_type: 'forehand' | 'backhand' | 'serve' | 'volley' | 'slice' | 'unknown' | 'idle'
  confidence: number
}

function compactFrameData(frames: PoseFrame[], sampleInterval: number): string {
  const sampled: string[] = []
  for (let i = 0; i < frames.length; i += sampleInterval) {
    const f = frames[i]
    const a = f.joint_angles
    // Compact representation: frame_index | timestamp | key angles (rounded)
    sampled.push(
      `${f.frame_index}|${Math.round(f.timestamp_ms)}|` +
      `re:${Math.round(a.right_elbow ?? 0)}|le:${Math.round(a.left_elbow ?? 0)}|` +
      `rs:${Math.round(a.right_shoulder ?? 0)}|ls:${Math.round(a.left_shoulder ?? 0)}|` +
      `rk:${Math.round(a.right_knee ?? 0)}|lk:${Math.round(a.left_knee ?? 0)}|` +
      `hr:${Math.round(a.hip_rotation ?? 0)}|tr:${Math.round(a.trunk_rotation ?? 0)}`
    )
    if (sampled.length >= MAX_SAMPLED_FRAMES) break
  }
  return sampled.join('\n')
}

async function classifySegments(keypointsJson: KeypointsJson): Promise<DetectedSegment[]> {
  const frames = keypointsJson.frames
  if (!frames.length) return []

  const compactData = compactFrameData(frames, SAMPLE_INTERVAL)
  const totalFrames = frames.length
  const fps = keypointsJson.fps_sampled || 30

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 4096,
    system: `You are a tennis video analysis system. You receive time-series joint angle data extracted from a practice video and identify individual shot segments.

FORMAT of each line: frame_index|timestamp_ms|re:right_elbow|le:left_elbow|rs:right_shoulder|ls:left_shoulder|rk:right_knee|lk:left_knee|hr:hip_rotation|tr:trunk_rotation

All values are angles in degrees. Data is sampled every ${SAMPLE_INTERVAL} frames at ${fps}fps.

SHOT DETECTION RULES:
- A shot starts when significant joint motion begins (hip/trunk rotation spike, arm movement)
- A shot ends when the follow-through completes and the player returns to ready position
- FOREHAND: dominant-side hip rotation followed by trunk rotation, then right elbow extension. Right shoulder leads.
- BACKHAND: hip rotation followed by trunk rotation, left elbow extends through contact. Left shoulder leads.
- SERVE: deep knee bend (right_knee drops), trophy position (right_shoulder ~90), then full extension upward.
- VOLLEY: compact motion, minimal backswing (trunk_rotation < 40), quick punch with firm elbow.
- SLICE: shoulder turn with high-to-low swing path, moderate trunk rotation, controlled elbow.
- IDLE: minimal joint angle variation across many frames. Player is standing, walking, or waiting.
- Minimum shot duration: ~15 frames. Minimum idle duration: ~30 frames.
- Use frame_index values from the data, not sampled indices.

Respond with ONLY a JSON array, no other text:
[{"start_frame": N, "end_frame": N, "shot_type": "forehand|backhand|serve|volley|slice|idle|unknown", "confidence": 0.0-1.0}, ...]`,
    messages: [{
      role: 'user',
      content: `Analyze this tennis practice video data (${totalFrames} total frames at ${fps}fps). Identify each shot segment:\n\n${compactData}`,
    }],
  })

  const text = response.content[0]?.type === 'text' ? response.content[0].text : ''

  // Extract JSON array from response (Claude may wrap it in markdown code blocks)
  const jsonMatch = text.match(/\[[\s\S]*\]/)
  if (!jsonMatch) return []

  const parsed = JSON.parse(jsonMatch[0]) as DetectedSegment[]

  // Validate and sanitize
  const validTypes = new Set(['forehand', 'backhand', 'serve', 'volley', 'slice', 'unknown', 'idle'])
  return parsed
    .filter((s) => validTypes.has(s.shot_type) && s.start_frame < s.end_frame)
    .map((s) => ({
      start_frame: Math.max(0, Math.round(s.start_frame)),
      end_frame: Math.min(totalFrames - 1, Math.round(s.end_frame)),
      shot_type: s.shot_type,
      confidence: Math.max(0, Math.min(1, s.confidence)),
    }))
}

export async function POST(request: NextRequest) {
  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }

  const { sessionId, keypointsJson } = body

  if (!sessionId || !keypointsJson?.frames?.length) {
    return NextResponse.json(
      { error: 'sessionId and keypointsJson with frames are required' },
      { status: 400 }
    )
  }

  try {
    // Classify shots from keypoint data
    const segments = await classifySegments(keypointsJson as KeypointsJson)

    if (!segments.length) {
      return NextResponse.json(
        { error: 'No shots detected in the video' },
        { status: 422 }
      )
    }

    const fps = (keypointsJson as KeypointsJson).fps_sampled || 30
    const allFrames = (keypointsJson as KeypointsJson).frames

    // Build segment rows with extracted keypoints per segment
    const segmentRows = segments.map((seg, idx) => {
      const segmentFrames = allFrames.filter(
        (f) => f.frame_index >= seg.start_frame && f.frame_index <= seg.end_frame
      )
      return {
        session_id: sessionId,
        segment_index: idx + 1,
        shot_type: seg.shot_type,
        start_frame: seg.start_frame,
        end_frame: seg.end_frame,
        start_ms: Math.round((seg.start_frame / fps) * 1000),
        end_ms: Math.round((seg.end_frame / fps) * 1000),
        confidence: seg.confidence,
        keypoints_json: {
          fps_sampled: fps,
          frame_count: segmentFrames.length,
          frames: segmentFrames,
        },
      }
    })

    // Insert all segments
    const { error: insertError } = await supabase
      .from('video_segments')
      .insert(segmentRows)

    if (insertError) {
      console.error('[segment] Insert error:', insertError.message)
      return NextResponse.json({ error: 'Failed to save segments' }, { status: 500 })
    }

    // Update session with multi-shot metadata
    const shotSegments = segments.filter((s) => s.shot_type !== 'idle')
    await supabase
      .from('user_sessions')
      .update({
        is_multi_shot: true,
        segment_count: shotSegments.length,
      })
      .eq('id', sessionId)

    return NextResponse.json({
      segments: segmentRows.map((r) => ({
        segment_index: r.segment_index,
        shot_type: r.shot_type,
        start_ms: r.start_ms,
        end_ms: r.end_ms,
        confidence: r.confidence,
      })),
      total: segments.length,
      shots: shotSegments.length,
    })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Segmentation failed'
    console.error('[segment] Error:', msg)
    return NextResponse.json({ error: 'Shot segmentation failed' }, { status: 500 })
  }
}
