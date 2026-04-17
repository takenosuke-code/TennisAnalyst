import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { supabase } from '@/lib/supabase'
import { buildAngleSummary } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import type { KeypointsJson } from '@/lib/supabase'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string; segmentId: string }> }
) {
  const { sessionId, segmentId } = await params

  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }

  const { proSwingId } = body

  // Fetch the segment's keypoints
  const { data: segment, error: segError } = await supabase
    .from('video_segments')
    .select('id, shot_type, keypoints_json, segment_index')
    .eq('id', segmentId)
    .eq('session_id', sessionId)
    .single()

  if (segError || !segment) {
    return NextResponse.json({ error: 'Segment not found' }, { status: 404 })
  }

  const segKeypoints = segment.keypoints_json as KeypointsJson | null
  if (!segKeypoints?.frames?.length) {
    return NextResponse.json({ error: 'Segment has no keypoints data' }, { status: 400 })
  }

  // Fetch pro swing if requested
  let proSwing: { keypoints_json?: KeypointsJson; pros?: { name: string }; shot_type?: string } | null = null
  if (proSwingId) {
    const { data, error: proError } = await supabase
      .from('pro_swings')
      .select('*, pros(name)')
      .eq('id', proSwingId)
      .single()

    if (proError || !data) {
      return NextResponse.json({ error: 'Pro swing not found' }, { status: 404 })
    }
    proSwing = data
  }

  const userSummary = buildAngleSummary(segKeypoints.frames)
  const shotType = segment.shot_type ?? 'forehand'

  let prompt: string
  if (proSwing) {
    const proKeypoints = proSwing.keypoints_json as KeypointsJson | null
    const proName = proSwing.pros?.name ?? 'Pro player'
    if (!proKeypoints?.frames?.length) {
      return NextResponse.json({ error: 'Pro swing has no keypoints data' }, { status: 400 })
    }
    const proSummary = buildAngleSummary(proKeypoints.frames)
    const strokeRef = getBiomechanicsReference(
      ['forehand', 'backhand', 'serve'].includes(shotType) ? shotType as 'forehand' | 'backhand' | 'serve' : 'all'
    )

    prompt = `You are a tennis coach. This is segment #${segment.segment_index} from a practice video, classified as a ${shotType}. Compare it to ${proName}'s ${proSwing.shot_type ?? shotType}.

STRICT RULES:
- NEVER mention degrees, angles, or numbers. Describe in feel and body language.
- NEVER rate or score. Just give advice.
- NEVER use em dashes. Use commas or periods.
- Reference ${proName} by name. Compare what the player does vs ${proName}.

REFERENCE: ${strokeRef}
USER SWING (segment #${segment.segment_index}): ${userSummary}
${proName.toUpperCase()}: ${proSummary}

## What You're Doing Well
Two or three sentences.

## How to Get Closer to ${proName}'s ${proSwing.shot_type ?? shotType}
Three numbered coaching cues with drills.

## Your Practice Plan
Three one-sentence focus points.

Keep it under 350 words.`
  } else {
    const soloRef = getBiomechanicsReference('all')
    prompt = `You are a tennis coach. This is segment #${segment.segment_index} from a practice video, classified as a ${shotType}.

STRICT RULES:
- NEVER mention degrees, angles, or numbers. Describe in feel and body language.
- NEVER rate or score. Just give advice.
- NEVER use em dashes. Use commas or periods.

REFERENCE: ${soloRef}
USER SWING (segment #${segment.segment_index}): ${userSummary}

## What You're Doing Well
Two or three sentences.

## 3 Things to Work On
Three numbered coaching cues with drills.

## Your Practice Plan
Three one-sentence focus points.

Keep it under 350 words.`
  }

  const messageStream = anthropic.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: `You are a veteran tennis coach. Talk like a real person. Short sentences. Casual tone. Use "you" and "your" constantly. No em dashes. No raw numbers.`,
    messages: [{ role: 'user', content: prompt }],
  })

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of messageStream) {
          if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
            controller.enqueue(encoder.encode(chunk.delta.text))
          }
        }
        controller.close()
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Analysis stream failed'
        controller.enqueue(encoder.encode(`${ERROR_PREFIX}${msg}`))
        controller.close()
      }
    },
  })

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}
