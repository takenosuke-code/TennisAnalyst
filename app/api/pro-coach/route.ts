import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { buildAngleSummary } from '@/lib/jointAngles'
import { getBiomechanicsReference } from '@/lib/biomechanics-reference'
import { streamGemini } from '@/lib/gemini'
import type { KeypointsJson } from '@/lib/supabase'

export async function POST(request: NextRequest) {
  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const { proSwingId, messages } = body

  if (!proSwingId || typeof proSwingId !== 'string') {
    return NextResponse.json(
      { error: 'proSwingId is required' },
      { status: 400 }
    )
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return NextResponse.json(
      { error: 'messages array is required and must not be empty' },
      { status: 400 }
    )
  }

  // Fetch pro swing with pro info
  const { data: proSwing, error: proError } = await supabase
    .from('pro_swings')
    .select('*, pros(*)')
    .eq('id', proSwingId)
    .single()

  if (proError || !proSwing) {
    return NextResponse.json(
      { error: 'Pro swing not found' },
      { status: 404 }
    )
  }

  const proKeypoints: KeypointsJson | null = proSwing.keypoints_json ?? null
  const proName = proSwing.pros?.name ?? 'Pro player'
  const shotType = proSwing.shot_type ?? 'forehand'

  if (!proKeypoints?.frames?.length) {
    return NextResponse.json(
      { error: 'Pro swing has no keypoints data. It may not have been processed yet.' },
      { status: 400 }
    )
  }

  // Build reference data
  const strokeRef = getBiomechanicsReference(
    ['forehand', 'backhand', 'serve'].includes(shotType)
      ? (shotType as 'forehand' | 'backhand' | 'serve')
      : 'all'
  )
  const proSummary = buildAngleSummary(proKeypoints.frames)

  // Build system prompt
  const systemPrompt = `You are a tennis coach answering questions about ${proName}'s ${shotType}. A player is watching the swing video and asking you about it.

RULES:
- Keep answers SHORT. 2-3 sentences max unless they ask you to explain more.
- NEVER use degree symbols, numbers, or measurements. Describe positions in plain English: "arm is almost straight", "knees are deeply bent", "hips are turned sideways", "racket is pointing at the back fence".
- NEVER use em dashes. Use periods or commas.
- Reference ${proName} by first name naturally.
- Talk like a real coach on court. Casual, direct.
- If they want more detail, they'll ask. Don't over-explain.

You have ${proName}'s swing data below for reference. Use it to inform your answers but translate everything into plain body language descriptions. The player never sees these numbers.

REFERENCE: ${strokeRef}

${proName.toUpperCase()}'S DATA: ${proSummary}`

  // Limit messages to last 20 to avoid token overflow
  const truncatedMessages = messages.length > 20
    ? messages.slice(messages.length - 20)
    : messages

  // Ensure conversation starts with a user message after truncation
  const validMessages = truncatedMessages[0]?.role === 'assistant'
    ? truncatedMessages.slice(1)
    : truncatedMessages

  const encoder = new TextEncoder()
  const ERROR_PREFIX = '\n\n[ERROR] '
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const iter = streamGemini({
          systemPrompt,
          messages: validMessages.map((m: { role: string; content: string }) => ({
            role: m.role as 'user' | 'assistant',
            content: m.content,
          })),
          maxTokens: 400,
        })
        for await (const delta of iter) {
          controller.enqueue(encoder.encode(delta))
        }
        controller.close()
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Chat stream failed'
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
