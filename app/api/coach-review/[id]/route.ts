import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/lib/supabase'

/*
 * Phase 4.1 — Public coach-review endpoint.
 *
 * GET fetches the review payload for the public coach page (no auth —
 * access is gated by the unguessable review id).
 *
 * POST records the coach's verdict + optional note. Same auth model:
 * whoever has the link can respond. Once verdict is set we lock the
 * row — coaches can't overwrite an earlier response.
 */

type RouteCtx = { params: Promise<{ id: string }> }

export async function GET(_request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params
  if (!id) return NextResponse.json({ error: 'id required' }, { status: 400 })

  const { data, error } = await supabaseAdmin
    .from('coach_reviews')
    .select('id, blob_url, cue_title, cue_body, coach_label, verdict, note, responded_at, created_at, deleted_at')
    .eq('id', id)
    .maybeSingle()

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })
  if (!data || data.deleted_at) {
    return NextResponse.json({ error: 'Review not found' }, { status: 404 })
  }

  return NextResponse.json({ review: data })
}

export async function POST(request: NextRequest, ctx: RouteCtx) {
  const { id } = await ctx.params
  if (!id) return NextResponse.json({ error: 'id required' }, { status: 400 })

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const verdict = body?.verdict
  if (verdict !== 'looks_right' && verdict !== 'add_this') {
    return NextResponse.json({ error: 'verdict must be looks_right or add_this' }, { status: 400 })
  }
  const note = typeof body?.note === 'string' ? body.note.trim().slice(0, 1000) : null

  // Don't allow re-submitting an already-responded review.
  const { data: existing } = await supabaseAdmin
    .from('coach_reviews')
    .select('id, responded_at, deleted_at')
    .eq('id', id)
    .maybeSingle()

  if (!existing || existing.deleted_at) {
    return NextResponse.json({ error: 'Review not found' }, { status: 404 })
  }
  if (existing.responded_at) {
    return NextResponse.json({ error: 'Review already responded to' }, { status: 409 })
  }

  const { data, error } = await supabaseAdmin
    .from('coach_reviews')
    .update({
      verdict,
      note,
      responded_at: new Date().toISOString(),
    })
    .eq('id', id)
    .is('responded_at', null)
    .select('id, verdict, note, responded_at')
    .single()

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })
  return NextResponse.json({ review: data })
}
