import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'

/*
 * Phase 4.1 — Create a coach review request.
 *
 * Body:
 *   {
 *     sessionId: string,
 *     blobUrl: string,
 *     cueTitle: string,
 *     cueBody: string,
 *     coachLabel?: string,
 *   }
 *
 * Returns: { id, shareUrl } where shareUrl is the public page the user
 * sends to their coach (typically by text/copy-paste — there's no
 * server-side email/SMS this round).
 */

export async function POST(request: NextRequest) {
  // Auth: must be a signed-in user. Anonymous "send a clip to a coach"
  // is too easy to abuse without auth + rate-limiting.
  const authClient = await createClient()
  const { data: { user } } = await authClient.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Sign in required' }, { status: 401 })
  }

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const sessionId = typeof body?.sessionId === 'string' ? body.sessionId : null
  const blobUrl = typeof body?.blobUrl === 'string' ? body.blobUrl : null
  const cueTitle = typeof body?.cueTitle === 'string' ? body.cueTitle.trim() : ''
  const cueBody = typeof body?.cueBody === 'string' ? body.cueBody.trim() : ''
  const coachLabel = typeof body?.coachLabel === 'string' ? body.coachLabel.trim().slice(0, 80) : null

  if (!blobUrl || !cueTitle || !cueBody) {
    return NextResponse.json(
      { error: 'blobUrl, cueTitle, cueBody are required' },
      { status: 400 },
    )
  }
  if (cueTitle.length > 200 || cueBody.length > 2000) {
    return NextResponse.json({ error: 'cue too long' }, { status: 400 })
  }

  const { data, error } = await supabaseAdmin
    .from('coach_reviews')
    .insert({
      user_id: user.id,
      session_id: sessionId,
      blob_url: blobUrl,
      cue_title: cueTitle,
      cue_body: cueBody,
      coach_label: coachLabel,
    })
    .select('id')
    .single()

  if (error || !data) {
    return NextResponse.json({ error: error?.message ?? 'create failed' }, { status: 500 })
  }

  // Build the public share URL. Use the request origin so dev / preview
  // / prod all produce links that resolve back to the same deployment.
  const origin = request.nextUrl.origin
  const shareUrl = `${origin}/coach-review/${data.id}`

  return NextResponse.json({ id: data.id, shareUrl })
}

/*
 * GET — list reviews owned by the signed-in user. Used by the analyze
 * page to render "Pending coach review" / "Reviewed by Marco" badges
 * next to the relevant cue.
 */
export async function GET(request: NextRequest) {
  const authClient = await createClient()
  const { data: { user } } = await authClient.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Sign in required' }, { status: 401 })
  }

  const sessionId = request.nextUrl.searchParams.get('sessionId')
  let q = supabaseAdmin
    .from('coach_reviews')
    .select('id, session_id, cue_title, cue_body, coach_label, verdict, note, responded_at, created_at')
    .eq('user_id', user.id)
    .is('deleted_at', null)
    .order('created_at', { ascending: false })

  if (sessionId) q = q.eq('session_id', sessionId)

  const { data, error } = await q
  if (error) return NextResponse.json({ error: error.message }, { status: 500 })
  return NextResponse.json({ reviews: data ?? [] })
}
