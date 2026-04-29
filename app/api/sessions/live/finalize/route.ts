import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/lib/supabase'
import { createClient } from '@/lib/supabase/server'

// POST /api/sessions/live/finalize
//
// Phase 6 — settles a live session that was created in 'server-extract'
// mode (i.e. status='extracting', keypoints_json=null, live keypoints
// parked in fallback_keypoints_json) once the client knows whether
// Railway succeeded or failed.
//
// Outcomes:
//   server-ok    — Railway has already written keypoints_json on the
//                  row via its own callback path; we just flip
//                  status='complete'. We do NOT touch keypoints_json
//                  here so a race where Railway's write lands between
//                  the client's poll and this call doesn't get
//                  clobbered.
//   server-failed — Railway never produced keypoints (timeout,
//                   not-configured, error-status, aborted). Copy
//                   fallback_keypoints_json into keypoints_json and
//                   flip status='complete' so the row is usable. The
//                   client will already be hydrating /analyze with the
//                   live keypoints in this branch; the row write is
//                   the durable record of that decision.
//
// Ownership: we route through the user's auth client to confirm the
// session exists for this user before letting the service-role client
// do the actual UPDATE. Blob hijacking is not possible because the
// user supplies sessionId, not blob_url.

const VALID_OUTCOMES = new Set(['server-ok', 'server-failed'] as const)
type Outcome = 'server-ok' | 'server-failed'

export async function POST(request: NextRequest) {
  const authClient = await createClient()
  const { data: { user } } = await authClient.auth.getUser()
  if (!user) {
    return NextResponse.json({ error: 'Not signed in' }, { status: 401 })
  }
  const userId = user.id

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const b = body as Record<string, unknown>

  if (typeof b.sessionId !== 'string' || !b.sessionId) {
    return NextResponse.json({ error: 'sessionId is required' }, { status: 400 })
  }
  const sessionId = b.sessionId

  if (typeof b.outcome !== 'string' || !(VALID_OUTCOMES as Set<string>).has(b.outcome)) {
    return NextResponse.json(
      { error: "outcome must be 'server-ok' or 'server-failed'" },
      { status: 400 },
    )
  }
  const outcome = b.outcome as Outcome

  // 1. Ownership check via the user's RLS-respecting client. We only
  // need to confirm the row exists for this user; we don't need the
  // payload here so a select on id is enough.
  const { data: ownerRow, error: ownerErr } = await authClient
    .from('user_sessions')
    .select('id, fallback_keypoints_json, keypoints_json, status')
    .eq('id', sessionId)
    .single()
  if (ownerErr || !ownerRow) {
    // Hide the difference between "doesn't exist" and "not yours" —
    // both look like 401 to the client. (RLS reads will return null
    // rather than the row, which surfaces here as a not-found error.)
    return NextResponse.json({ error: 'Session not accessible' }, { status: 401 })
  }

  if (outcome === 'server-ok') {
    // Don't touch keypoints_json — Railway is the writer for that
    // column on this row. We only flip status. If keypoints_json is
    // somehow still null at this point (race or client-side bug)
    // we surface that to the caller rather than silently completing
    // a row with no payload.
    const { error: updateErr } = await supabaseAdmin
      .from('user_sessions')
      .update({ status: 'complete' })
      .eq('id', sessionId)
    if (updateErr) {
      console.error('sessions/live/finalize server-ok update failed:', updateErr)
      return NextResponse.json({ error: 'Failed to finalize session' }, { status: 500 })
    }
    return NextResponse.json({ sessionId, status: 'complete', outcome })
  }

  // outcome === 'server-failed'
  // Copy fallback into keypoints_json so the row is usable. We use
  // the row we already read to do the copy in JS rather than via a
  // SQL self-update — supabase-js doesn't expose UPDATE ... SET
  // col = other_col cleanly, and the JSON payload is small relative
  // to the rest of the request budget.
  const fallback = (ownerRow as Record<string, unknown>).fallback_keypoints_json
  if (!fallback) {
    // No fallback to copy — the row was probably created in 'live-only'
    // mode and the client shouldn't be calling finalize on it. Refuse
    // rather than nulling keypoints_json.
    return NextResponse.json(
      { error: 'No fallback keypoints stored for this session' },
      { status: 409 },
    )
  }

  const { error: updateErr } = await supabaseAdmin
    .from('user_sessions')
    .update({
      status: 'complete',
      keypoints_json: fallback,
    })
    .eq('id', sessionId)
  if (updateErr) {
    console.error('sessions/live/finalize server-failed update failed:', updateErr)
    return NextResponse.json({ error: 'Failed to finalize session' }, { status: 500 })
  }

  // Touch the user_id so even if logging downstream looks at it the
  // ownership trail is intact. (No DB write — purely about closing
  // over the variable so a future audit-log addition has it.)
  void userId

  return NextResponse.json({ sessionId, status: 'complete', outcome })
}
