import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

// POST /api/sessions
//
// Two shapes, decided by whether keypointsJson is present in the body:
//
// 1. Browser-extraction path (original): caller finished MediaPipe in-browser
//    and submits the final keypoints. Row is created with status='complete'.
//
// 2. Railway-extraction path (new): caller uploaded the blob and wants the
//    row to exist BEFORE asking Railway to extract. Keypoints come back
//    later via /api/extract -> Railway -> supabase update. The row is
//    created with status='pending' and no keypoints_json. The caller polls
//    /api/sessions/[id] for completion.
export async function POST(request: NextRequest) {
  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const { sessionId, blobUrl, shotType, keypointsJson } = body

  if (!blobUrl) {
    return NextResponse.json({ error: 'blobUrl is required' }, { status: 400 })
  }

  const hasKeypoints = !!keypointsJson
  const status = hasKeypoints ? 'complete' : 'pending'

  // If sessionId exists, update -require blobUrl to match so callers can't
  // overwrite arbitrary sessions by guessing a UUID
  if (sessionId) {
    const updates: Record<string, unknown> = {
      status,
      shot_type: shotType ?? 'unknown',
    }
    if (hasKeypoints) updates.keypoints_json = keypointsJson

    const { data, error } = await supabase
      .from('user_sessions')
      .update(updates)
      .eq('id', sessionId)
      .eq('blob_url', blobUrl) // ownership check
      .select()
      .single()

    if (error || !data) {
      return NextResponse.json(
        { error: 'Session not found or blob_url mismatch' },
        { status: 404 }
      )
    }

    return NextResponse.json({ sessionId: data.id, status: data.status })
  }

  const upsertPayload: Record<string, unknown> = {
    blob_url: blobUrl,
    shot_type: shotType ?? 'unknown',
    status,
  }
  if (hasKeypoints) upsertPayload.keypoints_json = keypointsJson

  const { data, error } = await supabase
    .from('user_sessions')
    .upsert(upsertPayload, { onConflict: 'blob_url' })
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: 'Failed to save session' }, { status: 500 })
  }

  return NextResponse.json({ sessionId: data.id, status: data.status })
}
