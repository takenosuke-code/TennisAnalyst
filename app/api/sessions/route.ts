import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

// POST /api/sessions -persist keypoints from client-side MediaPipe extraction
export async function POST(request: NextRequest) {
  let body
  try { body = await request.json() }
  catch { return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 }) }
  const { sessionId, blobUrl, shotType, keypointsJson } = body

  if (!blobUrl || !keypointsJson) {
    return NextResponse.json(
      { error: 'blobUrl and keypointsJson are required' },
      { status: 400 }
    )
  }

  // If sessionId exists, update -require blobUrl to match so callers can't
  // overwrite arbitrary sessions by guessing a UUID
  if (sessionId) {
    const { data, error } = await supabase
      .from('user_sessions')
      .update({
        keypoints_json: keypointsJson,
        status: 'complete',
        shot_type: shotType ?? 'unknown',
      })
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

    return NextResponse.json({ sessionId: data.id })
  }

  const { data, error } = await supabase
    .from('user_sessions')
    .upsert(
      {
        blob_url: blobUrl,
        shot_type: shotType ?? 'unknown',
        keypoints_json: keypointsJson,
        status: 'complete',
      },
      { onConflict: 'blob_url' }
    )
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: 'Failed to save session' }, { status: 500 })
  }

  return NextResponse.json({ sessionId: data.id })
}
