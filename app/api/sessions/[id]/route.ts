import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

// GET /api/sessions/[id]
//
// Dual-purpose:
//   • Default: returns session metadata (blob_url, shot_type, status, etc.).
//     Back-compat with existing consumers.
//   • `?include=keypoints`: also returns keypoints_json when the session is
//     complete. This is the polling endpoint for the Railway-extraction
//     path — the client calls this at ~3s intervals after /api/extract
//     queues a job, and reads keypoints_json once status flips to
//     'complete'. Kept behind a query param so we don't spray large
//     keypoint payloads to every caller that just wants metadata.
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const includeKeypoints = request.nextUrl.searchParams.get('include') === 'keypoints'

  // Phase 6: when a live session opted into server-extraction we stash
  // the live keypoints in fallback_keypoints_json. Surfacing it here as
  // a sibling field (NOT mapped onto keypoints_json) keeps the client's
  // logic unambiguous: keypoints_json present => server extraction
  // succeeded; keypoints_json null + fallback_keypoints_json present =>
  // extraction is mid-flight or failed and the caller should decide
  // whether to wait or fall back.
  const columns = includeKeypoints
    ? 'id, blob_url, shot_type, status, error_message, created_at, keypoints_json, fallback_keypoints_json'
    : 'id, blob_url, shot_type, status, error_message, created_at'

  const { data, error } = await supabase
    .from('user_sessions')
    .select(columns)
    .eq('id', id)
    .gt('expires_at', new Date().toISOString())
    .single()

  if (error) {
    return NextResponse.json({ error: 'Session not found' }, { status: 404 })
  }

  return NextResponse.json(data)
}
