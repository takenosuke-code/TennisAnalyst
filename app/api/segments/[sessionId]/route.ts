import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await params

  const { data, error } = await supabase
    .from('video_segments')
    .select('id, session_id, segment_index, shot_type, start_frame, end_frame, start_ms, end_ms, confidence, label, matched_pro_swing_id, created_at')
    .eq('session_id', sessionId)
    .order('segment_index', { ascending: true })

  if (error) {
    return NextResponse.json({ error: 'Failed to fetch segments' }, { status: 500 })
  }

  return NextResponse.json(data ?? [])
}
