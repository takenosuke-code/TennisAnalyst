import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

const BASE_COLUMNS = 'id, pro_id, shot_type, video_url, thumbnail_url, fps, frame_count, duration_ms, phase_labels, metadata, created_at'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const includeKeypoints = request.nextUrl.searchParams.get('include') === 'keypoints'
  const columns = includeKeypoints
    ? `${BASE_COLUMNS}, keypoints_json, pros(*)`
    : `${BASE_COLUMNS}, pros(*)`

  const { data, error } = await supabase
    .from('pro_swings')
    .select(columns)
    .eq('id', id)
    .single()

  if (error) {
    return NextResponse.json({ error: 'Swing not found' }, { status: 404 })
  }

  return NextResponse.json(data)
}
