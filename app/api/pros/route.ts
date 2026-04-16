import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const shotType = searchParams.get('shot_type')
  const proId = searchParams.get('pro_id')

  try {
    // Build the embedded resource select string.
    // Use !inner when filtering by shot_type so PostgREST applies it as a
    // restricting join (only pros with matching swings are returned).
    const swingsSelect = shotType
      ? `pro_swings!inner(id,shot_type,video_url,thumbnail_url,fps,frame_count,duration_ms,phase_labels,metadata)`
      : `pro_swings(id,shot_type,video_url,thumbnail_url,fps,frame_count,duration_ms,phase_labels,metadata)`

    let query = supabase
      .from('pros')
      .select(`*, ${swingsSelect}`)
      .order('name')

    if (proId) {
      query = query.eq('id', proId)
    }

    if (shotType) {
      query = query.eq('pro_swings.shot_type', shotType)
    }

    const { data, error } = await query

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    return NextResponse.json(data)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    console.error('[/api/pros] Failed to fetch:', message)
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
