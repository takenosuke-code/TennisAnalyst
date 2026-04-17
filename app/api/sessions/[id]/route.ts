import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params

  const { data, error } = await supabase
    .from('user_sessions')
    .select('id, blob_url, shot_type, status, error_message, created_at')
    .eq('id', id)
    .gt('expires_at', new Date().toISOString())
    .single()

  if (error) {
    return NextResponse.json({ error: 'Session not found' }, { status: 404 })
  }

  return NextResponse.json(data)
}
