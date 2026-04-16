import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params

  const { data, error } = await supabase
    .from('pro_swings')
    .select('*, pros(*)')
    .eq('id', id)
    .single()

  if (error) {
    return NextResponse.json({ error: 'Swing not found' }, { status: 404 })
  }

  return NextResponse.json(data)
}
