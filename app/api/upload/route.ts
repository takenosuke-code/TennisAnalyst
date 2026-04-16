import { put } from '@vercel/blob'
import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function POST(request: NextRequest) {
  const VALID_SHOT_TYPES = new Set(['forehand', 'backhand', 'serve', 'volley', 'unknown'])

  const formData = await request.formData()
  const file = formData.get('video') as File | null
  const rawShotType = (formData.get('shot_type') as string | null) ?? 'unknown'
  // Sanitize to prevent DB constraint violations from leaking Postgres errors
  const shotType = VALID_SHOT_TYPES.has(rawShotType) ? rawShotType : 'unknown'

  if (!file) {
    return NextResponse.json({ error: 'No video file provided' }, { status: 400 })
  }

  // Validate file type
  if (!file.type.startsWith('video/')) {
    return NextResponse.json({ error: 'File must be a video' }, { status: 400 })
  }

  // Max 200MB
  if (file.size > 200 * 1024 * 1024) {
    return NextResponse.json({ error: 'File too large (max 200MB)' }, { status: 400 })
  }

  // Sanitize filename: strip path components and disallowed characters
  const safeName = file.name
    .split(/[\\/]/).pop()!
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .slice(0, 100) || 'upload'

  const blobPath = `videos/${Date.now()}-${safeName}`
  const blob = await put(blobPath, file, { access: 'public' }).catch(() =>
    put(blobPath, file, { access: 'private' })
  )

  const { data: session, error } = await supabase
    .from('user_sessions')
    .insert({
      blob_url: blob.url,
      shot_type: shotType,
      status: 'uploaded',
    })
    .select()
    .single()

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({
    sessionId: session.id,
    blobUrl: blob.url,
    status: 'uploaded',
  })
}
