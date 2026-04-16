import { NextRequest, NextResponse } from 'next/server'

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY

export async function POST(request: NextRequest) {
  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    return NextResponse.json(
      { error: 'YouTube processing service is not configured' },
      { status: 503 }
    )
  }

  const body = await request.json()
  const { youtubeUrl, targetShotTypes, maxDuration } = body

  if (!youtubeUrl || typeof youtubeUrl !== 'string') {
    return NextResponse.json(
      { error: 'youtubeUrl is required' },
      { status: 400 }
    )
  }

  // Basic client-side URL validation before forwarding
  const urlPattern = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//
  if (!urlPattern.test(youtubeUrl)) {
    return NextResponse.json(
      { error: 'Invalid YouTube URL' },
      { status: 400 }
    )
  }

  try {
    const resp = await fetch(`${RAILWAY_SERVICE_URL}/process-youtube`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${EXTRACT_API_KEY}`,
      },
      body: JSON.stringify({
        youtube_url: youtubeUrl,
        target_shot_types: targetShotTypes ?? null,
        max_duration: maxDuration ?? 600,
      }),
    })

    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      return NextResponse.json(
        { error: errBody.detail ?? 'Processing service error' },
        { status: resp.status }
      )
    }

    const data = await resp.json()
    return NextResponse.json(data)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to reach processing service'
    console.error('YouTube processing proxy error:', message)
    return NextResponse.json(
      { error: 'Failed to reach processing service' },
      { status: 502 }
    )
  }
}
