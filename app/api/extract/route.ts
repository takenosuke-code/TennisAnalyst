import { NextRequest, NextResponse } from 'next/server'

// POST /api/extract — proxy that asks Railway to run server-side pose
// extraction on an already-uploaded Vercel Blob.
//
// Request body: { sessionId: string, blobUrl: string }
//
// Responses:
//   200 { status: 'queued' }        — Railway accepted the job
//   503 { error: 'railway-not-configured' } — RAILWAY_SERVICE_URL or
//       EXTRACT_API_KEY are unset. Client should fall back to browser
//       MediaPipe extraction so the upload flow still completes.
//   4xx/5xx from Railway are propagated so the client can log + fall
//   back.
//
// This is the only place the EXTRACT_API_KEY lives — never sent to the
// browser. Pattern mirrors app/api/debug/racket/route.ts and
// app/api/process-youtube/route.ts.

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY

export async function POST(request: NextRequest) {
  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    return NextResponse.json(
      { error: 'railway-not-configured' },
      { status: 503 }
    )
  }

  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { sessionId, blobUrl } = body
  if (!sessionId || !blobUrl) {
    return NextResponse.json(
      { error: 'sessionId and blobUrl are required' },
      { status: 400 }
    )
  }

  try {
    const resp = await fetch(`${RAILWAY_SERVICE_URL}/extract`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${EXTRACT_API_KEY}`,
      },
      body: JSON.stringify({
        video_url: blobUrl,
        session_id: sessionId,
      }),
    })

    const respBody = await resp.text()
    let parsed: unknown
    try {
      parsed = JSON.parse(respBody)
    } catch {
      parsed = { raw: respBody }
    }

    if (!resp.ok) {
      console.error('[extract] Railway rejected job', resp.status, parsed)
      return NextResponse.json(
        { error: 'railway-error', status: resp.status, body: parsed },
        { status: 502 }
      )
    }

    return NextResponse.json(parsed)
  } catch (err) {
    // Network error reaching Railway. Treat as a 503 so the client
    // gets the same "fall back to browser" signal as when the service
    // isn't configured.
    console.error('[extract] Railway network error', err)
    return NextResponse.json(
      { error: 'railway-unreachable' },
      { status: 503 }
    )
  }
}
