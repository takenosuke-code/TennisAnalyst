import { NextResponse } from 'next/server'

// Proxies to Railway's /debug/racket-detector so we can diagnose "Racket
// detected: 0 / N" failures from the browser without needing Railway
// dashboard access. Surfaces whether the ONNX file is on disk, whether
// the session loads, and whether blank-image inference runs.
const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY

export async function GET() {
  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    return NextResponse.json(
      { error: 'Railway service is not configured on this deploy' },
      { status: 503 },
    )
  }

  try {
    const resp = await fetch(`${RAILWAY_SERVICE_URL}/debug/racket-detector`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${EXTRACT_API_KEY}` },
    })
    const body = await resp.json().catch(() => ({ error: 'non-json response' }))
    return NextResponse.json(
      { railway_url: RAILWAY_SERVICE_URL, status: resp.status, body },
      { status: 200 },
    )
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    return NextResponse.json(
      { error: 'Failed to reach Railway', detail: msg },
      { status: 502 },
    )
  }
}
