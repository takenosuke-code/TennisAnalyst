import { NextRequest, NextResponse } from 'next/server'

const RAILWAY_SERVICE_URL = process.env.RAILWAY_SERVICE_URL
const EXTRACT_API_KEY = process.env.EXTRACT_API_KEY

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  if (!RAILWAY_SERVICE_URL || !EXTRACT_API_KEY) {
    return NextResponse.json(
      { error: 'YouTube processing service is not configured' },
      { status: 503 }
    )
  }

  const { jobId } = await params

  try {
    const resp = await fetch(
      `${RAILWAY_SERVICE_URL}/process-youtube/${encodeURIComponent(jobId)}`,
      {
        headers: {
          Authorization: `Bearer ${EXTRACT_API_KEY}`,
        },
      }
    )

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
    console.error('YouTube job status proxy error:', message)
    return NextResponse.json(
      { error: 'Failed to reach processing service' },
      { status: 502 }
    )
  }
}
