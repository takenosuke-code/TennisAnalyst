import { NextRequest, NextResponse } from 'next/server'
import { put, list } from '@vercel/blob'

/*
 * Phase 0.5 telemetry sink.
 *
 * Catches per-event records from the client + server and appends them
 * to a daily CSV in Vercel Blob (`telemetry/YYYY-MM-DD.csv`). Used to
 * watch for production regressions when shipping Phase 2 (TRT engines)
 * and to track user-funnel drops the synthesis flagged (homepage →
 * upload → analyze).
 *
 * Hard contract: telemetry MUST NEVER throw and break the request that
 * called it. All errors are swallowed and logged; the endpoint always
 * returns 204.
 */

type TelemetryEvent = {
  event: string
  sessionId?: string
  cold_start?: boolean
  inference_ms?: number
  download_ms?: number
  provider?: string
  angle_mae_vs_baseline?: number
  extra?: Record<string, unknown>
}

const HEADERS = [
  'ts',
  'event',
  'sessionId',
  'cold_start',
  'inference_ms',
  'download_ms',
  'provider',
  'angle_mae_vs_baseline',
  'extra',
] as const

function todayKey(): string {
  const d = new Date()
  const yyyy = d.getUTCFullYear()
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0')
  const dd = String(d.getUTCDate()).padStart(2, '0')
  return `telemetry/${yyyy}-${mm}-${dd}.csv`
}

function csvField(v: unknown): string {
  if (v === undefined || v === null) return ''
  const s = typeof v === 'string' ? v : JSON.stringify(v)
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

function rowFrom(body: TelemetryEvent): string {
  return [
    new Date().toISOString(),
    body.event,
    body.sessionId ?? '',
    body.cold_start === undefined ? '' : String(body.cold_start),
    body.inference_ms ?? '',
    body.download_ms ?? '',
    body.provider ?? '',
    body.angle_mae_vs_baseline ?? '',
    body.extra ? JSON.stringify(body.extra) : '',
  ]
    .map(csvField)
    .join(',')
}

export async function POST(request: NextRequest) {
  let body: TelemetryEvent
  try {
    body = (await request.json()) as TelemetryEvent
    if (!body || typeof body.event !== 'string') {
      return new NextResponse(null, { status: 204 })
    }
  } catch {
    return new NextResponse(null, { status: 204 })
  }

  const row = rowFrom(body)
  const key = todayKey()

  // Try to append to today's blob. Vercel Blob doesn't support true
  // append, so we read-modify-write. With low traffic (< few req/s)
  // this is fine; if traffic grows we'd swap to a queue or a real DB.
  try {
    const blobToken = process.env.BLOB_READ_WRITE_TOKEN
    if (!blobToken) {
      // Fallback: log-only. Telemetry still captured in Vercel logs.
      console.log(`[telemetry] ${row}`)
      return new NextResponse(null, { status: 204 })
    }

    let existing = ''
    try {
      const items = await list({ prefix: key, token: blobToken })
      const match = items.blobs.find((b) => b.pathname === key)
      if (match) {
        const res = await fetch(match.url, { cache: 'no-store' })
        if (res.ok) existing = await res.text()
      }
    } catch {
      /* read failure is non-fatal — we'll write a fresh file */
    }

    const next = existing
      ? existing.endsWith('\n')
        ? existing + row + '\n'
        : existing + '\n' + row + '\n'
      : HEADERS.join(',') + '\n' + row + '\n'

    await put(key, next, {
      access: 'public',
      token: blobToken,
      contentType: 'text/csv',
      addRandomSuffix: false,
      allowOverwrite: true,
    })
  } catch (err) {
    // Telemetry must never break a user request. Log the error and 204.
    console.error('[telemetry] write failed:', err)
  }

  return new NextResponse(null, { status: 204 })
}
