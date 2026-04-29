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

  // Default path: log to stdout. Vercel's log drain captures this and
  // it's race-free (each request gets its own log line). Production
  // observability sits on top of `[telemetry] ` grep, no I/O contention.
  console.log(`[telemetry] ${row}`)

  // Optional durable CSV in Vercel Blob, gated behind both an explicit
  // opt-in flag AND a write token. Disabled by default because Blob
  // doesn't support true append — read-modify-write loses rows under
  // any concurrent traffic. Re-enable only after wrapping with a queue
  // or migrating to a real DB / append-only sink.
  if (process.env.TELEMETRY_BLOB_ENABLED === '1' && process.env.BLOB_READ_WRITE_TOKEN) {
    try {
      const blobToken = process.env.BLOB_READ_WRITE_TOKEN
      const key = todayKey()
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
      console.error('[telemetry] blob write failed:', err)
    }
  }

  return new NextResponse(null, { status: 204 })
}
