/*
 * Client-side helper for the Phase 0.5 telemetry sink. Fire-and-forget
 * via fetch with `keepalive: true` so the request survives navigations
 * and tab closes (crucial for "user upload finished" / "user bounced"
 * events that fire right before the user leaves).
 *
 * Contract: this function MUST NEVER throw and MUST NEVER block the
 * caller. Every code path swallows errors. If telemetry is going to
 * destabilize the app, we'd rather lose the data point.
 */

export type TelemetryEvent = {
  event: string
  sessionId?: string
  cold_start?: boolean
  inference_ms?: number
  download_ms?: number
  provider?: string
  angle_mae_vs_baseline?: number
  extra?: Record<string, unknown>
}

export function track(event: string, props: Omit<TelemetryEvent, 'event'> = {}): void {
  if (typeof window === 'undefined') return
  try {
    void fetch('/api/telemetry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, ...props }),
      keepalive: true,
    }).catch(() => {
      /* non-fatal */
    })
  } catch {
    /* non-fatal */
  }
}
