import { NextRequest, NextResponse } from 'next/server'

export const ADMIN_TOKEN_HEADER = 'x-clip-admin-token'

/**
 * Server-side guard for admin-only API routes. Call at the very top of a
 * handler. Returns null on success; otherwise returns a NextResponse the
 * caller should return immediately.
 *
 *  - If `CLIP_ADMIN_PASSWORD` env is not set, returns 404 (admin disabled).
 *  - If the header is missing or wrong, returns 401.
 *
 * Uses a constant-time compare so a wrong password can't be brute-forced
 * byte-by-byte from response timings.
 */
export function requireAdminAuth(request: NextRequest): NextResponse | null {
  const configured = process.env.CLIP_ADMIN_PASSWORD
  if (!configured) {
    return NextResponse.json(
      { error: 'Admin features are not enabled on this deployment.' },
      { status: 404 },
    )
  }
  const provided = request.headers.get(ADMIN_TOKEN_HEADER)
  if (!provided || !timingSafeEqualStr(provided, configured)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  return null
}

function timingSafeEqualStr(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  let diff = 0
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i)
  }
  return diff === 0
}
