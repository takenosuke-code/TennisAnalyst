import { NextRequest, NextResponse } from 'next/server'
import { requireAdminAuth } from '@/lib/adminAuth'

// Lightweight endpoint for the client to verify an admin token. Returns 200
// if the token is valid, 401 otherwise. Used by the password prompt on
// /admin/tag-clips before caching the token in localStorage.
export async function GET(request: NextRequest) {
  const guard = requireAdminAuth(request)
  if (guard) return guard
  return NextResponse.json({ ok: true })
}
