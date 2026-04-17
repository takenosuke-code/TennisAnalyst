import { NextRequest, NextResponse } from 'next/server'
import { requireAdminAuth } from '@/lib/adminAuth'
import { existsSync, statSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

export const runtime = 'nodejs'

const PREVIEW_DIR = join(os.tmpdir(), 'tennisiq-clip-previews')

// Streams a previously-generated YouTube preview file to the caller. Replaces
// the old pattern of serving previews as static files under public/, which
// only had UUID obscurity between the file and the open internet.
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const guard = requireAdminAuth(request)
  if (guard) return guard

  const { id } = await params

  // Reject anything that isn't a straightforward UUID so no one can path-
  // traverse out of PREVIEW_DIR via `..` or slashes.
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    return NextResponse.json({ error: 'Invalid preview id' }, { status: 400 })
  }

  const filePath = join(PREVIEW_DIR, `${id}.mp4`)
  if (!existsSync(filePath)) {
    return NextResponse.json({ error: 'Preview not found or expired' }, { status: 404 })
  }

  try {
    const buf = readFileSync(filePath)
    const st = statSync(filePath)
    return new NextResponse(new Uint8Array(buf), {
      status: 200,
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Length': String(st.size),
        'Cache-Control': 'private, no-store',
      },
    })
  } catch (err) {
    console.error('[preview-clip GET] read error:', err)
    return NextResponse.json({ error: 'Failed to read preview' }, { status: 500 })
  }
}
