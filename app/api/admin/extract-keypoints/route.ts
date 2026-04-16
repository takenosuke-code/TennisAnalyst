import { NextRequest, NextResponse } from 'next/server'
import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { supabase } from '@/lib/supabase'

const EXTRACT_TIMEOUT_MS = 120_000

export async function POST(request: NextRequest) {
  let body
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { swingId } = body
  if (!swingId || typeof swingId !== 'string') {
    return NextResponse.json(
      { error: 'Missing or invalid swingId' },
      { status: 400 }
    )
  }

  // 1. Fetch the swing from Supabase
  const { data: swing, error: fetchError } = await supabase
    .from('pro_swings')
    .select('id, video_url')
    .eq('id', swingId)
    .single()

  if (fetchError || !swing) {
    return NextResponse.json(
      { error: `Swing not found: ${fetchError?.message ?? 'no data'}` },
      { status: 404 }
    )
  }

  const videoUrl: string = swing.video_url
  if (!videoUrl) {
    return NextResponse.json(
      { error: 'Swing has no video_url' },
      { status: 400 }
    )
  }

  // 2. Resolve the video path on disk
  const videoPath = path.join(process.cwd(), 'public', videoUrl)
  if (!existsSync(videoPath)) {
    return NextResponse.json(
      { error: `Video file not found on disk: ${videoUrl}` },
      { status: 404 }
    )
  }

  // 3. Shell out to the Python extraction script
  const scriptPath = path.join(
    process.cwd(),
    'railway-service',
    'extract_clip_keypoints.py'
  )
  if (!existsSync(scriptPath)) {
    return NextResponse.json(
      { error: 'Extraction script not found at railway-service/extract_clip_keypoints.py' },
      { status: 500 }
    )
  }

  let result: {
    fps_sampled: number
    frame_count: number
    frames: unknown[]
    duration_ms: number
  }

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        'python3',
        [scriptPath, videoPath],
        { timeout: EXTRACT_TIMEOUT_MS, maxBuffer: 50 * 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            const msg =
              error.code === 'ENOENT'
                ? 'python3 is not installed or not in PATH'
                : error.killed
                  ? `Extraction timed out after ${EXTRACT_TIMEOUT_MS / 1000}s`
                  : `Extraction failed: ${stderr || error.message}`
            reject(new Error(msg))
            return
          }
          resolve(stdout)
        }
      )
    })

    const parsed = JSON.parse(stdout)
    if (parsed.error) {
      return NextResponse.json({ error: parsed.error }, { status: 500 })
    }
    result = parsed
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown extraction error'
    console.error('[/api/admin/extract-keypoints] Extraction error:', message)
    return NextResponse.json({ error: message }, { status: 500 })
  }

  // 4. Update the Supabase pro_swings row
  const { error: updateError } = await supabase
    .from('pro_swings')
    .update({
      keypoints_json: result,
      frame_count: result.frame_count,
      duration_ms: result.duration_ms,
      fps: result.fps_sampled,
    })
    .eq('id', swingId)

  if (updateError) {
    console.error('[/api/admin/extract-keypoints] Supabase update error:', updateError.message)
    return NextResponse.json(
      { error: `Failed to update swing: ${updateError.message}` },
      { status: 500 }
    )
  }

  // 5. Return success
  return NextResponse.json({
    success: true,
    frameCount: result.frame_count,
    durationMs: result.duration_ms,
  })
}
