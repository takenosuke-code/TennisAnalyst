import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { execFileSync } from 'child_process'
import {
  mkdtempSync,
  unlinkSync,
  existsSync,
  mkdirSync,
  renameSync,
} from 'fs'
import { join } from 'path'
import os from 'os'

const VALID_SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const
const VALID_CAMERA_ANGLES = ['side', 'behind', 'front', 'court_level'] as const

type ShotType = (typeof VALID_SHOT_TYPES)[number]
type CameraAngle = (typeof VALID_CAMERA_ANGLES)[number]

const YOUTUBE_URL_PATTERN = /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//

function sanitizeFilename(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
}

function validateBody(body: Record<string, unknown>): string | null {
  const { youtubeUrl, startTime, endTime, proName, shotType, cameraAngle } =
    body

  if (!youtubeUrl || typeof youtubeUrl !== 'string') {
    return 'youtubeUrl is required and must be a string'
  }
  if (!YOUTUBE_URL_PATTERN.test(youtubeUrl)) {
    return 'youtubeUrl must be a valid YouTube URL'
  }

  if (typeof startTime !== 'number' || isNaN(startTime)) {
    return 'startTime is required and must be a number'
  }
  if (typeof endTime !== 'number' || isNaN(endTime)) {
    return 'endTime is required and must be a number'
  }
  if (endTime <= startTime) {
    return 'endTime must be greater than startTime'
  }

  if (!proName || typeof proName !== 'string') {
    return 'proName is required and must be a string'
  }

  if (
    !shotType ||
    typeof shotType !== 'string' ||
    !VALID_SHOT_TYPES.includes(shotType as ShotType)
  ) {
    return `shotType must be one of: ${VALID_SHOT_TYPES.join(', ')}`
  }

  if (
    !cameraAngle ||
    typeof cameraAngle !== 'string' ||
    !VALID_CAMERA_ANGLES.includes(cameraAngle as CameraAngle)
  ) {
    return `cameraAngle must be one of: ${VALID_CAMERA_ANGLES.join(', ')}`
  }

  return null
}

export async function POST(request: NextRequest) {
  let body: Record<string, unknown>
  try {
    body = await request.json()
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON in request body' },
      { status: 400 }
    )
  }

  const validationError = validateBody(body)
  if (validationError) {
    return NextResponse.json({ error: validationError }, { status: 400 })
  }

  const {
    youtubeUrl,
    startTime,
    endTime,
    proName,
    nationality,
    shotType,
    cameraAngle,
  } = body as {
    youtubeUrl: string
    startTime: number
    endTime: number
    proName: string
    nationality?: string
    shotType: ShotType
    cameraAngle: CameraAngle
  }

  const duration = endTime - startTime
  const projectRoot = process.cwd()
  const ffmpegBin = join(projectRoot, 'pro-videos', 'bin', 'ffmpeg')
  const proVideosDir = join(projectRoot, 'public', 'pro-videos')

  // Ensure public/pro-videos/ directory exists
  if (!existsSync(proVideosDir)) {
    mkdirSync(proVideosDir, { recursive: true })
  }

  const tmpDir = mkdtempSync(join(os.tmpdir(), 'tag-clip-'))
  const tmpFullVideo = join(tmpDir, 'full.mp4')
  const sanitizedName = sanitizeFilename(proName)
  const filename = `${sanitizedName}_${shotType}_${cameraAngle}_${Date.now()}.mp4`
  const trimmedPath = join(tmpDir, 'trimmed.mp4')
  const finalPath = join(proVideosDir, filename)

  try {
    // Step 1: Download the video with yt-dlp (using execFileSync to avoid shell injection)
    console.log('[tag-clip] Downloading video from:', youtubeUrl)
    execFileSync(
      'yt-dlp',
      [
        '-f',
        'mp4[height<=720]/best[ext=mp4]/best',
        '--no-playlist',
        '-o',
        tmpFullVideo,
        youtubeUrl,
      ],
      { stdio: 'pipe', timeout: 120_000 }
    )

    // Step 2: Trim with ffmpeg (using execFileSync to avoid shell injection)
    console.log('[tag-clip] Trimming clip:', { startTime, duration })
    execFileSync(
      ffmpegBin,
      [
        '-y',
        '-ss',
        String(startTime),
        '-i',
        tmpFullVideo,
        '-t',
        String(duration),
        '-c:v',
        'libx264',
        '-preset',
        'fast',
        '-crf',
        '23',
        '-c:a',
        'aac',
        '-movflags',
        '+faststart',
        trimmedPath,
      ],
      { stdio: 'pipe', timeout: 60_000 }
    )

    // Step 3: Move trimmed clip to public/pro-videos/
    renameSync(trimmedPath, finalPath)

    // Step 4: Upsert the pro in Supabase
    const { data: existingPros, error: lookupError } = await supabase
      .from('pros')
      .select('id, name')
      .eq('name', proName)
      .limit(1)

    if (lookupError) {
      throw new Error(`Failed to look up pro: ${lookupError.message}`)
    }

    let pro: { id: string; name: string }

    if (existingPros && existingPros.length > 0) {
      pro = existingPros[0]
    } else {
      const { data: newPro, error: insertError } = await supabase
        .from('pros')
        .insert({ name: proName, nationality: nationality ?? null })
        .select('id, name')
        .single()

      if (insertError || !newPro) {
        throw new Error(
          `Failed to create pro: ${insertError?.message ?? 'Unknown error'}`
        )
      }
      pro = newPro
    }

    // Step 5: Insert the pro_swing
    const videoUrl = `/pro-videos/${filename}`
    const durationMs = Math.round(duration * 1000)

    const { data: swing, error: swingError } = await supabase
      .from('pro_swings')
      .insert({
        pro_id: pro.id,
        shot_type: shotType,
        video_url: videoUrl,
        keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
        fps: 30,
        duration_ms: durationMs,
        phase_labels: {},
        metadata: {
          camera_angle: cameraAngle,
          source: 'youtube',
          original_url: youtubeUrl,
          label: `${shotType}_${cameraAngle}`,
        },
      })
      .select('id, shot_type, video_url')
      .single()

    if (swingError || !swing) {
      throw new Error(
        `Failed to create pro_swing: ${swingError?.message ?? 'Unknown error'}`
      )
    }

    console.log('[tag-clip] Successfully created swing:', swing.id)

    return NextResponse.json({
      success: true,
      pro: { id: pro.id, name: pro.name },
      swing: {
        id: swing.id,
        shot_type: swing.shot_type,
        video_url: swing.video_url,
      },
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    console.error('[tag-clip] Processing failed:', message)
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    // Clean up temp files
    try {
      if (existsSync(tmpFullVideo)) unlinkSync(tmpFullVideo)
    } catch {
      // Temp file cleanup is best-effort
    }
    try {
      if (existsSync(trimmedPath)) unlinkSync(trimmedPath)
    } catch {
      // Temp file cleanup is best-effort
    }
    try {
      if (existsSync(tmpDir)) {
        execFileSync('rm', ['-rf', tmpDir], { stdio: 'pipe' })
      }
    } catch {
      // Temp dir cleanup is best-effort
    }
  }
}
