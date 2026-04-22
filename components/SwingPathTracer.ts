import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

const TRAIL_LENGTH = 40 // number of frames to keep in trail buffer

// Velocity ceiling (px/sec) used to map segment speed -> hue. A 1080p tennis
// swing typically peaks around 1500 px/s at the wrist; tune empirically.
const MAX_TRAIL_VELOCITY = 1500

// Client-side fallback for when the server didn't emit a racket_head or
// emitted it below the render threshold. Extends 40% past the wrist along
// the elbow->wrist vector, which lands roughly at the middle of a standard
// racket at full extension for a side-view frame. Mirrors the server-side
// fallback in railway-service/extract_clip_keypoints.py — duplicated here
// so the trail renders regardless of Railway deploy state.
const RACKET_FALLBACK_EXTENSION = 0.4
const RACKET_FALLBACK_VIS_GATE = 0.5

function racketCenterFromPose(frame: PoseFrame):
  | { x: number; y: number }
  | null {
  const find = (id: number) => frame.landmarks.find((l) => l.id === id)
  const lWrist = find(LANDMARK_INDICES.LEFT_WRIST)
  const rWrist = find(LANDMARK_INDICES.RIGHT_WRIST)
  const lElbow = find(LANDMARK_INDICES.LEFT_ELBOW)
  const rElbow = find(LANDMARK_INDICES.RIGHT_ELBOW)

  // Pick the higher-visibility wrist as the racket-holding side.
  // Strict > matches the wrist-trail convention in push() above.
  const lVis = lWrist?.visibility ?? 0
  const rVis = rWrist?.visibility ?? 0
  if (lVis <= RACKET_FALLBACK_VIS_GATE && rVis <= RACKET_FALLBACK_VIS_GATE) return null

  const wrist = rVis >= lVis ? rWrist : lWrist
  const elbow = rVis >= lVis ? rElbow : lElbow
  if (!wrist) return null

  if (elbow && (elbow.visibility ?? 0) > RACKET_FALLBACK_VIS_GATE) {
    const dx = wrist.x - elbow.x
    const dy = wrist.y - elbow.y
    return {
      x: Math.max(0, Math.min(1, wrist.x + dx * RACKET_FALLBACK_EXTENSION)),
      y: Math.max(0, Math.min(1, wrist.y + dy * RACKET_FALLBACK_EXTENSION)),
    }
  }
  // Elbow too weak: fall back to raw wrist position. Still better than
  // an empty trail.
  return { x: wrist.x, y: wrist.y }
}

export type TrailPoint = {
  x: number
  y: number
  timestamp: number
}

export type SwingPathRenderOptions = {
  rightColor?: string
  leftColor?: string
  racketColor?: string
  showRacketTrail?: boolean
  showWristTrails?: boolean
  // When set, only the named side's wrist trail renders. Mirrors the
  // off-hand filter in PoseRenderer so forehand / one-handed backhand /
  // serve views show a single swing-arm trail instead of both.
  dominantHand?: 'left' | 'right' | null
}

// Maintains a rolling buffer of wrist (and racket-head) positions for swing
// path tracing.
export class SwingPathTracer {
  private rightWristTrail: TrailPoint[] = []
  private leftWristTrail: TrailPoint[] = []
  private racketTrail: TrailPoint[] = []

  push(frame: PoseFrame, canvasWidth: number, canvasHeight: number): void {
    const rWrist = frame.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.RIGHT_WRIST && l.visibility > 0.5
    )
    const lWrist = frame.landmarks.find(
      (l) => l.id === LANDMARK_INDICES.LEFT_WRIST && l.visibility > 0.5
    )

    if (rWrist) {
      this.rightWristTrail.push({
        x: rWrist.x * canvasWidth,
        y: rWrist.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.rightWristTrail.length > TRAIL_LENGTH) {
        this.rightWristTrail.shift()
      }
    }

    if (lWrist) {
      this.leftWristTrail.push({
        x: lWrist.x * canvasWidth,
        y: lWrist.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.leftWristTrail.length > TRAIL_LENGTH) {
        this.leftWristTrail.shift()
      }
    }

    // Racket-head: prefer the server's YOLO detection when confident, but
    // fall back to a forearm-extension estimate from the pose so the trail
    // ALWAYS renders when the arm is visible. Server-side fallback also
    // exists (afb0043) but the client copy guarantees rendering even if
    // the server didn't redeploy or stored older all-null racket_head
    // values in keypoints_json.
    const racket = frame.racket_head
    let racketPoint: { x: number; y: number } | null = null
    if (racket && racket.confidence >= 0.3) {
      racketPoint = { x: racket.x, y: racket.y }
    } else {
      racketPoint = racketCenterFromPose(frame)
    }
    if (racketPoint) {
      this.racketTrail.push({
        x: racketPoint.x * canvasWidth,
        y: racketPoint.y * canvasHeight,
        timestamp: frame.timestamp_ms,
      })
      if (this.racketTrail.length > TRAIL_LENGTH) {
        this.racketTrail.shift()
      }
    }
  }

  render(
    ctx: CanvasRenderingContext2D,
    rightColorOrOptions: string | SwingPathRenderOptions = '#10b981',
    leftColor = '#60a5fa'
  ): void {
    // Back-compat: render(ctx, rightColor, leftColor) still works.
    let opts: Required<SwingPathRenderOptions>
    if (typeof rightColorOrOptions === 'string') {
      opts = {
        rightColor: rightColorOrOptions,
        leftColor,
        racketColor: '#fbbf24',
        showRacketTrail: true,
        showWristTrails: true,
        dominantHand: null,
      }
    } else {
      opts = {
        rightColor: rightColorOrOptions.rightColor ?? '#10b981',
        leftColor: rightColorOrOptions.leftColor ?? '#60a5fa',
        racketColor: rightColorOrOptions.racketColor ?? '#fbbf24',
        showRacketTrail: rightColorOrOptions.showRacketTrail ?? true,
        showWristTrails: rightColorOrOptions.showWristTrails ?? true,
        dominantHand: rightColorOrOptions.dominantHand ?? null,
      }
    }

    if (opts.showWristTrails) {
      // dominantHand === 'right' hides left trail; 'left' hides right; null
      // shows both (back-compat).
      if (opts.dominantHand !== 'left') {
        this.drawTrail(ctx, this.rightWristTrail, opts.rightColor, {
          tipRadius: 5,
        })
      }
      if (opts.dominantHand !== 'right') {
        this.drawTrail(ctx, this.leftWristTrail, opts.leftColor, {
          tipRadius: 5,
        })
      }
    }
    if (opts.showRacketTrail) {
      this.drawTrail(ctx, this.racketTrail, opts.racketColor, {
        tipRadius: 8,
        baseLineWidth: 3,
      })
    }
  }

  private drawTrail(
    ctx: CanvasRenderingContext2D,
    trail: TrailPoint[],
    color: string,
    {
      tipRadius = 5,
      baseLineWidth = 2,
    }: { tipRadius?: number; baseLineWidth?: number } = {}
  ): void {
    if (trail.length < 2) return

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    // Draw segments with tapering alpha/width and velocity-encoded hue.
    for (let i = 1; i < trail.length; i++) {
      const progress = i / trail.length // 0 (oldest) → 1 (newest)
      const alpha = progress * 0.85
      const lineWidth = baseLineWidth + progress * 4

      // px/sec between this and previous point; graceful on zero dt.
      const prevPt = trail[i - 1]
      const currPt = trail[i]
      const dx = currPt.x - prevPt.x
      const dy = currPt.y - prevPt.y
      const pxDist = Math.sqrt(dx * dx + dy * dy)
      const dtSec = (currPt.timestamp - prevPt.timestamp) / 1000
      const velocity = dtSec > 0 ? pxDist / dtSec : 0
      const hue = Math.max(
        0,
        Math.min(240, 240 - (velocity / MAX_TRAIL_VELOCITY) * 240)
      )
      const saturation = 90
      const lightness = 55

      ctx.globalAlpha = alpha
      ctx.strokeStyle = `hsl(${hue.toFixed(1)}, ${saturation}%, ${lightness}%)`
      ctx.lineWidth = lineWidth

      ctx.beginPath()

      if (i === 1) {
        ctx.moveTo(trail[0].x, trail[0].y)
        ctx.lineTo(trail[1].x, trail[1].y)
      } else {
        // Smooth quadratic curve: start at midpoint of previous segment so
        // each segment begins exactly where the last one ended (no overlap)
        const prev = trail[i - 1]
        const curr = trail[i]
        const startMidX = (trail[i - 2].x + prev.x) / 2
        const startMidY = (trail[i - 2].y + prev.y) / 2
        const endMidX = (prev.x + curr.x) / 2
        const endMidY = (prev.y + curr.y) / 2
        ctx.moveTo(startMidX, startMidY)
        ctx.quadraticCurveTo(prev.x, prev.y, endMidX, endMidY)
      }

      ctx.stroke()
    }

    // Draw a bright dot at the tip (current position) in the trail's base color
    // so each trail stays visually distinguishable when velocities are similar.
    const tip = trail[trail.length - 1]
    ctx.globalAlpha = 1
    ctx.beginPath()
    ctx.arc(tip.x, tip.y, tipRadius, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.strokeStyle = 'white'
    ctx.lineWidth = 1.5
    ctx.stroke()

    ctx.restore()
  }

  reset(): void {
    this.rightWristTrail = []
    this.leftWristTrail = []
    this.racketTrail = []
  }

  // Build full trail from all frames (for reviewing a complete swing)
  buildFromFrames(
    frames: PoseFrame[],
    canvasWidth: number,
    canvasHeight: number
  ): void {
    this.reset()
    for (const frame of frames) {
      this.push(frame, canvasWidth, canvasHeight)
    }
  }
}
