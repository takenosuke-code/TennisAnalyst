import { LANDMARK_INDICES } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

const TRAIL_LENGTH = 40 // number of frames to keep in trail buffer

export type TrailPoint = {
  x: number
  y: number
  timestamp: number
}

// Maintains a rolling buffer of wrist positions for swing path tracing
export class SwingPathTracer {
  private rightWristTrail: TrailPoint[] = []
  private leftWristTrail: TrailPoint[] = []

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
  }

  render(
    ctx: CanvasRenderingContext2D,
    rightColor = '#10b981',
    leftColor = '#60a5fa'
  ): void {
    this.drawTrail(ctx, this.rightWristTrail, rightColor)
    this.drawTrail(ctx, this.leftWristTrail, leftColor)
  }

  private drawTrail(
    ctx: CanvasRenderingContext2D,
    trail: TrailPoint[],
    color: string
  ): void {
    if (trail.length < 2) return

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    // Draw segments with tapering alpha and width
    for (let i = 1; i < trail.length; i++) {
      const progress = i / trail.length // 0 (oldest) → 1 (newest)
      const alpha = progress * 0.85
      const lineWidth = 2 + progress * 4

      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
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

    // Draw a bright dot at the tip (current position)
    const tip = trail[trail.length - 1]
    ctx.globalAlpha = 1
    ctx.beginPath()
    ctx.arc(tip.x, tip.y, 5, 0, Math.PI * 2)
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
