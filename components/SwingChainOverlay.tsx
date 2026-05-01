'use client'

/**
 * SwingChainOverlay
 *
 * A passive, animated SVG annotation that visualises the kinetic chain
 * (hips -> trunk -> shoulders -> wrist) as a sequence of pulses. There
 * is no video-time synchronisation here — the animation runs on a
 * looping CSS keyframe, infinite. Real currentTime-driven animation is
 * a future iteration.
 *
 * Design notes
 * ------------
 * The component renders an SVG that scales to its parent (responsive
 * via 100% width / aspect-ratio: 5/3). Four nodes sit on a vertical
 * spine — hip, trunk, shoulder, wrist — and the animation cycles the
 * pulse upward. Colors come from the cream/ink/clay/hard-court palette
 * to stay consistent with the redesign; no new tokens introduced.
 *
 * A subtle hint of why this exists: SwingVision shows joint dots; we
 * show the *order* the chain fires in, which is the actual coaching
 * insight. v1 ships the visual idiom; v2 will sync to peak-velocity
 * timestamps.
 */

import type { CSSProperties } from 'react'

interface SwingChainOverlayProps {
  /** Optional caption rendered below the diagram. */
  caption?: string
  className?: string
}

// Per-stop delays in seconds. Total cycle = 2.4s with 0.6s gap before
// the loop re-fires from the hips. Tuned by feel — slow enough that
// the eye can track the wave, fast enough that nobody mistakes it
// for a static graphic.
const STOPS = [
  { id: 'hip', y: 78, label: 'Hips', color: 'var(--color-clay)', delay: 0 },
  { id: 'trunk', y: 56, label: 'Trunk', color: 'var(--color-clay-soft)', delay: 0.45 },
  { id: 'shoulder', y: 36, label: 'Shoulders', color: 'var(--color-hard-court)', delay: 0.9 },
  { id: 'wrist', y: 18, label: 'Wrist', color: 'var(--color-hard-court-soft)', delay: 1.35 },
] as const

export default function SwingChainOverlay({ caption, className = '' }: SwingChainOverlayProps) {
  return (
    <div className={`bg-ink text-cream border border-ink/40 ${className}`}>
      <div className="px-5 py-3 border-b border-cream/10 flex items-baseline justify-between gap-3">
        <h3 className="font-display font-bold text-sm tracking-wide">Kinetic chain</h3>
        <span className="text-[11px] uppercase tracking-[0.18em] text-cream/50">
          Hips → Trunk → Shoulders → Wrist
        </span>
      </div>
      <div className="relative px-5 py-6">
        <style>{KEYFRAMES}</style>
        <svg
          viewBox="0 0 60 100"
          preserveAspectRatio="xMidYMid meet"
          className="w-full max-w-[260px] mx-auto block"
          aria-hidden="true"
        >
          {/* Spine */}
          <line
            x1={30}
            y1={STOPS[STOPS.length - 1].y}
            x2={30}
            y2={STOPS[0].y}
            stroke="var(--color-cream)"
            strokeOpacity={0.25}
            strokeWidth={1}
            strokeDasharray="2 2"
          />

          {STOPS.map((stop) => {
            const styleVars: CSSProperties & Record<string, string> = {
              animationDelay: `${stop.delay}s`,
              ['--ring-color']: stop.color,
            }
            return (
              <g key={stop.id}>
                {/* Pulse ring — animates outward from the node. */}
                <circle
                  cx={30}
                  cy={stop.y}
                  r={3.5}
                  fill="none"
                  stroke={stop.color}
                  strokeWidth={1.2}
                  className="chain-pulse"
                  style={styleVars}
                />
                {/* Solid node. */}
                <circle
                  cx={30}
                  cy={stop.y}
                  r={2.6}
                  fill={stop.color}
                  className="chain-dot"
                  style={styleVars}
                />
                {/* Label */}
                <text
                  x={38}
                  y={stop.y + 1.2}
                  fontSize="3.6"
                  fill="var(--color-cream)"
                  fillOpacity={0.7}
                  fontFamily="var(--font-sans)"
                >
                  {stop.label}
                </text>
              </g>
            )
          })}
        </svg>
        {caption && (
          <p className="text-xs text-cream/60 leading-relaxed mt-3 max-w-md">
            {caption}
          </p>
        )}
      </div>
    </div>
  )
}

const KEYFRAMES = `
@keyframes chain-pulse {
  0%   { r: 3.5; opacity: 0.8; }
  60%  { r: 9;   opacity: 0; }
  100% { r: 9;   opacity: 0; }
}
@keyframes chain-dot {
  0%   { r: 2.6; }
  20%  { r: 3.6; }
  60%  { r: 2.6; }
  100% { r: 2.6; }
}
.chain-pulse {
  transform-origin: center;
  animation: chain-pulse 2.4s ease-out infinite;
}
.chain-dot {
  animation: chain-dot 2.4s ease-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  .chain-pulse, .chain-dot {
    animation: none;
  }
}
`
