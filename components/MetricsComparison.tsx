'use client'

import { useMemo } from 'react'
import type { PoseFrame, JointAngles } from '@/lib/supabase'
import type { PhaseTimestamp, SwingPhase } from '@/lib/syncAlignment'
import { detectSwingPhases } from '@/lib/syncAlignment'
import {
  getShotTypeConfig,
  scoreAngleDeviation,
  scoreAngleVsIdeal,
  type DeviationLevel,
  type MistakeCheck,
} from '@/lib/shotTypeConfig'

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MetricsComparisonProps {
  userFrames: PoseFrame[]
  proFrames: PoseFrame[]
  shotType?: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PHASE_LABELS: Record<SwingPhase, string> = {
  preparation: 'Prep',
  backswing: 'Load',
  forward_swing: 'Swing',
  contact: 'Contact',
  follow_through: 'Finish',
}

const DEVIATION_STYLES: Record<DeviationLevel, { bg: string; text: string; border: string }> = {
  good: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/30' },
  moderate: { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30' },
  poor: { bg: 'bg-red-500/15', text: 'text-red-300', border: 'border-red-500/30' },
  unknown: { bg: 'bg-white/5', text: 'text-white/40', border: 'border-white/10' },
}

/** Find the frame closest to a given phase timestamp. */
function getFrameAtPhase(
  frames: PoseFrame[],
  phases: PhaseTimestamp[],
  target: SwingPhase
): PoseFrame | null {
  const phase = phases.find((p) => p.phase === target)
  if (!phase) return null
  let closest = frames[0] ?? null
  let minDiff = Infinity
  for (const f of frames) {
    const diff = Math.abs(f.timestamp_ms - phase.timestampMs)
    if (diff < minDiff) {
      minDiff = diff
      closest = f
    }
  }
  return closest
}

/** Format angle to a short string. */
function fmtAngle(v: number | undefined): string {
  if (v == null) return '--'
  return `${Math.round(v)}°`
}

// ---------------------------------------------------------------------------
// Kinetic Chain Analysis
// ---------------------------------------------------------------------------

interface ChainEvent {
  label: string
  key: keyof JointAngles
  peakTimeMs: number | null
}

/**
 * For each joint in the kinetic chain, find the frame where that angle
 * changes most rapidly (peak angular velocity). Returns events in the
 * order they actually fired.
 */
function analyzeKineticChain(
  frames: PoseFrame[],
  chainOrder: (keyof JointAngles)[]
): ChainEvent[] {
  if (frames.length < 3) return []

  const events: ChainEvent[] = chainOrder.map((key) => {
    let peakVel = -Infinity
    let peakTimeMs: number | null = null

    for (let i = 1; i < frames.length; i++) {
      const prev = frames[i - 1].joint_angles?.[key]
      const curr = frames[i].joint_angles?.[key]
      if (prev == null || curr == null) continue
      const dt = (frames[i].timestamp_ms - frames[i - 1].timestamp_ms) / 1000
      if (dt <= 0) continue
      const vel = Math.abs(curr - prev) / dt
      if (vel > peakVel) {
        peakVel = vel
        peakTimeMs = (frames[i].timestamp_ms + frames[i - 1].timestamp_ms) / 2
      }
    }

    const friendlyLabels: Partial<Record<keyof JointAngles, string>> = {
      hip_rotation: 'Hips',
      trunk_rotation: 'Trunk',
      right_shoulder: 'R Shoulder',
      left_shoulder: 'L Shoulder',
      right_elbow: 'R Elbow',
      left_elbow: 'L Elbow',
      right_knee: 'R Knee',
    }

    return {
      label: friendlyLabels[key] ?? key,
      key,
      peakTimeMs,
    }
  })

  // Sort by actual firing time
  const withTime = events.filter((e) => e.peakTimeMs != null)
  withTime.sort((a, b) => a.peakTimeMs! - b.peakTimeMs!)
  return withTime
}

/**
 * Check if the kinetic chain fires in the correct order.
 * Returns true if the actual order matches the expected order.
 */
function isChainCorrect(
  actual: ChainEvent[],
  expectedOrder: (keyof JointAngles)[]
): boolean {
  if (actual.length < 2) return true // not enough data to judge
  const actualKeys = actual.map((e) => e.key)
  // Check pairwise ordering: each pair in expected order should appear in same order in actual
  for (let i = 0; i < expectedOrder.length - 1; i++) {
    const idxA = actualKeys.indexOf(expectedOrder[i])
    const idxB = actualKeys.indexOf(expectedOrder[i + 1])
    if (idxA === -1 || idxB === -1) continue
    if (idxA > idxB) return false
  }
  return true
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MetricsComparison({
  userFrames,
  proFrames,
  shotType,
}: MetricsComparisonProps) {
  const config = useMemo(() => getShotTypeConfig(shotType), [shotType])

  const userPhases = useMemo(() => detectSwingPhases(userFrames), [userFrames])
  const proPhases = useMemo(() => detectSwingPhases(proFrames), [proFrames])

  // Build per-spec metric rows
  const metrics = useMemo(() => {
    return config.keyAngleSpecs.map((spec) => {
      const userFrame = getFrameAtPhase(userFrames, userPhases, spec.phase)
      const proFrame = getFrameAtPhase(proFrames, proPhases, spec.phase)
      const userAngle = userFrame?.joint_angles?.[spec.angleKey]
      const proAngle = proFrame?.joint_angles?.[spec.angleKey]
      const vsProScore = scoreAngleDeviation(userAngle, proAngle)
      const vsIdealScore = scoreAngleVsIdeal(userAngle, spec.idealRange)
      // Use the worse of the two scores for the overall color
      const level: DeviationLevel =
        vsProScore.level === 'unknown' || vsIdealScore.level === 'unknown'
          ? 'unknown'
          : vsProScore.level === 'poor' || vsIdealScore.level === 'poor'
            ? 'poor'
            : vsProScore.level === 'moderate' || vsIdealScore.level === 'moderate'
              ? 'moderate'
              : 'good'
      return { spec, userAngle, proAngle, level, vsProScore, vsIdealScore }
    })
  }, [config, userFrames, proFrames, userPhases, proPhases])

  // Detect mistakes
  const detectedMistakes = useMemo(() => {
    const results: { check: MistakeCheck; detected: boolean }[] = []
    for (const check of config.mistakeChecks) {
      const frame = getFrameAtPhase(userFrames, userPhases, check.phase)
      if (!frame) {
        results.push({ check, detected: false })
        continue
      }
      results.push({ check, detected: check.detect(frame.joint_angles) })
    }
    return results.filter((r) => r.detected)
  }, [config, userFrames, userPhases])

  // Kinetic chain analysis
  const userChain = useMemo(
    () => analyzeKineticChain(userFrames, config.kineticChainOrder),
    [userFrames, config]
  )
  const chainCorrect = useMemo(
    () => isChainCorrect(userChain, config.kineticChainOrder),
    [userChain, config]
  )

  // Find the biggest area for improvement
  const worstMetric = useMemo(() => {
    const poor = metrics.filter((m) => m.level === 'poor')
    if (poor.length > 0) return poor[0]
    const moderate = metrics.filter((m) => m.level === 'moderate')
    return moderate[0] ?? null
  }, [metrics])

  if (userFrames.length < 5 || proFrames.length < 5) {
    return null
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-white/5 border-b border-white/10 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          Key Metrics - {config.label}
        </h3>
        {worstMetric && (
          <span className={`text-xs px-2 py-0.5 rounded-full border ${DEVIATION_STYLES[worstMetric.level].bg} ${DEVIATION_STYLES[worstMetric.level].text} ${DEVIATION_STYLES[worstMetric.level].border}`}>
            Focus: {worstMetric.spec.label}
          </span>
        )}
      </div>

      {/* Angle comparison table */}
      <div className="px-4 py-3 space-y-1">
        {/* Column headers */}
        <div className="grid grid-cols-[1fr_60px_60px_60px_60px] gap-2 text-xs text-white/40 pb-1">
          <span>Metric</span>
          <span className="text-center">Phase</span>
          <span className="text-center">You</span>
          <span className="text-center">Pro</span>
          <span className="text-center">Ideal</span>
        </div>

        {metrics.map(({ spec, userAngle, proAngle, level }) => {
          const style = DEVIATION_STYLES[level]
          return (
            <div
              key={`${spec.angleKey}-${spec.phase}`}
              className={`grid grid-cols-[1fr_60px_60px_60px_60px] gap-2 items-center rounded-lg px-2 py-1.5 text-sm border ${style.bg} ${style.border}`}
            >
              <span className={`font-medium ${style.text}`}>{spec.label}</span>
              <span className="text-center text-white/50 text-xs">
                {PHASE_LABELS[spec.phase]}
              </span>
              <span className={`text-center font-mono ${style.text}`}>
                {fmtAngle(userAngle)}
              </span>
              <span className="text-center font-mono text-white/60">
                {fmtAngle(proAngle)}
              </span>
              <span className="text-center text-white/40 text-xs">
                {spec.idealRange[0]}-{spec.idealRange[1]}°
              </span>
            </div>
          )
        })}
      </div>

      {/* Kinetic Chain */}
      <div className="px-4 py-3 border-t border-white/10">
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-xs font-semibold text-white/70 uppercase tracking-wider">
            Kinetic Chain
          </h4>
          <span
            className={`text-xs px-2 py-0.5 rounded-full border ${
              chainCorrect
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                : 'bg-red-500/15 text-red-300 border-red-500/30'
            }`}
          >
            {chainCorrect ? 'Correct sequence' : 'Out of order'}
          </span>
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {userChain.map((event, i) => (
            <div key={event.key} className="flex items-center gap-1">
              {i > 0 && (
                <svg viewBox="0 0 12 12" className="w-3 h-3 text-white/30 flex-shrink-0">
                  <path d="M2 6h8M7 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              <span
                className={`text-xs px-2 py-1 rounded border ${
                  (() => {
                    // Check if this event is in the correct position
                    const expectedIdx = config.kineticChainOrder.indexOf(event.key)
                    const prevEvents = userChain.slice(0, i)
                    const isOrdered = prevEvents.every((prev) => {
                      const prevExpected = config.kineticChainOrder.indexOf(prev.key)
                      return prevExpected <= expectedIdx
                    })
                    return isOrdered
                      ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
                      : 'bg-red-500/10 text-red-300 border-red-500/20'
                  })()
                }`}
              >
                {event.label}
              </span>
            </div>
          ))}
          {userChain.length === 0 && (
            <span className="text-xs text-white/30">Not enough data to analyze</span>
          )}
        </div>
        {!chainCorrect && (
          <p className="text-xs text-white/50 mt-2">
            Your hips should fire first, then trunk, then shoulder and arm. This is the
            &quot;whip&quot; that generates effortless power.
          </p>
        )}
      </div>

      {/* Detected Mistakes */}
      {detectedMistakes.length > 0 && (
        <div className="px-4 py-3 border-t border-white/10">
          <h4 className="text-xs font-semibold text-white/70 uppercase tracking-wider mb-2">
            Areas to Improve
          </h4>
          <div className="space-y-2">
            {detectedMistakes.map(({ check }) => (
              <div
                key={check.id}
                className="rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2"
              >
                <p className="text-sm font-medium text-amber-300">
                  {check.label}
                </p>
                <p className="text-xs text-white/50 mt-0.5">{check.tip}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top coaching cue */}
      {worstMetric && (
        <div className="px-4 py-3 border-t border-white/10 bg-white/[0.02]">
          <p className="text-xs text-white/40 uppercase tracking-wider mb-1">Top Priority</p>
          <p className="text-sm text-white/80">{worstMetric.spec.coachingCue}</p>
        </div>
      )}
    </div>
  )
}
