'use client'

import { useMemo } from 'react'
import { useProLibraryStore } from '@/store'
import { detectSwingPhases } from '@/lib/syncAlignment'
import type { SwingPhase, PhaseTimestamp } from '@/lib/syncAlignment'
import { getShotTypeConfig, scoreAngleVsIdeal } from '@/lib/shotTypeConfig'
import type { DeviationLevel } from '@/lib/shotTypeConfig'
import { analyzeKineticChain } from '@/lib/kineticChain'
import type { PoseFrame } from '@/lib/supabase'

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
  unknown: { bg: 'bg-zinc-500/15', text: 'text-zinc-400', border: 'border-zinc-500/30' },
}

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

function fmtAngle(v: number | undefined): string {
  if (v == null) return '--'
  return `${Math.round(v)}°`
}

export default function ProSwingAnalysis() {
  const selectedSwing = useProLibraryStore((s) => s.selectedSwing)

  const frames = useMemo(
    () => selectedSwing?.keypoints_json?.frames ?? [],
    [selectedSwing]
  )
  const shotType = selectedSwing?.shot_type ?? null

  const config = useMemo(() => getShotTypeConfig(shotType), [shotType])
  const phases = useMemo(() => detectSwingPhases(frames), [frames])

  // Key angles table data
  const metrics = useMemo(() => {
    return config.keyAngleSpecs.map((spec) => {
      const frame = getFrameAtPhase(frames, phases, spec.phase)
      const proAngle = frame?.joint_angles?.[spec.angleKey]
      const vsIdeal = scoreAngleVsIdeal(proAngle, spec.idealRange)
      return { spec, proAngle, level: vsIdeal.level }
    })
  }, [config, frames, phases])

  // Kinetic chain analysis
  const chainResult = useMemo(() => {
    const chainDef = config.kineticChainOrder.map((angleKey) => {
      const friendlyLabels: Record<string, string> = {
        hip_rotation: 'Hips',
        trunk_rotation: 'Trunk',
        right_shoulder: 'R Shoulder',
        left_shoulder: 'L Shoulder',
        right_elbow: 'R Elbow',
        left_elbow: 'L Elbow',
        right_knee: 'R Knee',
      }
      return { name: friendlyLabels[angleKey] ?? angleKey, angleKey }
    })
    return analyzeKineticChain(frames, chainDef)
  }, [frames, config])

  // Technique highlights (specs where pro is in ideal range)
  const highlights = useMemo(
    () => metrics.filter((m) => m.level === 'good' && m.proAngle != null),
    [metrics]
  )

  if (frames.length < 5) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6 text-center">
        <p className="text-white/30 text-sm">
          Not enough frame data to analyze this swing.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-white/5 border-b border-white/10">
        <h3 className="text-sm font-semibold text-white">What Makes This Swing Great</h3>
        <p className="text-xs text-white/40 mt-0.5">{config.label} technique breakdown</p>
      </div>

      {/* Key angles table */}
      <div className="px-4 py-3 space-y-1">
        <div className="grid grid-cols-[1fr_50px_50px_70px] gap-2 text-xs text-white/40 pb-1">
          <span>Metric</span>
          <span className="text-center">Phase</span>
          <span className="text-center">Pro</span>
          <span className="text-center">Ideal</span>
        </div>

        {metrics.map(({ spec, proAngle, level }) => {
          const style = DEVIATION_STYLES[level]
          return (
            <div
              key={`${spec.angleKey}-${spec.phase}`}
              className={`grid grid-cols-[1fr_50px_50px_70px] gap-2 items-center rounded-lg px-2 py-1.5 text-sm border ${style.bg} ${style.border}`}
            >
              <span className={`font-medium text-xs ${style.text}`}>{spec.label}</span>
              <span className="text-center text-white/50 text-xs">
                {PHASE_LABELS[spec.phase]}
              </span>
              <span className={`text-center font-mono text-xs ${style.text}`}>
                {fmtAngle(proAngle)}
              </span>
              <span className="text-center text-white/40 text-xs">
                {spec.idealRange[0]}-{spec.idealRange[1]}°
              </span>
            </div>
          )
        })}
      </div>

      {/* Kinetic chain */}
      <div className="px-4 py-3 border-t border-white/10">
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-xs font-semibold text-white/70 uppercase tracking-wider">
            Kinetic Chain
          </h4>
          <span
            className={`text-xs px-2 py-0.5 rounded-full border ${
              chainResult.isSequenceCorrect
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                : 'bg-red-500/15 text-red-300 border-red-500/30'
            }`}
          >
            {chainResult.isSequenceCorrect ? 'Correct sequence' : 'Out of order'}
          </span>
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {chainResult.segments
            .filter((s) => s.peakFrame >= 0)
            .sort((a, b) => a.peakTimestampMs - b.peakTimestampMs)
            .map((seg, i) => {
              const isOutOfOrder = chainResult.outOfOrderSegments.includes(seg.name)
              return (
                <div key={seg.angleKey} className="flex items-center gap-1">
                  {i > 0 && (
                    <svg viewBox="0 0 12 12" className="w-3 h-3 text-white/30 flex-shrink-0">
                      <path
                        d="M2 6h8M7 3l3 3-3 3"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                  <span
                    className={`text-xs px-2 py-1 rounded border ${
                      isOutOfOrder
                        ? 'bg-red-500/10 text-red-300 border-red-500/20'
                        : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
                    }`}
                  >
                    {seg.name}
                  </span>
                </div>
              )
            })}
          {chainResult.segments.filter((s) => s.peakFrame >= 0).length === 0 && (
            <span className="text-xs text-white/30">Not enough data to analyze</span>
          )}
        </div>
      </div>

      {/* Technique highlights */}
      {highlights.length > 0 && (
        <div className="px-4 py-3 border-t border-white/10">
          <h4 className="text-xs font-semibold text-white/70 uppercase tracking-wider mb-2">
            Technique Highlights
          </h4>
          <div className="space-y-2">
            {highlights.map(({ spec }) => (
              <div
                key={`${spec.angleKey}-${spec.phase}-highlight`}
                className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2"
              >
                <p className="text-xs text-emerald-300 font-medium">{spec.label}</p>
                <p className="text-xs text-white/50 mt-0.5">{spec.coachingCue}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
