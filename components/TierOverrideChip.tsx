'use client'

import { useState } from 'react'
import type { SkillTier } from '@/lib/profile'
import { SKILL_TIER_LABELS } from '@/lib/profile'

interface Props {
  inferredTier: SkillTier
  currentTier: SkillTier | null
  reasons: string[]
  confidence: number
  onConfirm: (tier: SkillTier) => Promise<void> | void
}

const PICK_ORDER: SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']

/*
 * Phase 1.5 — surfaced on the analyze view after a clip extracts. Asks
 * the user to confirm or override the inferred tier without sending
 * them back to /onboarding. The chip renders only when the inference
 * disagrees with the stored tier (or the user has no stored tier yet);
 * skip-rendering when they already match avoids nagging the user on
 * every clip.
 */
export default function TierOverrideChip({
  inferredTier,
  currentTier,
  reasons,
  confidence,
  onConfirm,
}: Props) {
  const [busy, setBusy] = useState(false)
  const [closed, setClosed] = useState(false)

  if (closed) return null
  if (currentTier && currentTier === inferredTier) return null

  const submit = async (tier: SkillTier) => {
    if (busy) return
    setBusy(true)
    try {
      await onConfirm(tier)
      setClosed(true)
    } finally {
      setBusy(false)
    }
  }

  const inferredLabel = SKILL_TIER_LABELS[inferredTier]
  const reasonText = reasons.length > 0 ? reasons.slice(0, 2).join(' · ') : null

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 flex flex-col sm:flex-row sm:items-center gap-3">
      <div className="flex-1 min-w-0">
        <p className="text-white text-sm">
          We&apos;re reading this swing as{' '}
          <span className="font-semibold text-emerald-300">{inferredLabel}</span>
          {currentTier && (
            <span className="text-white/50">
              {' '}(your profile says <span className="text-white/70">{SKILL_TIER_LABELS[currentTier]}</span>)
            </span>
          )}
          .
        </p>
        {reasonText && (
          <p className="text-white/50 text-xs mt-1">
            {reasonText} · confidence {Math.round(confidence * 100)}%
          </p>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => submit(inferredTier)}
          disabled={busy}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500 hover:bg-emerald-400 text-white disabled:opacity-50 transition-colors"
        >
          That matches
        </button>
        {PICK_ORDER.filter((t) => t !== inferredTier).map((t) => (
          <button
            key={t}
            onClick={() => submit(t)}
            disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 text-white/70 hover:text-white disabled:opacity-50 transition-colors"
          >
            {SKILL_TIER_LABELS[t]}
          </button>
        ))}
        <button
          onClick={() => setClosed(true)}
          className="px-2 py-1.5 rounded-lg text-xs text-white/40 hover:text-white/70"
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  )
}
