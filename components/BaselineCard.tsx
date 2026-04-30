'use client'

import { useState } from 'react'
import Link from 'next/link'
import type { Baseline } from '@/lib/supabase'

interface BaselineCardProps {
  baseline: Baseline
  onSetActive?: (id: string) => Promise<void> | void
  onRename?: (id: string, label: string) => Promise<void> | void
  onDelete?: (id: string) => Promise<void> | void
}

// Same shot-type → stripe mapping as SegmentCard. Kept inline (no lib/)
// so the two card families share the visual language without coupling.
const SHOT_STRIPE: Record<string, string> = {
  forehand: 'bg-clay',
  backhand: 'bg-hard-court',
  serve: 'bg-lavender-2',
  volley: 'bg-clay',
  slice: 'bg-hard-court',
}

function relativeDate(iso: string): string {
  const t = new Date(iso).getTime()
  const days = Math.floor((Date.now() - t) / (1000 * 60 * 60 * 24))
  if (days === 0) return 'today'
  if (days === 1) return 'yesterday'
  if (days < 30) return `${days}d ago`
  if (days < 365) return `${Math.floor(days / 30)}mo ago`
  return `${Math.floor(days / 365)}y ago`
}

export default function BaselineCard({ baseline, onSetActive, onRename, onDelete }: BaselineCardProps) {
  const [editing, setEditing] = useState(false)
  const [labelDraft, setLabelDraft] = useState(baseline.label)
  const [busy, setBusy] = useState(false)

  const submitRename = async () => {
    const next = labelDraft.trim()
    if (!next || next === baseline.label) {
      setEditing(false)
      setLabelDraft(baseline.label)
      return
    }
    setBusy(true)
    try {
      await onRename?.(baseline.id, next)
    } finally {
      setBusy(false)
      setEditing(false)
    }
  }

  const stripe = SHOT_STRIPE[baseline.shot_type] ?? 'bg-lavender-2'

  return (
    <div
      className={`flex bg-cream text-ink ${
        baseline.is_active ? 'outline outline-2 outline-clay' : ''
      }`}
    >
      {/* Vertical shot-type stripe — replaces the old emerald border on
          is_active. The stripe color tells you the shot at a glance, the
          clay outline tells you which baseline is active. */}
      <div className={`w-2 shrink-0 ${stripe}`} />

      <div className="flex-1 p-4 flex gap-4 items-center">
        <Link
          href={`/baseline/${baseline.id}`}
          className="shrink-0 group relative"
          title="Watch with pose overlay"
        >
          <video
            src={baseline.blob_url}
            className="w-24 h-24 object-cover bg-ink"
            muted
            playsInline
            preload="metadata"
          />
          <div className="absolute inset-0 bg-ink/0 group-hover:bg-ink/40 flex items-center justify-center transition-colors">
            <svg viewBox="0 0 24 24" className="w-6 h-6 fill-cream opacity-0 group-hover:opacity-100 transition-opacity">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          </div>
        </Link>

        <div className="flex-1 min-w-0">
          {editing ? (
            <input
              autoFocus
              value={labelDraft}
              onChange={(e) => setLabelDraft(e.target.value)}
              onBlur={submitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitRename()
                if (e.key === 'Escape') {
                  setEditing(false)
                  setLabelDraft(baseline.label)
                }
              }}
              className="w-full bg-cream-soft border border-ink/15 px-2 py-1 text-ink text-sm"
            />
          ) : (
            <button
              onClick={() => setEditing(true)}
              className="text-left text-ink font-display font-bold hover:text-clay transition-colors truncate block max-w-full"
              title="Click to rename"
            >
              {baseline.label}
            </button>
          )}

          <div className="flex gap-2 items-center text-xs text-ink/55 mt-1">
            <span className="lowercase">{baseline.shot_type}</span>
            <span>·</span>
            <span>{baseline.keypoints_json?.frame_count ?? 0} frames</span>
            <span>·</span>
            <span>{relativeDate(baseline.created_at)}</span>
            {baseline.is_active && (
              <>
                <span>·</span>
                <span className="text-clay font-semibold uppercase tracking-[0.14em] text-[10px]">
                  Active
                </span>
              </>
            )}
          </div>
        </div>

        <div className="flex gap-2 shrink-0">
          <Link
            href={`/baseline/${baseline.id}`}
            className="px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide bg-cream-soft hover:bg-ink/10 text-ink transition-colors"
          >
            Watch
          </Link>
          {!baseline.is_active && onSetActive && (
            <button
              onClick={async () => {
                setBusy(true)
                try {
                  await onSetActive(baseline.id)
                } finally {
                  setBusy(false)
                }
              }}
              disabled={busy}
              className="px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide bg-ink text-cream hover:bg-ink-soft disabled:opacity-50"
            >
              Set active
            </button>
          )}
          {onDelete && (
            <button
              onClick={async () => {
                if (!confirm(`Delete baseline "${baseline.label}"?`)) return
                setBusy(true)
                try {
                  await onDelete(baseline.id)
                } finally {
                  setBusy(false)
                }
              }}
              disabled={busy}
              className="px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide bg-transparent hover:bg-clay/15 text-ink/60 hover:text-clay disabled:opacity-50"
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
