'use client'

import { useEffect, useMemo } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import BaselineCard from '@/components/BaselineCard'
import { useBaselineStore } from '@/store/baseline'
import { useUser } from '@/hooks/useUser'

export default function BaselinePage() {
  const { baselines, loading, error, refresh, setActive, rename, remove } = useBaselineStore()
  const { user, loading: authLoading } = useUser()
  const router = useRouter()

  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace('/login?next=/baseline')
      return
    }
    refresh()
  }, [authLoading, user, refresh, router])

  const { active, history } = useMemo(() => {
    const active = baselines.filter((b) => b.is_active)
    const history = baselines.filter((b) => !b.is_active)
    return { active, history }
  }, [baselines])

  return (
    <div className="max-w-4xl mx-auto px-5 sm:px-8 py-12">
      <div className="mb-10 flex items-start justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cream/60 mb-3">your baselines</p>
          <h1 className="font-display font-extrabold text-cream text-4xl sm:text-5xl leading-[1.05] tracking-tight mb-4">
            best-day swings,<br />saved for later.
          </h1>
          <p className="text-cream/70 max-w-xl text-sm sm:text-base leading-relaxed">
            Upload a new swing and we&apos;ll show you what improved against the baseline you pinned.
          </p>
        </div>
        <Link
          href="/baseline/compare"
          className="shrink-0 px-5 py-2.5 rounded-full bg-clay hover:bg-[#c4633f] text-cream font-semibold text-sm tracking-wide transition-colors"
        >
          compare new swing
        </Link>
      </div>

      {error && (
        <div className="bg-cream p-4 mb-6 border-l-4 border-clay">
          <p className="text-ink text-sm">{error}</p>
        </div>
      )}

      {loading && baselines.length === 0 && (
        <div className="bg-cream/10 p-8 text-center">
          <p className="text-cream/60 text-sm">Loading baselines...</p>
        </div>
      )}

      {!loading && baselines.length === 0 && !error && (
        <div className="bg-cream text-ink">
          <div className="h-2 bg-clay" />
          <div className="p-12 text-center">
            {/* Tennis ball — same SVG glyph used in Nav. Replaces the 🎾 emoji. */}
            <svg viewBox="0 0 24 24" className="w-10 h-10 mx-auto mb-4 text-clay" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <circle cx="12" cy="12" r="9" />
              <path d="M3.5 9c4 1 8 1 12 0M3.5 16c4-1 8-1 12 0" transform="translate(2,-1)" />
            </svg>
            <p className="font-display font-bold text-ink text-xl mb-2">No baselines yet</p>
            <p className="text-ink/60 text-sm mb-6 max-w-sm mx-auto">
              Analyze a swing and mark it as your baseline to start tracking progress.
            </p>
            <Link
              href="/analyze"
              className="inline-block px-6 py-3 rounded-full bg-ink hover:bg-ink-soft text-cream font-semibold text-sm tracking-wide transition-colors"
            >
              analyze a swing
            </Link>
          </div>
        </div>
      )}

      {active.length > 0 && (
        <section className="space-y-3 mb-10">
          <h2 className="text-[11px] font-semibold text-cream/60 uppercase tracking-[0.18em]">Active</h2>
          <div className="space-y-3">
            {active.map((b) => (
              <BaselineCard
                key={b.id}
                baseline={b}
                onRename={rename}
                onDelete={remove}
              />
            ))}
          </div>
        </section>
      )}

      {history.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-[11px] font-semibold text-cream/60 uppercase tracking-[0.18em]">History</h2>
          <div className="space-y-3">
            {history.map((b) => (
              <BaselineCard
                key={b.id}
                baseline={b}
                onSetActive={setActive}
                onRename={rename}
                onDelete={remove}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
