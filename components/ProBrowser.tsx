'use client'

import { useEffect, useState } from 'react'
import { useProLibraryStore } from '@/store'
import type { Pro, ProSwing } from '@/lib/supabase'

type ProWithSwings = Pro & { pro_swings: ProSwing[] }

const SHOT_TYPES = ['all', 'forehand', 'backhand', 'serve', 'volley'] as const

const SHOT_TYPE_COLORS: Record<string, string> = {
  forehand: 'bg-emerald-500/20 text-emerald-400',
  backhand: 'bg-blue-500/20 text-blue-400',
  serve: 'bg-orange-500/20 text-orange-400',
  volley: 'bg-violet-500/20 text-violet-400',
}

export default function ProBrowser() {
  const [pros, setPros] = useState<ProWithSwings[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState(false)
  const [selectedShot, setSelectedShot] = useState<string>('all')
  const [expandedPro, setExpandedPro] = useState<string | null>(null)
  const [loadingSwingId, setLoadingSwingId] = useState<string | null>(null)

  const { selectedPro, selectedSwing, setSelectedPro, setSelectedSwing, setLoadingSwing, resetChat } =
    useProLibraryStore()

  useEffect(() => {
    fetch('/api/pros')
      .then((r) => {
        if (!r.ok) throw new Error('API error')
        return r.json()
      })
      .then((data) => {
        setPros(data ?? [])
        setLoading(false)
      })
      .catch(() => {
        setFetchError(true)
        setLoading(false)
      })
  }, [])

  const selectSwing = async (pro: ProWithSwings, swing: ProSwing) => {
    setSelectedPro(pro)
    setLoadingSwingId(swing.id)
    setLoadingSwing(true)
    resetChat()
    try {
      const res = await fetch(`/api/pro-swings/${swing.id}?include=keypoints`)
      if (!res.ok) throw new Error('Failed to load swing data')
      const fullSwing: ProSwing = await res.json()
      setSelectedSwing(fullSwing)
    } catch {
      // Fall back to partial swing data
      setSelectedSwing(swing)
    } finally {
      setLoadingSwingId(null)
      setLoadingSwing(false)
    }
  }

  const filteredPros = pros
    .map((pro) => ({
      ...pro,
      pro_swings:
        selectedShot === 'all'
          ? pro.pro_swings
          : pro.pro_swings.filter((s) => s.shot_type === selectedShot),
    }))
    .filter((pro) => pro.pro_swings.length > 0)

  if (loading) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
        <h3 className="text-sm font-semibold text-white mb-3">Browse Pros</h3>
        <div className="animate-pulse space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-14 rounded-xl bg-white/5" />
          ))}
        </div>
      </div>
    )
  }

  if (fetchError) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
        <h3 className="text-sm font-semibold text-white mb-3">Browse Pros</h3>
        <div className="text-center py-8 text-red-400/70 text-sm">
          Failed to load players. Check your connection.
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 max-h-[calc(100vh-120px)] overflow-y-auto">
      <h3 className="text-sm font-semibold text-white mb-3">Browse Pros</h3>

      {/* Shot type filter pills */}
      <div className="flex gap-1.5 flex-wrap mb-3">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setSelectedShot(type)}
            className={`px-3 py-1 rounded-full text-xs font-medium capitalize transition-all ${
              selectedShot === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/50 hover:bg-white/15 hover:text-white'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Pro list */}
      {filteredPros.length === 0 ? (
        <div className="text-center py-8 text-white/40 text-sm">
          No pros found for this shot type.
        </div>
      ) : (
        <div className="space-y-2">
          {filteredPros.map((pro) => (
            <div
              key={pro.id}
              className={`rounded-xl border overflow-hidden transition-all ${
                selectedPro?.id === pro.id
                  ? 'border-emerald-500/50 bg-emerald-500/10'
                  : 'border-white/10 bg-white/5'
              }`}
            >
              <button
                onClick={() =>
                  setExpandedPro(expandedPro === pro.id ? null : pro.id)
                }
                className="w-full flex items-center gap-3 px-4 py-3 text-left"
              >
                {pro.profile_image_url ? (
                  <img
                    src={pro.profile_image_url}
                    alt={pro.name}
                    className="w-9 h-9 rounded-full object-cover flex-shrink-0"
                  />
                ) : (
                  <div className="w-9 h-9 rounded-full bg-white/10 flex items-center justify-center text-sm flex-shrink-0">
                    {pro.name[0]}
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium text-sm truncate">
                    {pro.name}
                  </p>
                  <p className="text-white/40 text-xs">
                    {pro.nationality} ·{' '}
                    {pro.pro_swings.length} swing
                    {pro.pro_swings.length !== 1 ? 's' : ''}
                  </p>
                </div>
                <svg
                  className={`w-4 h-4 text-white/30 transition-transform ${
                    expandedPro === pro.id ? 'rotate-180' : ''
                  }`}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </button>

              {/* Swing list */}
              {expandedPro === pro.id && (
                <div className="border-t border-white/10 divide-y divide-white/5">
                  {pro.pro_swings.map((swing) => (
                    <button
                      key={swing.id}
                      onClick={() => selectSwing(pro, swing)}
                      disabled={loadingSwingId === swing.id}
                      className={`w-full flex items-center justify-between px-4 py-2.5 text-left transition-colors ${
                        selectedSwing?.id === swing.id
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : 'text-white/70 hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-xs font-medium capitalize px-2 py-0.5 rounded-full ${
                            SHOT_TYPE_COLORS[swing.shot_type] ?? 'bg-white/10 text-white/60'
                          }`}
                        >
                          {swing.shot_type}
                          {swing.metadata?.camera_angle ? ` (${swing.metadata.camera_angle})` : ''}
                        </span>
                        {swing.duration_ms && (
                          <span className="text-white/30 text-xs font-mono">
                            {(swing.duration_ms / 1000).toFixed(1)}s
                          </span>
                        )}
                      </div>
                      {loadingSwingId === swing.id ? (
                        <span className="text-xs text-white/40">Loading...</span>
                      ) : selectedSwing?.id === swing.id ? (
                        <span className="text-xs bg-emerald-500 text-white px-2 py-0.5 rounded-full">
                          Selected
                        </span>
                      ) : null}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
