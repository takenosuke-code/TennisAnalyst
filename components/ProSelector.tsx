'use client'

import { useEffect, useRef, useState } from 'react'
import { useComparisonStore, usePoseStore } from '@/store'
import type { Pro, ProSwing } from '@/lib/supabase'

type ProWithSwings = Pro & { pro_swings: ProSwing[] }

const SHOT_TYPES = ['all', 'forehand', 'backhand', 'serve', 'volley'] as const

interface ProSelectorProps {
  compact?: boolean
}

export default function ProSelector({ compact = false }: ProSelectorProps) {
  const [pros, setPros] = useState<ProWithSwings[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState(false)
  const userShotType = usePoseStore((s) => s.shotType)
  const [selectedShot, setSelectedShot] = useState<string>('all')
  const [expandedPro, setExpandedPro] = useState<string | null>(null)
  const [autoSelected, setAutoSelected] = useState(false)

  const [loadingSwing, setLoadingSwing] = useState<string | null>(null)
  const { activePro, activeProSwing, setActivePro, setActiveProSwing } =
    useComparisonStore()

  const selectSwing = async (pro: ProWithSwings, swing: ProSwing) => {
    setActivePro(pro)
    setLoadingSwing(swing.id)
    try {
      const res = await fetch(`/api/pro-swings/${swing.id}?include=keypoints`)
      if (!res.ok) throw new Error('Failed to load swing data')
      const fullSwing: ProSwing = await res.json()
      setActiveProSwing(fullSwing)
    } catch {
      // Fall back to partial swing (LLM analysis still works server-side)
      setActiveProSwing(swing)
    } finally {
      setLoadingSwing(null)
    }
  }

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

  // Default the shot type filter to the user's selected shot type
  useEffect(() => {
    if (userShotType && selectedShot === 'all') {
      setSelectedShot(userShotType)
    }
  }, [userShotType, selectedShot])

  // Reset autoSelected when userShotType changes so auto-select can re-fire
  const prevShotTypeRef = useRef(userShotType)
  useEffect(() => {
    if (userShotType !== prevShotTypeRef.current) {
      prevShotTypeRef.current = userShotType
      setAutoSelected(false)
    }
  }, [userShotType])

  // Auto-select the first matching swing when pros are loaded and shot type doesn't match
  useEffect(() => {
    if (autoSelected || !userShotType || loading || pros.length === 0) return
    // If there's already an active swing that matches the current shot type, skip
    if (activeProSwing && activeProSwing.shot_type === userShotType) return

    const matchingSwings: { pro: ProWithSwings; swing: ProSwing }[] = []
    for (const pro of pros) {
      for (const swing of pro.pro_swings) {
        if (swing.shot_type === userShotType) {
          matchingSwings.push({ pro, swing })
        }
      }
    }

    if (matchingSwings.length > 0) {
      setAutoSelected(true)
      const { pro, swing } = matchingSwings[0]
      setExpandedPro(pro.id)
      selectSwing(pro, swing)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pros, loading, userShotType, activeProSwing, autoSelected])

  const filteredPros = pros.map((pro) => ({
    ...pro,
    pro_swings:
      selectedShot === 'all'
        ? pro.pro_swings
        : pro.pro_swings.filter((s) => s.shot_type === selectedShot),
  })).filter((pro) => pro.pro_swings.length > 0)

  if (loading) {
    return (
      <div className="animate-pulse space-y-2">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-14 rounded-xl bg-white/5" />
        ))}
      </div>
    )
  }

  if (fetchError) {
    return (
      <div className="text-center py-8 text-red-400/70 text-sm">
        Failed to load players. Check your connection.
      </div>
    )
  }

  if (!pros.length) {
    return (
      <div className="text-center py-8 text-white/40 text-sm">
        No pro players in database yet.
        <br />
        Run the seeding script to add pros.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Shot type filter */}
      <div className="flex gap-1.5 flex-wrap">
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
      <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
        {filteredPros.map((pro) => (
          <div
            key={pro.id}
            className={`rounded-xl border overflow-hidden transition-all ${
              activePro?.id === pro.id
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
                    disabled={loadingSwing === swing.id}
                    className={`w-full flex items-center justify-between px-4 py-2.5 text-left transition-colors ${
                      activeProSwing?.id === swing.id
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'text-white/70 hover:bg-white/5 hover:text-white'
                    }`}
                  >
                    <span className="capitalize text-sm">{swing.shot_type}</span>
                    {loadingSwing === swing.id ? (
                      <span className="text-xs text-white/40">Loading...</span>
                    ) : activeProSwing?.id === swing.id ? (
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
    </div>
  )
}
