'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import VideoCanvas from '@/components/VideoCanvas'
import { useUser } from '@/hooks/useUser'
import type { Baseline } from '@/lib/supabase'
import type { JointGroup } from '@/lib/jointAngles'

// Probe the blob URL with a HEAD request before rendering the video.
// Saves the user a confusing blank/black <video> state when the
// underlying file has been deleted (e.g. by the bulk-cleanup script
// or a Vercel Blob quota purge) — the row in user_baselines is still
// there but its blob_url is dead.
async function probeBlob(url: string, signal: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(url, { method: 'HEAD', signal })
    return res.ok
  } catch {
    return false
  }
}

const ALL_VISIBLE: Record<JointGroup, boolean> = {
  shoulders: true,
  elbows: true,
  wrists: true,
  hips: true,
  knees: true,
  ankles: true,
}

const ALL_HIDDEN: Record<JointGroup, boolean> = {
  shoulders: false,
  elbows: false,
  wrists: false,
  hips: false,
  knees: false,
  ankles: false,
}

export default function BaselineWatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const { user, loading: authLoading } = useUser()
  const router = useRouter()
  const [baseline, setBaseline] = useState<Baseline | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [videoMissing, setVideoMissing] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (authLoading) return
    if (!user) {
      router.replace(`/login?next=/baseline/${id}`)
      return
    }
    const abort = new AbortController()
    let cancelled = false
    setLoading(true)
    setVideoMissing(false)
    fetch(`/api/baselines/${encodeURIComponent(id)}`, { signal: abort.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).error || `HTTP ${r.status}`)
        return r.json()
      })
      .then(async (body: { baseline: Baseline }) => {
        if (cancelled) return
        setBaseline(body.baseline)
        setError(null)
        // HEAD-probe the blob URL. If the file is gone, render a
        // "video unavailable" branch with a delete option instead of
        // letting the <video> element fall into its silent error
        // state. Skips the probe when we have no URL to check.
        if (body.baseline?.blob_url) {
          const alive = await probeBlob(body.baseline.blob_url, abort.signal)
          if (!cancelled) setVideoMissing(!alive)
        }
      })
      .catch((e: Error) => {
        if (cancelled || e.name === 'AbortError') return
        setError(e.message)
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })
    return () => {
      cancelled = true
      abort.abort()
    }
  }, [authLoading, user, id, router])

  const handleDelete = async () => {
    if (!baseline || deleting) return
    if (!confirm('Delete this baseline? The DB row is removed; the underlying video file is already gone.')) return
    setDeleting(true)
    try {
      const res = await fetch(`/api/baselines/${encodeURIComponent(baseline.id)}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        setError((await res.json().catch(() => null))?.error ?? `HTTP ${res.status}`)
        setDeleting(false)
        return
      }
      router.push('/baseline')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed')
      setDeleting(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-6">
        <Link
          href="/baseline"
          className="text-sm text-white/50 hover:text-white transition-colors"
        >
          ← Baselines
        </Link>
      </div>

      {loading && (
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-8 text-center">
          <p className="text-white/50 text-sm">Loading baseline...</p>
        </div>
      )}

      {error && !loading && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {baseline && !loading && (
        <>
          <div className="mb-6 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="text-2xl font-black text-white truncate">{baseline.label}</h1>
              <div className="flex gap-2 items-center text-xs text-white/50 mt-1">
                <span className="capitalize">{baseline.shot_type}</span>
                <span>·</span>
                <span>{baseline.keypoints_json?.frame_count ?? 0} frames</span>
                {baseline.is_active && (
                  <>
                    <span>·</span>
                    <span className="text-emerald-400 font-medium">Active</span>
                  </>
                )}
              </div>
            </div>
            <div className="flex gap-2 shrink-0">
              <button
                onClick={() => setShowOverlay((v) => !v)}
                className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white text-xs font-medium transition-colors"
              >
                {showOverlay ? 'Hide joints' : 'Show joints'}
              </button>
              <Link
                href="/baseline/compare"
                className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white text-xs font-medium transition-colors"
              >
                Compare new swing
              </Link>
            </div>
          </div>

          {videoMissing ? (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-6 space-y-3">
              <p className="text-amber-200 font-semibold">
                Video file no longer available
              </p>
              <p className="text-amber-100/80 text-sm leading-relaxed">
                The underlying recording for this baseline has been deleted
                from storage (probably by a quota cleanup or the bulk-delete
                script). The pose data is still saved on this row, but the
                video itself can&apos;t be played back. Save a fresh swing
                as your baseline to replace it.
              </p>
              <div className="flex gap-2 pt-1">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-200 text-xs font-semibold transition-colors disabled:opacity-50"
                >
                  {deleting ? 'Deleting…' : 'Delete this baseline'}
                </button>
                <Link
                  href="/analyze"
                  className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white text-xs font-semibold transition-colors"
                >
                  Record a new one
                </Link>
              </div>
            </div>
          ) : (
            <VideoCanvas
              src={baseline.blob_url}
              framesData={baseline.keypoints_json?.frames ?? []}
              visible={showOverlay ? ALL_VISIBLE : ALL_HIDDEN}
              showSkeleton={showOverlay}
              showAngles={showOverlay}
              shotType={baseline.shot_type}
            />
          )}
        </>
      )}
    </div>
  )
}
