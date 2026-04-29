'use client'

import { useEffect, useState, use } from 'react'

/*
 * Phase 4.1 — Public coach-review page.
 *
 * No auth: access is gated by the unguessable review id in the URL.
 * Coaches typically open this from a text message link, watch the
 * 5-second clip, tap "Looks right" or "Add this:" with an optional
 * note, and walk away. Page is mobile-first since most coaches will
 * view this on a phone between lessons.
 */

type Review = {
  id: string
  blob_url: string
  cue_title: string
  cue_body: string
  coach_label: string | null
  verdict: 'looks_right' | 'add_this' | null
  note: string | null
  responded_at: string | null
  created_at: string
}

export default function CoachReviewPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const [review, setReview] = useState<Review | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [verdict, setVerdict] = useState<'looks_right' | 'add_this' | null>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch(`/api/coach-review/${encodeURIComponent(id)}`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `HTTP ${r.status}`)
        return r.json() as Promise<{ review: Review }>
      })
      .then(({ review: r }) => {
        if (cancelled) return
        setReview(r)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [id])

  const submit = async (chosen: 'looks_right' | 'add_this') => {
    if (submitting) return
    if (chosen === 'add_this' && !note.trim()) {
      setVerdict('add_this')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`/api/coach-review/${encodeURIComponent(id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict: chosen, note: note.trim() || null }),
      })
      if (!res.ok) {
        throw new Error((await res.json().catch(() => ({}))).error ?? `HTTP ${res.status}`)
      }
      const body = (await res.json()) as { review: Pick<Review, 'verdict' | 'note' | 'responded_at'> }
      setReview((prev) => (prev ? { ...prev, ...body.review } : prev))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to submit')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="max-w-md mx-auto px-4 py-12 text-center">
        <p className="text-white/50 text-sm">Loading review…</p>
      </div>
    )
  }

  if (error && !review) {
    return (
      <div className="max-w-md mx-auto px-4 py-12 text-center">
        <p className="text-rose-300 text-sm">{error}</p>
      </div>
    )
  }

  if (!review) return null

  // Already-responded view: show the verdict + note, don't re-prompt.
  if (review.responded_at && review.verdict) {
    return (
      <div className="max-w-md mx-auto px-4 py-10">
        <p className="text-xs uppercase tracking-wide text-white/50 mb-2">Coach review</p>
        <h1 className="text-xl font-semibold text-white mb-4">{review.cue_title}</h1>
        <video
          src={review.blob_url}
          controls
          playsInline
          className="w-full rounded-xl bg-black mb-4"
        />
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 mb-3">
          <p className="text-emerald-300 text-sm font-medium mb-1">
            {review.verdict === 'looks_right' ? 'Marked “looks right”' : 'Marked “add this”'}
          </p>
          {review.note && <p className="text-white/80 text-sm leading-relaxed">{review.note}</p>}
          <p className="text-white/40 text-xs mt-2">
            Responded {new Date(review.responded_at).toLocaleString()}
          </p>
        </div>
        <p className="text-white/40 text-xs">
          Thanks — your reply has been sent back to the player.
        </p>
      </div>
    )
  }

  return (
    <div className="max-w-md mx-auto px-4 py-10">
      <p className="text-xs uppercase tracking-wide text-white/50 mb-2">Coach review</p>
      <h1 className="text-xl font-semibold text-white mb-2">{review.cue_title}</h1>
      <p className="text-white/60 text-sm leading-relaxed mb-4">{review.cue_body}</p>
      <video
        src={review.blob_url}
        controls
        playsInline
        muted
        className="w-full rounded-xl bg-black mb-5"
      />
      {error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 mb-4">
          <p className="text-rose-300 text-sm">{error}</p>
        </div>
      )}
      <div className="flex flex-col gap-3">
        <button
          onClick={() => submit('looks_right')}
          disabled={submitting}
          className="w-full px-4 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-semibold transition-colors disabled:opacity-50"
        >
          Looks right
        </button>
        <div>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder={verdict === 'add_this' ? 'What would you add or change?' : 'Or, type a quick note and tap “Add this”'}
            rows={3}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-sm placeholder:text-white/30 mb-2"
          />
          <button
            onClick={() => submit('add_this')}
            disabled={submitting || !note.trim()}
            className="w-full px-4 py-3 rounded-xl bg-white/10 hover:bg-white/15 text-white font-semibold transition-colors disabled:opacity-30"
          >
            Add this
          </button>
        </div>
      </div>
    </div>
  )
}
