'use client'

import { useState } from 'react'

interface Props {
  sessionId: string | null
  blobUrl: string
  cueTitle: string
  cueBody: string
  signedIn: boolean
}

/*
 * Phase 4.1 — "Send to my coach" button surfaced next to the LLM cue.
 *
 * On click: POST /api/coach-review with the cue context, get back a
 * shareUrl, then offer the user navigator.share / clipboard copy. No
 * server-side email/SMS this round — the user pastes the link into
 * iMessage / WhatsApp and sends it themselves.
 */
export default function SendToCoachButton({ sessionId, blobUrl, cueTitle, cueBody, signedIn }: Props) {
  const [busy, setBusy] = useState(false)
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const create = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      const res = await fetch('/api/coach-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, blobUrl, cueTitle, cueBody }),
      })
      if (!res.ok) {
        const msg = (await res.json().catch(() => ({}))).error ?? `HTTP ${res.status}`
        throw new Error(msg)
      }
      const body = (await res.json()) as { shareUrl: string }
      setShareUrl(body.shareUrl)

      // Use the Web Share API if available (mobile Safari + Chrome).
      // Falls back to clipboard copy below.
      if (typeof navigator !== 'undefined' && typeof navigator.share === 'function') {
        try {
          await navigator.share({
            title: 'Quick coach review',
            text: `Mind reviewing this swing? "${cueTitle}"`,
            url: body.shareUrl,
          })
        } catch {
          /* user cancelled share — fall through to copy fallback */
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create review')
    } finally {
      setBusy(false)
    }
  }

  const copy = async () => {
    if (!shareUrl) return
    try {
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      setError('Copy failed; long-press the link to copy manually.')
    }
  }

  if (!signedIn) return null

  if (shareUrl) {
    return (
      <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3 flex flex-col gap-2 text-sm">
        <p className="text-emerald-300 font-medium">Coach review link ready</p>
        <code className="text-white/70 text-xs break-all bg-black/40 rounded px-2 py-1">{shareUrl}</code>
        <div className="flex gap-2">
          <button
            onClick={copy}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500 hover:bg-emerald-400 text-white transition-colors"
          >
            {copied ? 'Copied' : 'Copy link'}
          </button>
          <a
            href={`sms:?body=${encodeURIComponent(`Mind reviewing this swing? "${cueTitle}" ${shareUrl}`)}`}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/10 hover:bg-white/15 text-white/80 transition-colors"
          >
            Text it
          </a>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={create}
        disabled={busy}
        className="text-xs px-3 py-1.5 rounded-lg font-medium bg-white/5 hover:bg-white/10 text-white/70 hover:text-white border border-white/10 transition-colors disabled:opacity-50"
      >
        {busy ? 'Creating link…' : 'Send to my coach'}
      </button>
      {error && <p className="text-rose-300 text-xs">{error}</p>}
    </div>
  )
}
