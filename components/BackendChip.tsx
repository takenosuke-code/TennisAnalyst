'use client'

import type { ExtractorBackend } from '@/lib/poseExtraction'

// Diagnostic chip surfacing which backend produced the keypoints.
// When tracing looks bad in the browser the user can immediately see
// whether they're on RTMPose · Railway (CPU server), RTMPose · Modal
// GPU (server GPU path), RTMPose · Browser (in-browser ONNX, the
// upload-default for some flows), or RTMPose · Browser (fallback)
// (Railway failed silently and the client took over).
const LABELS: Record<ExtractorBackend, { text: string; classes: string }> = {
  'rtmpose-railway': {
    text: 'RTMPose · Railway',
    classes: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  },
  'rtmpose-modal': {
    text: 'RTMPose · Modal GPU',
    classes: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
  },
  'rtmpose-browser': {
    text: 'RTMPose · Browser',
    classes: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  },
  'rtmpose-browser-fallback': {
    text: 'RTMPose · Browser (fallback)',
    classes: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
  },
}

// Human-readable explanation of why Railway fell back. Shown next to
// the red fallback chip so we can debug without opening DevTools.
const REASON_HINTS: Record<string, string> = {
  'not-configured': 'Vercel env missing RAILWAY_SERVICE_URL or EXTRACT_API_KEY',
  'queue-failed': 'Railway rejected the job (auth or bad request)',
  'error-status': 'Railway crashed mid-extract',
  timeout: 'Railway exceeded the client budget (cold start? OOM? slow model load?)',
  aborted: 'extraction was cancelled',
}

// Reason strings are now formatted as 'category: message' (e.g.
// 'error-status: YOLO failed: CUDA OOM'). Look up the hint by the
// category prefix so the user still gets a description for the broad
// failure mode, while the raw text below shows the actual error.
function lookupHint(reason: string): string | undefined {
  const colon = reason.indexOf(':')
  const category = colon === -1 ? reason : reason.slice(0, colon)
  return REASON_HINTS[category]
}

export default function BackendChip({
  backend,
  reason,
  className = '',
}: {
  backend: ExtractorBackend | null
  reason?: string | null
  className?: string
}) {
  if (!backend) return null
  const { text, classes } = LABELS[backend]
  const hint = reason ? lookupHint(reason) ?? reason : null
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${classes} ${className}`}
      title={
        hint
          ? `${text} — ${hint}`
          : 'Which extractor produced these keypoints. Surfaced for debugging when tracing looks off.'
      }
    >
      <span>{text}</span>
      {reason && (
        <span className="opacity-70 normal-case">· {reason}</span>
      )}
    </span>
  )
}
