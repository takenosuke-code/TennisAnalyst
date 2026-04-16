import type { ProSwing } from '@/lib/supabase'

/**
 * Resolve a playable video URL for a pro swing.
 * YouTube URLs are mapped to local files by shot type;
 * other URLs pass through unchanged.
 */
export function getProVideoUrl(swing: ProSwing | null): string | null {
  if (!swing) return null
  const url = swing.video_url
  // YouTube URLs cannot play in <video>. Return null so the UI can
  // distinguish "no playable video" from "no swing selected".
  if (url && (url.includes('youtube.com') || url.includes('youtu.be'))) {
    return null
  }
  return url ?? null
}
