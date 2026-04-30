'use client'

import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

interface SegmentLightboxProps {
  open: boolean
  onClose: () => void
  videoUrl: string
  startMs: number
  endMs: number
  title?: string
}

/*
 * SegmentLightbox — full-screen overlay that plays a single shot
 * segment from a longer source clip. The card grids share one source
 * blobUrl and each shot is just a [startMs, endMs] range, so this
 * mounts a fresh <video>, seeks to startMs, plays, and loops back to
 * startMs as soon as currentTime crosses endMs. Looping makes the
 * forehand watchable repeatedly without the user reaching for a
 * scrubber.
 *
 * Closes on ESC, on backdrop click, and on the close button. The
 * video is torn down on close so the decoder doesn't stay pinned
 * across multiple opens.
 */
export default function SegmentLightbox({
  open,
  onClose,
  videoUrl,
  startMs,
  endMs,
  title,
}: SegmentLightboxProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)

  // Portal target lives on document.body, but only after mount —
  // SSR-safe via the conditional render below.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    // Lock body scroll while the lightbox is open so the user isn't
    // scrolling the page behind the modal.
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [open, onClose])

  // Seek + autoplay on open. The loop logic (timeupdate -> reset to
  // startMs when we cross endMs) lives inside the same effect so it
  // teardown cleanly when the lightbox closes.
  useEffect(() => {
    if (!open) return
    const video = videoRef.current
    if (!video) return

    const seekStart = () => {
      try {
        video.currentTime = startMs / 1000
        void video.play()
      } catch {
        /* duration not ready yet — durationchange retries below */
      }
    }
    const onLoaded = () => seekStart()
    const onDuration = () => seekStart()
    const onTime = () => {
      if (video.currentTime * 1000 >= endMs) {
        video.currentTime = startMs / 1000
        void video.play()
      }
    }

    video.addEventListener('loadedmetadata', onLoaded)
    video.addEventListener('durationchange', onDuration)
    video.addEventListener('timeupdate', onTime)

    // If metadata is already loaded (cached <video src> from the
    // thumbnail), kick the seek manually — loadedmetadata won't refire.
    if (video.readyState >= 1) seekStart()

    return () => {
      video.removeEventListener('loadedmetadata', onLoaded)
      video.removeEventListener('durationchange', onDuration)
      video.removeEventListener('timeupdate', onTime)
      video.pause()
      video.removeAttribute('src')
      video.load()
    }
  }, [open, startMs, endMs, videoUrl])

  if (!open) return null
  if (typeof document === 'undefined') return null

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title ?? 'Shot preview'}
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/85 backdrop-blur-sm px-4 py-8"
    >
      <div
        // Stop clicks on the player from bubbling to the backdrop.
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-5xl flex flex-col gap-3"
      >
        {title && (
          <p className="text-cream/85 text-sm font-medium tracking-wide">
            {title}
          </p>
        )}
        <div className="relative bg-black aspect-video w-full overflow-hidden">
          <video
            ref={videoRef}
            src={videoUrl}
            preload="auto"
            muted
            playsInline
            controls
            className="w-full h-full object-contain"
          />
        </div>
        <div className="flex items-center justify-between text-cream/65 text-xs">
          <span>
            Looping {(startMs / 1000).toFixed(1)}s – {(endMs / 1000).toFixed(1)}s · press ESC to close
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-full bg-cream text-ink text-xs font-semibold hover:bg-cream-soft transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
