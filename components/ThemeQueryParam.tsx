'use client'

import { useEffect } from 'react'

/*
 * Phase 1.4a: read ?light=1 (or ?light=0 to force-dark) on first client
 * render and toggle a `data-theme` attribute on <html>. Persists in
 * localStorage so a user who opted in on a sunny court stays in light
 * across navigations without keeping the param glued to every URL.
 *
 * Phase 1.4b will replace this with prefers-color-scheme auto-detect
 * once we've confirmed the light palette doesn't break any
 * dark-mode-only component (segment cards, joint-toggle pills, etc.)
 * via a real outdoor dogfood pass.
 */
export default function ThemeQueryParam() {
  useEffect(() => {
    if (typeof window === 'undefined') return
    const params = new URLSearchParams(window.location.search)
    const param = params.get('light')

    let theme: 'light' | 'dark' | null = null
    if (param === '1') theme = 'light'
    else if (param === '0') theme = 'dark'

    if (theme) {
      window.localStorage.setItem('tia.theme', theme)
    } else {
      const saved = window.localStorage.getItem('tia.theme')
      if (saved === 'light' || saved === 'dark') theme = saved
    }

    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light')
    } else {
      document.documentElement.removeAttribute('data-theme')
    }
  }, [])

  return null
}
