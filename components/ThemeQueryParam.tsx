'use client'

import { useEffect } from 'react'

/*
 * Theme resolver. Priority:
 *   1. URL override: ?light=1 (force light) or ?light=0 (force dark).
 *      Persists to localStorage so the choice survives navigation.
 *   2. Saved localStorage preference from a previous URL override.
 *   3. OS preference via prefers-color-scheme: light. (Phase 1.4b —
 *      auto-detect, on the assumption that a user who runs the OS in
 *      light mode would prefer the app to match.)
 *   4. Default: dark.
 *
 * Listens for OS theme changes too — flipping the OS theme while the
 * app is open re-resolves immediately, but only when the user hasn't
 * manually pinned a preference.
 */
function applyTheme(theme: 'light' | 'dark') {
  if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light')
  else document.documentElement.removeAttribute('data-theme')
}

function resolveTheme(): 'light' | 'dark' {
  const params = new URLSearchParams(window.location.search)
  const param = params.get('light')
  if (param === '1') {
    window.localStorage.setItem('tia.theme', 'light')
    return 'light'
  }
  if (param === '0') {
    window.localStorage.setItem('tia.theme', 'dark')
    return 'dark'
  }
  const saved = window.localStorage.getItem('tia.theme')
  if (saved === 'light' || saved === 'dark') return saved
  if (window.matchMedia('(prefers-color-scheme: light)').matches) return 'light'
  return 'dark'
}

export default function ThemeQueryParam() {
  useEffect(() => {
    if (typeof window === 'undefined') return
    applyTheme(resolveTheme())

    // Re-resolve when the OS theme flips. Only matters when the user
    // hasn't pinned a preference via ?light=1/0 (saved state wins).
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const onChange = () => {
      const saved = window.localStorage.getItem('tia.theme')
      if (saved === 'light' || saved === 'dark') return
      applyTheme(mq.matches ? 'light' : 'dark')
    }
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  return null
}
