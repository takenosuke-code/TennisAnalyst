// Client-side helpers for admin password auth. The password is cached in
// localStorage so the user isn't prompted on every page load.
//
// Security note: localStorage is NOT a secure store — any JS on the page can
// read it, and a malicious extension or a future XSS bug could exfiltrate it.
// This auth model only guards against random internet visitors, not active
// attackers. For stronger auth (per-user sessions, CSRF, etc.) migrate to
// Supabase Auth or Vercel Password Protection.

export const ADMIN_TOKEN_HEADER = 'x-clip-admin-token'
const LS_KEY = 'clip_admin_token'

export function getAdminToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage.getItem(LS_KEY)
  } catch {
    // localStorage can throw in private modes
    return null
  }
}

export function setAdminToken(token: string): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(LS_KEY, token)
  } catch {
    // best-effort
  }
}

export function clearAdminToken(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LS_KEY)
  } catch {
    // best-effort
  }
}

export function adminAuthHeaders(): HeadersInit {
  const token = getAdminToken()
  return token ? { [ADMIN_TOKEN_HEADER]: token } : {}
}

/**
 * Calls /api/admin/ping to verify a candidate token. Returns true iff the
 * server accepts it. Does NOT persist on success — the caller decides.
 */
export async function verifyAdminToken(candidate: string): Promise<boolean> {
  try {
    const res = await fetch('/api/admin/ping', {
      method: 'GET',
      headers: { [ADMIN_TOKEN_HEADER]: candidate },
    })
    return res.ok
  } catch {
    return false
  }
}
