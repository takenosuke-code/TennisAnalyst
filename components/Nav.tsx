'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useUser } from '@/hooks/useUser'

export default function Nav() {
  const { user, loading } = useUser()

  return (
    <nav className="border-b border-cream/10 bg-ink/40 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 font-display font-extrabold text-cream text-lg tracking-tight">
          {/* Tennis ball — small inline SVG with the curve seam. Replaces
              the 🎾 emoji so the brand mark inherits the new aesthetic. */}
          <svg viewBox="0 0 24 24" className="w-5 h-5 text-clay-soft" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
            <circle cx="12" cy="12" r="9" />
            <path d="M3.5 9c4 1 8 1 12 0M3.5 16c4-1 8-1 12 0" transform="translate(2,-1)" />
          </svg>
          <span>Swingframe</span>
        </Link>

        <div className="flex items-center gap-1">
          <NavLink href="/live">Live</NavLink>
          <NavLink href="/analyze">Analyze</NavLink>
          <NavLink href="/baseline">Baselines</NavLink>

          <div className="ml-2 pl-2 border-l border-cream/15">
            {loading ? (
              <div className="w-8 h-6" aria-hidden />
            ) : user ? (
              <AccountMenu email={user.email ?? 'Account'} />
            ) : (
              <Link
                href="/login"
                className="px-4 py-1.5 rounded-full text-sm font-semibold tracking-wide bg-clay hover:bg-[#c4633f] text-cream transition-colors"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

function AccountMenu({ email }: { email: string }) {
  // Lightweight dropdown — pure CSS details/summary to avoid click-outside plumbing.
  const initial = email[0]?.toUpperCase() ?? '?'
  return (
    <details className="relative group">
      <summary className="list-none cursor-pointer w-8 h-8 rounded-full bg-clay/30 border border-clay/40 flex items-center justify-center text-clay-soft text-sm font-semibold hover:bg-clay/40">
        {initial}
      </summary>
      <div className="absolute right-0 mt-2 w-56 border border-ink/40 bg-ink shadow-xl overflow-hidden">
        <div className="px-3 py-2 border-b border-cream/10">
          <p className="text-cream/50 text-xs">Signed In As</p>
          <p className="text-cream text-sm truncate">{email}</p>
        </div>
        <Link
          href="/profile"
          className="block w-full text-left px-3 py-2 text-sm text-cream/70 hover:bg-cream/5 hover:text-cream transition-colors border-b border-cream/10"
        >
          Profile
        </Link>
        <form action="/auth/signout" method="post">
          <button
            type="submit"
            className="w-full text-left px-3 py-2 text-sm text-cream/70 hover:bg-cream/5 hover:text-cream transition-colors"
          >
            Sign Out
          </button>
        </form>
      </div>
    </details>
  )
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  const pathname = usePathname()
  const isActive = pathname != null && (pathname === href || pathname.startsWith(href + '/'))

  return (
    <Link
      href={href}
      className={`px-3 py-1.5 text-sm font-medium tracking-wide transition-colors ${
        isActive
          ? 'text-cream'
          : 'text-cream/60 hover:text-cream'
      }`}
    >
      {children}
    </Link>
  )
}
