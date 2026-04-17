'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function Nav() {
  return (
    <nav className="border-b border-white/5 bg-black/30 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-white text-lg">
          <span className="text-2xl">🎾</span>
          <span>TennisIQ</span>
        </Link>

        <div className="flex items-center gap-1">
          <NavLink href="/analyze">Analyze</NavLink>
          <NavLink href="/compare">Compare</NavLink>
          <NavLink href="/pros">Pro Library</NavLink>
          <NavLink href="/admin/tag-clips">Tag Clips</NavLink>
        </div>
      </div>
    </nav>
  )
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  const pathname = usePathname()
  const isActive = pathname != null && (pathname === href || pathname.startsWith(href + '/'))

  return (
    <Link
      href={href}
      className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
        isActive
          ? 'text-white bg-white/10'
          : 'text-white/60 hover:text-white hover:bg-white/5'
      }`}
    >
      {children}
    </Link>
  )
}
