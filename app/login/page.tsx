'use client'

import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

type Mode = 'signin' | 'signup'

function AuthForm() {
  const params = useSearchParams()
  const nextRaw = params.get('next') ?? '/baseline'
  // Guard against open-redirect: only allow same-origin paths.
  const next = nextRaw.startsWith('/') && !nextRaw.startsWith('//') ? nextRaw : '/baseline'
  const router = useRouter()

  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const switchMode = (m: Mode) => {
    if (m === mode) return
    setMode(m)
    setError(null)
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)

    const supabase = createClient()

    // Sign-up: create a pre-confirmed user via the admin endpoint, then fall
    // through to signInWithPassword to set cookies.
    if (mode === 'signup') {
      const res = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password }),
      })
      if (!res.ok) {
        let msg = `Signup failed (${res.status})`
        try {
          const j = await res.json()
          if (j?.error) msg = j.error
        } catch {
          /* fall through to default */
        }
        setError(msg)
        setBusy(false)
        return
      }
    }

    const { error: signInErr } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    })
    if (signInErr) {
      setError(
        mode === 'signin'
          ? 'Wrong email or password.'
          : `Account created, but sign-in failed: ${signInErr.message}`
      )
      setBusy(false)
      return
    }

    // Refresh server components so Nav + gated pages see the new session,
    // then navigate to the requested next page.
    router.refresh()
    router.replace(next)
  }

  const isSignIn = mode === 'signin'

  return (
    <div className="max-w-md mx-auto px-4 py-16">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-black text-white mb-2">
          {isSignIn ? 'Welcome back' : 'Create an account'}
        </h1>
        <p className="text-white/50 text-sm">
          {isSignIn
            ? 'Sign in to pick up where you left off.'
            : 'Save baselines and compare every future swing against them.'}
        </p>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-full mb-6">
        <button
          type="button"
          onClick={() => switchMode('signin')}
          className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            isSignIn ? 'bg-white text-black' : 'text-white/50 hover:text-white'
          }`}
        >
          Sign in
        </button>
        <button
          type="button"
          onClick={() => switchMode('signup')}
          className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            !isSignIn ? 'bg-white text-black' : 'text-white/50 hover:text-white'
          }`}
        >
          Create account
        </button>
      </div>

      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="block text-sm text-white/60 mb-2" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
            disabled={busy}
          />
        </div>

        <div>
          <label className="block text-sm text-white/60 mb-2" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            type="password"
            required
            minLength={6}
            autoComplete={isSignIn ? 'current-password' : 'new-password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={isSignIn ? 'Your password' : 'At least 6 characters'}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50"
            disabled={busy}
          />
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button
          type="submit"
          disabled={busy || !email.trim() || !password}
          className="w-full px-4 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {busy
            ? isSignIn
              ? 'Signing in...'
              : 'Creating account...'
            : isSignIn
              ? 'Sign in'
              : 'Create account'}
        </button>

        <p className="text-center text-white/40 text-xs pt-2">
          You can still{' '}
          <Link href="/analyze" className="text-white/60 hover:text-white underline">
            analyze a swing without an account
          </Link>
          . Your data just won&apos;t persist.
        </p>
      </form>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="max-w-md mx-auto px-4 py-16 text-white/50">Loading...</div>
      }
    >
      <AuthForm />
    </Suspense>
  )
}
