import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/lib/supabase'

// Create a password-based user with the email already confirmed. Skips the
// default "click the link in your inbox" confirmation flow entirely.
//
// Uses the service-role admin client (SUPABASE_SECRETKEY). Never invoked from
// the browser directly — the client hits this route, then separately calls
// signInWithPassword() to set the auth cookies.
export async function POST(request: NextRequest) {
  let body: { email?: unknown; password?: unknown }
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const email = typeof body.email === 'string' ? body.email.trim() : ''
  const password = typeof body.password === 'string' ? body.password : ''

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ error: 'Enter a valid email' }, { status: 400 })
  }
  if (password.length < 6) {
    return NextResponse.json(
      { error: 'Password must be at least 6 characters' },
      { status: 400 }
    )
  }

  const { data, error } = await supabaseAdmin.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  })

  if (error) {
    // Supabase returns 422 with "already registered" for duplicates.
    const isDup = /already|exists|registered/i.test(error.message)
    return NextResponse.json(
      { error: isDup ? 'An account with this email already exists. Try signing in instead.' : error.message },
      { status: isDup ? 409 : 400 }
    )
  }

  return NextResponse.json({ userId: data.user?.id, email: data.user?.email })
}
