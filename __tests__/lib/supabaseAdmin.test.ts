import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock @supabase/supabase-js before importing the module under test
const mockCreateClient = vi.fn().mockReturnValue({ from: vi.fn() })
vi.mock('@supabase/supabase-js', () => ({
  createClient: (...args: unknown[]) => mockCreateClient(...args),
}))

describe('supabase client exports', () => {
  const ORIGINAL_ENV = process.env

  beforeEach(() => {
    vi.resetModules()
    process.env = {
      ...ORIGINAL_ENV,
      NEXT_PUBLIC_SUPABASE_URL: 'https://test.supabase.co',
      NEXT_PUBLIC_SUPABASE_ANON_KEY: 'anon-key-123',
      SUPABASE_SERVICE_KEY: 'service-key-456',
    }
    mockCreateClient.mockClear()
  })

  afterEach(() => {
    process.env = ORIGINAL_ENV
  })

  it('getSupabase creates client with anon key', async () => {
    const { getSupabase } = await import('@/lib/supabase')
    getSupabase()
    expect(mockCreateClient).toHaveBeenCalledWith(
      'https://test.supabase.co',
      'anon-key-123'
    )
  })

  it('getSupabaseAdmin creates client with service key', async () => {
    const { getSupabaseAdmin } = await import('@/lib/supabase')
    getSupabaseAdmin()
    expect(mockCreateClient).toHaveBeenCalledWith(
      'https://test.supabase.co',
      'service-key-456'
    )
  })

  it('getSupabaseAdmin throws when SUPABASE_SERVICE_KEY is missing', async () => {
    delete process.env.SUPABASE_SERVICE_KEY
    const { getSupabaseAdmin } = await import('@/lib/supabase')
    expect(() => getSupabaseAdmin()).toThrow('SUPABASE_SERVICE_KEY')
  })

  it('getSupabase returns same instance on repeated calls', async () => {
    const { getSupabase } = await import('@/lib/supabase')
    const a = getSupabase()
    const b = getSupabase()
    expect(a).toBe(b)
    expect(mockCreateClient).toHaveBeenCalledTimes(1)
  })

  it('getSupabaseAdmin returns same instance on repeated calls', async () => {
    const { getSupabaseAdmin } = await import('@/lib/supabase')
    const a = getSupabaseAdmin()
    const b = getSupabaseAdmin()
    expect(a).toBe(b)
    // Only 1 call for admin (anon client not instantiated)
    expect(mockCreateClient).toHaveBeenCalledTimes(1)
  })
})
