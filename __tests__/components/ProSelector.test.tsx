import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ProSelector from '@/components/ProSelector'
import type { Pro, ProSwing } from '@/lib/supabase'

// Store mock values
let mockComparisonState = {
  activePro: null as Pro | null,
  activeProSwing: null as ProSwing | null,
  setActivePro: vi.fn(),
  setActiveProSwing: vi.fn(),
}

let mockPoseState = {
  shotType: null as string | null,
}

vi.mock('@/store', () => ({
  useComparisonStore: vi.fn(() => mockComparisonState),
  usePoseStore: vi.fn((selector?: (state: typeof mockPoseState) => unknown) => {
    if (typeof selector === 'function') return selector(mockPoseState)
    return mockPoseState
  }),
}))

function makeProSwing(id: string, shotType: ProSwing['shot_type']): ProSwing {
  return {
    id,
    pro_id: 'pro-1',
    shot_type: shotType,
    video_url: `https://example.com/${id}.mp4`,
    thumbnail_url: null,
    keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
    fps: 30,
    frame_count: null,
    duration_ms: null,
    phase_labels: {},
    metadata: {},
    created_at: '2025-01-01',
  }
}

function makePro(id: string, name: string, nationality: string, swings: ProSwing[]): Pro & { pro_swings: ProSwing[] } {
  return {
    id,
    name,
    nationality,
    ranking: null,
    bio: null,
    profile_image_url: null,
    created_at: '2025-01-01',
    pro_swings: swings,
  }
}

describe('ProSelector', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.clearAllMocks()
    mockComparisonState = {
      activePro: null,
      activeProSwing: null,
      setActivePro: vi.fn(),
      setActiveProSwing: vi.fn(),
    }
    mockPoseState = {
      shotType: null,
    }
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it('shows loading skeleton initially', () => {
    // fetch never resolves => loading stays true
    vi.mocked(globalThis.fetch).mockReturnValue(new Promise(() => {}))

    const { container } = render(<ProSelector />)

    // Loading state has animate-pulse skeleton divs
    const pulseEl = container.querySelector('.animate-pulse')
    expect(pulseEl).not.toBeNull()
  })

  it('shows error message on fetch failure', async () => {
    vi.mocked(globalThis.fetch).mockRejectedValue(new Error('Network error'))

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText(/Failed to load players/)).toBeInTheDocument()
    })
  })

  it('shows "No pro players" when fetch returns empty array', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => [],
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText(/No pro players in database yet/)).toBeInTheDocument()
    })
  })

  it('renders pro list after successful fetch', async () => {
    const pros = [
      makePro('p1', 'Roger Federer', 'SUI', [makeProSwing('s1', 'forehand')]),
      makePro('p2', 'Rafael Nadal', 'ESP', [makeProSwing('s2', 'backhand')]),
    ]

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => pros,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
      expect(screen.getByText('Rafael Nadal')).toBeInTheDocument()
    })
  })

  it('renders shot type filter buttons', async () => {
    const pros = [
      makePro('p1', 'Roger Federer', 'SUI', [makeProSwing('s1', 'forehand')]),
    ]

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => pros,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('all')).toBeInTheDocument()
      expect(screen.getByText('forehand')).toBeInTheDocument()
      expect(screen.getByText('backhand')).toBeInTheDocument()
      expect(screen.getByText('serve')).toBeInTheDocument()
      expect(screen.getByText('volley')).toBeInTheDocument()
    })
  })

  it('shot type filter hides pros that have no matching swings', async () => {
    const pros = [
      makePro('p1', 'Roger Federer', 'SUI', [makeProSwing('s1', 'forehand')]),
      makePro('p2', 'Rafael Nadal', 'ESP', [makeProSwing('s2', 'backhand')]),
    ]

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => pros,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
    })

    // Filter to backhand only
    fireEvent.click(screen.getByText('backhand'))

    // Federer should disappear (only has forehand), Nadal should remain
    await waitFor(() => {
      expect(screen.queryByText('Roger Federer')).not.toBeInTheDocument()
      expect(screen.getByText('Rafael Nadal')).toBeInTheDocument()
    })
  })

  it('clicking a pro expands swing list', async () => {
    const pros = [
      makePro('p1', 'Roger Federer', 'SUI', [
        makeProSwing('s1', 'forehand'),
        makeProSwing('s2', 'serve'),
      ]),
    ]

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => pros,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
    })

    // Before expanding, only the filter bar has "forehand" (1 instance)
    const beforeExpand = screen.getAllByText('forehand')
    expect(beforeExpand).toHaveLength(1)

    // Click pro to expand
    fireEvent.click(screen.getByText('Roger Federer'))

    // After expanding, the swing list also shows "forehand" (2 instances)
    await waitFor(() => {
      const forehandItems = screen.getAllByText('forehand')
      expect(forehandItems).toHaveLength(2)
    })
  })

  it('clicking a swing calls selectSwing which sets active pro and fetches swing', async () => {
    const swing = makeProSwing('s1', 'forehand')
    const pro = makePro('p1', 'Roger Federer', 'SUI', [swing])
    const pros = [pro]

    const fetchMock = vi.mocked(globalThis.fetch)

    // First call: /api/pros
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => pros,
    } as Response)

    // Second call: /api/pro-swings/s1 (when swing is clicked)
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...swing, keypoints_json: { fps_sampled: 30, frame_count: 1, frames: [] } }),
    } as Response)

    render(<ProSelector />)

    // Wait for pros to load
    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
    })

    // Expand the pro
    fireEvent.click(screen.getByText('Roger Federer'))

    // Wait for swing list
    await waitFor(() => {
      // filter bar "forehand" + swing list "forehand"
      const forehandItems = screen.getAllByText('forehand')
      expect(forehandItems.length).toBeGreaterThanOrEqual(2)
    })

    // Click the swing in the swing list (not the filter button)
    // The swing buttons are inside a border-t section
    const swingButtons = screen.getAllByText('forehand')
    // The last one should be the swing button (in the expanded list)
    fireEvent.click(swingButtons[swingButtons.length - 1])

    // setActivePro should be called with the pro
    expect(mockComparisonState.setActivePro).toHaveBeenCalledWith(pro)

    // After fetch resolves, setActiveProSwing should be called
    await waitFor(() => {
      expect(mockComparisonState.setActiveProSwing).toHaveBeenCalled()
    })
  })

  it('selected swing shows "Selected" badge', async () => {
    const swing = makeProSwing('s1', 'forehand')
    const pro = makePro('p1', 'Roger Federer', 'SUI', [swing])

    mockComparisonState.activeProSwing = swing

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => [pro],
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
    })

    // Expand
    fireEvent.click(screen.getByText('Roger Federer'))

    await waitFor(() => {
      expect(screen.getByText('Selected')).toBeInTheDocument()
    })
  })

  it('shows nationality and swing count for each pro', async () => {
    const pros = [
      makePro('p1', 'Roger Federer', 'SUI', [
        makeProSwing('s1', 'forehand'),
        makeProSwing('s2', 'serve'),
      ]),
    ]

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => pros,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText(/SUI/)).toBeInTheDocument()
      expect(screen.getByText(/2 swings/)).toBeInTheDocument()
    })
  })

  it('auto-selects first matching swing when userShotType is set', async () => {
    mockPoseState.shotType = 'forehand'

    const swing = makeProSwing('s1', 'forehand')
    const pro = makePro('p1', 'Roger Federer', 'SUI', [swing])

    const fetchMock = vi.mocked(globalThis.fetch)

    // First call: /api/pros
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => [pro],
    } as Response)

    // Second call: /api/pro-swings/s1 (auto-select fetch)
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => swing,
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(mockComparisonState.setActivePro).toHaveBeenCalledWith(pro)
    })
  })

  it('auto-select re-fires when userShotType changes', async () => {
    mockPoseState.shotType = 'forehand'

    const forehandSwing = makeProSwing('s1', 'forehand')
    const backhandSwing = makeProSwing('s2', 'backhand')
    const pro = makePro('p1', 'Roger Federer', 'SUI', [forehandSwing, backhandSwing])

    const fetchMock = vi.mocked(globalThis.fetch)

    // First call: /api/pros
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => [pro],
    } as Response)

    // Second call: auto-select forehand swing
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => forehandSwing,
    } as Response)

    const { rerender } = render(<ProSelector />)

    // Wait for auto-select to fire with forehand
    await waitFor(() => {
      expect(mockComparisonState.setActivePro).toHaveBeenCalledWith(pro)
    })

    // Now simulate userShotType change to backhand
    mockPoseState.shotType = 'backhand'

    // Add fetch mock for backhand swing
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => backhandSwing,
    } as Response)

    rerender(<ProSelector />)

    // The auto-select should re-fire because userShotType changed
    await waitFor(() => {
      expect(mockComparisonState.setActivePro).toHaveBeenCalledTimes(2)
    })
  })

  it('does not auto-select when activeProSwing already matches userShotType', async () => {
    mockPoseState.shotType = 'forehand'

    const forehandSwing = makeProSwing('s1', 'forehand')
    // Set activeProSwing to already match
    mockComparisonState.activeProSwing = forehandSwing

    const pro = makePro('p1', 'Roger Federer', 'SUI', [forehandSwing])

    vi.mocked(globalThis.fetch).mockResolvedValue({
      ok: true,
      json: async () => [pro],
    } as Response)

    render(<ProSelector />)

    await waitFor(() => {
      expect(screen.getByText('Roger Federer')).toBeInTheDocument()
    })

    // setActivePro should NOT have been called since the swing already matches
    expect(mockComparisonState.setActivePro).not.toHaveBeenCalled()
  })
})
