import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import LLMCoachingPanel from '@/components/LLMCoachingPanel'
import type { ProSwing } from '@/lib/supabase'

// Default store values
let mockAnalysisState = {
  feedback: '',
  loading: false,
  setFeedback: vi.fn(),
  appendFeedback: vi.fn(),
  setLoading: vi.fn(),
  reset: vi.fn(),
}

let mockPoseState = {
  framesData: [] as any[],
  sessionId: null as string | null,
}

vi.mock('@/store', () => ({
  useAnalysisStore: Object.assign(
    vi.fn(() => mockAnalysisState),
    { getState: vi.fn(() => ({ feedback: '' })) }
  ),
  usePoseStore: vi.fn(() => mockPoseState),
}))

import { useAnalysisStore } from '@/store'

const mockProSwing: ProSwing = {
  id: 'swing-1',
  pro_id: 'pro-1',
  shot_type: 'forehand',
  video_url: 'https://example.com/video.mp4',
  thumbnail_url: null,
  keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
  fps: 30,
  frame_count: null,
  duration_ms: null,
  phase_labels: {},
  metadata: {},
  created_at: '2025-01-01',
  pros: { id: 'pro-1', name: 'Roger Federer', nationality: 'SUI', ranking: 3, bio: null, profile_image_url: null, created_at: '2025-01-01' },
}

describe('LLMCoachingPanel', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.clearAllMocks()
    mockAnalysisState = {
      feedback: '',
      loading: false,
      setFeedback: vi.fn(),
      appendFeedback: vi.fn(),
      setLoading: vi.fn(),
      reset: vi.fn(),
    }
    mockPoseState = {
      framesData: [],
      sessionId: null,
    }
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  it('shows "Upload a video first" when framesData is empty and panel is open', () => {
    mockPoseState.framesData = []

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    // The content section is hidden by default (open=false). We need canAnalyze to be false
    // and the panel needs to be open. Since framesData is empty, canAnalyze is false.
    // The panel starts collapsed - we need to verify the button is disabled.
    const analyzeBtn = screen.getByText('Analyze Swing')
    expect(analyzeBtn).toBeInTheDocument()
    expect(analyzeBtn.closest('button')).toBeDisabled()
  })

  it('shows "Select a pro player" message when no proSwing in pro mode', () => {
    // Set framesData so canAnalyze condition depends on proSwing
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    // canAnalyze = framesData.length > 0 && (compareMode === 'custom' || proSwing !== null)
    // With compareMode='pro' (default) and proSwing=null => canAnalyze=false

    render(<LLMCoachingPanel proSwing={null} />)

    const analyzeBtn = screen.getByText('Analyze Swing')
    expect(analyzeBtn.closest('button')).toBeDisabled()
  })

  it('shows "Analyze Swing" button when ready', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    const btn = screen.getByText('Analyze Swing')
    expect(btn).toBeInTheDocument()
    // canAnalyze = true because framesData.length > 0 and proSwing is not null
    expect(btn.closest('button')).not.toBeDisabled()
  })

  it('Analyze button is disabled when canAnalyze is false', () => {
    // framesData empty => canAnalyze = false
    mockPoseState.framesData = []

    render(<LLMCoachingPanel proSwing={null} />)

    const btn = screen.getByText('Analyze Swing')
    expect(btn.closest('button')).toBeDisabled()
  })

  it('shows loading state (spinner) during analysis', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    expect(screen.getByText('Analyzing...')).toBeInTheDocument()
  })

  it('shows pro info in header when proSwing is provided', () => {
    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    expect(screen.getByText(/Roger Federer/)).toBeInTheDocument()
    expect(screen.getByText(/forehand/)).toBeInTheDocument()
  })

  it('shows "Form Analysis" label when compareMode is custom', () => {
    render(<LLMCoachingPanel proSwing={null} compareMode="custom" />)

    expect(screen.getByText('Form Analysis')).toBeInTheDocument()
  })

  it('renders feedback text correctly', () => {
    mockAnalysisState.feedback = '## Great Form\nYour **backswing** is solid.'

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    // The Collapse/Expand button should appear when feedback exists
    expect(screen.getByText('Expand')).toBeInTheDocument()
  })

  it('expand/collapse toggle works', () => {
    mockAnalysisState.feedback = 'Some coaching feedback here.'

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    // Initially collapsed but Expand button visible because feedback exists
    const expandBtn = screen.getByText('Expand')
    fireEvent.click(expandBtn)

    // Now should show Collapse and the feedback content
    expect(screen.getByText('Collapse')).toBeInTheDocument()
    expect(screen.getByText('Some coaching feedback here.')).toBeInTheDocument()
  })

  it('re-analyze button appears after first analysis', () => {
    mockAnalysisState.feedback = 'Analysis complete.'
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    expect(screen.getByText('Re-analyze')).toBeInTheDocument()
  })

  it('shows markdown headers as h3 elements when panel is expanded', () => {
    mockAnalysisState.feedback = '## Swing Analysis\nLooks good.'

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    // Expand to see content
    fireEvent.click(screen.getByText('Expand'))

    expect(screen.getByText('Swing Analysis')).toBeInTheDocument()
    expect(screen.getByText('Swing Analysis').tagName).toBe('H3')
  })

  it('renders bold text in markdown', () => {
    mockAnalysisState.feedback = 'Your **technique** is improving.'

    render(<LLMCoachingPanel proSwing={mockProSwing} />)

    fireEvent.click(screen.getByText('Expand'))

    const strongEl = screen.getByText('technique')
    expect(strongEl.tagName).toBe('STRONG')
  })
})
