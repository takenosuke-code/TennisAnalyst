import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import LLMCoachingPanel from '@/components/LLMCoachingPanel'

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

  it('disables Analyze Swing when no frames are loaded', () => {
    mockPoseState.framesData = []

    render(<LLMCoachingPanel />)

    const analyzeBtn = screen.getByText('Analyze Swing')
    expect(analyzeBtn).toBeInTheDocument()
    expect(analyzeBtn.closest('button')).toBeDisabled()
  })

  it('enables Analyze Swing once frames are present (solo mode)', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    const btn = screen.getByText('Analyze Swing')
    expect(btn.closest('button')).not.toBeDisabled()
  })

  it('shows loading state (spinner) during analysis', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Analyzing...')).toBeInTheDocument()
  })

  it('shows "Form Analysis" label when compareMode is custom', () => {
    render(<LLMCoachingPanel compareMode="custom" />)

    expect(screen.getByText('Form Analysis')).toBeInTheDocument()
  })

  it('shows baseline label when compareMode is baseline', () => {
    render(<LLMCoachingPanel compareMode="baseline" baselineLabel="May 3 rally" />)

    expect(screen.getByText(/May 3 rally/)).toBeInTheDocument()
  })

  it('renders feedback text correctly', () => {
    mockAnalysisState.feedback = '## Great Form\nYour **backswing** is solid.'

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Expand')).toBeInTheDocument()
  })

  it('expand/collapse toggle works', () => {
    mockAnalysisState.feedback = 'Some coaching feedback here.'

    render(<LLMCoachingPanel />)

    const expandBtn = screen.getByText('Expand')
    fireEvent.click(expandBtn)

    expect(screen.getByText('Collapse')).toBeInTheDocument()
    expect(screen.getByText('Some coaching feedback here.')).toBeInTheDocument()
  })

  it('re-analyze button appears after first analysis', () => {
    mockAnalysisState.feedback = 'Analysis complete.'
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Re-analyze')).toBeInTheDocument()
  })

  it('shows markdown headers as h3 elements when panel is expanded', () => {
    mockAnalysisState.feedback = '## Swing Analysis\nLooks good.'

    render(<LLMCoachingPanel />)

    fireEvent.click(screen.getByText('Expand'))

    expect(screen.getByText('Swing Analysis')).toBeInTheDocument()
    expect(screen.getByText('Swing Analysis').tagName).toBe('H3')
  })

  it('renders bold text in markdown', () => {
    mockAnalysisState.feedback = 'Your **technique** is improving.'

    render(<LLMCoachingPanel />)

    fireEvent.click(screen.getByText('Expand'))

    const strongEl = screen.getByText('technique')
    expect(strongEl.tagName).toBe('STRONG')
  })
})
