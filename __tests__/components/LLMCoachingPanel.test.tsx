import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
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
  framesData: [] as Array<Record<string, unknown>>,
  sessionId: null as string | null,
}

vi.mock('@/store', () => ({
  useAnalysisStore: Object.assign(
    vi.fn(() => mockAnalysisState),
    { getState: vi.fn(() => ({ feedback: mockAnalysisState.feedback })) }
  ),
  usePoseStore: vi.fn(() => mockPoseState),
}))

// useUser is mocked per-test via mockUseUser.mockReturnValue.
const mockUseUser = vi.fn(() => ({ user: { id: 'u1' }, loading: false }) as { user: unknown; loading: boolean })
vi.mock('@/hooks/useUser', () => ({
  useUser: () => mockUseUser(),
}))

// Helper: build a fetch Response whose body streams the given chunks and
// whose headers include the given event id. Mirrors what the real
// /api/analyze route emits.
function streamingResponse(chunks: string[], headers: Record<string, string> = {}) {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers })
}

const FIXTURE_THREE_SECTIONS = [
  '## Primary cue',
  'Stay low through contact and let the racket finish over your shoulder.',
  '',
  '## Other things I noticed',
  '- Your hip rotation kicked in slightly later than usual.',
  '- Front foot loaded a touch less.',
  '',
  '## Show your work',
  '- **Right elbow at contact**: 132° on baseline, 112° today (drifted 20°)',
  '- **Right knee at loading**: 149° on baseline, 165° today (drifted 16°)',
  '',
].join('\n')

describe('LLMCoachingPanel', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.clearAllMocks()
    mockAnalysisState = {
      feedback: '',
      loading: false,
      setFeedback: vi.fn((v: string) => {
        mockAnalysisState.feedback = v
      }),
      // Mutate shared state so a re-render (triggered by any subsequent
      // setState in the component) picks up the streamed feedback text.
      appendFeedback: vi.fn((v: string) => {
        mockAnalysisState.feedback += v
      }),
      setLoading: vi.fn(),
      reset: vi.fn(() => {
        mockAnalysisState.feedback = ''
      }),
    }
    mockPoseState = {
      framesData: [],
      sessionId: null,
    }
    mockUseUser.mockReturnValue({ user: { id: 'u1' }, loading: false })
    globalThis.fetch = vi.fn()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  // --- Initial / button state ---------------------------------------------

  it('disables Analyze Swing when no frames are loaded', () => {
    mockPoseState.framesData = []

    render(<LLMCoachingPanel />)

    const analyzeBtn = screen.getByText('Analyze Swing')
    expect(analyzeBtn.closest('button')).toBeDisabled()
  })

  it('enables Analyze Swing once frames are present (solo mode)', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    const btn = screen.getByText('Analyze Swing')
    expect(btn.closest('button')).not.toBeDisabled()
  })

  it('shows loading label while analysis runs', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel />)

    expect(screen.getByText(/Analyzing/i)).toBeInTheDocument()
  })

  it('shows "Form Analysis" label when compareMode is custom', () => {
    render(<LLMCoachingPanel compareMode="custom" />)

    expect(screen.getByText('Form Analysis')).toBeInTheDocument()
  })

  it('shows baseline label when compareMode is baseline', () => {
    render(<LLMCoachingPanel compareMode="baseline" baselineLabel="May 3 rally" />)

    expect(screen.getByText(/May 3 rally/)).toBeInTheDocument()
  })

  it('re-analyze button appears after first analysis', () => {
    mockAnalysisState.feedback = '## Primary cue\nGreat shape today.'
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Re-analyze')).toBeInTheDocument()
  })

  // --- Three-section parser ------------------------------------------------

  it('renders all three sections from a complete fixture markdown string', () => {
    mockAnalysisState.feedback = FIXTURE_THREE_SECTIONS

    render(<LLMCoachingPanel />)

    // Primary cue
    expect(screen.getByText('Primary cue')).toBeInTheDocument()
    expect(
      screen.getByText(/Stay low through contact/)
    ).toBeInTheDocument()

    // Other observations
    expect(screen.getByText('Other things I noticed')).toBeInTheDocument()
    expect(
      screen.getByText(/Your hip rotation kicked in slightly later/)
    ).toBeInTheDocument()

    // Show your work disclosure
    const showWork = screen.getByText('Show your work')
    expect(showWork).toBeInTheDocument()
    // The disclosure should be a <details>/<summary>; collapsed by default.
    const details = showWork.closest('details')
    expect(details).toBeTruthy()
    expect(details?.open).toBe(false)
  })

  it('renders bold inline markdown inside Show your work', () => {
    mockAnalysisState.feedback = FIXTURE_THREE_SECTIONS

    render(<LLMCoachingPanel />)

    // The right-elbow label is bold in the fixture.
    const strong = screen.getByText('Right elbow at contact')
    expect(strong.tagName).toBe('STRONG')
  })

  it('progressive render: shows Primary cue before later sections arrive', () => {
    // Only the first section is in the buffer so far.
    mockAnalysisState.feedback = '## Primary cue\nKeep your head still through contact.'

    render(<LLMCoachingPanel />)

    expect(screen.getByText('Primary cue')).toBeInTheDocument()
    expect(screen.queryByText('Other things I noticed')).not.toBeInTheDocument()
    expect(screen.queryByText('Show your work')).not.toBeInTheDocument()
  })

  // --- Empty-state branch --------------------------------------------------

  it('renders the empty-state body when X-Analyze-Empty-State header is true', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse(
        [
          "We couldn't read your swing clearly enough to give specific coaching.\nTry shooting from the side at chest height, with the player filling the frame.",
        ],
        { 'X-Analyze-Empty-State': 'true' }
      )
    )

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })

    expect(
      screen.getByText(/We couldn.t read your swing clearly enough/)
    ).toBeInTheDocument()
    // No Primary-cue heading, no disclosure in empty state.
    expect(screen.queryByText('Primary cue')).not.toBeInTheDocument()
    expect(screen.queryByText('Show your work')).not.toBeInTheDocument()
  })

  // --- Thumbs feedback strip ----------------------------------------------

  it('does not render thumbs strip while loading (even with feedback)', () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]
    mockAnalysisState.feedback = '## Primary cue\npartial...'
    mockAnalysisState.loading = true

    render(<LLMCoachingPanel />)

    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('does not render thumbs strip when feedback is empty', () => {
    mockAnalysisState.feedback = ''
    mockAnalysisState.loading = false

    render(<LLMCoachingPanel />)

    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('does not render thumbs strip when user is signed out', async () => {
    mockUseUser.mockReturnValue({ user: null, loading: false })
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-1' })
    )

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })

    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })

  it('clicking "Spot on" PATCHes the feedback endpoint with correct body', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-42' })
    )
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })

    expect(screen.getByText('Was this coaching right for you?')).toBeInTheDocument()

    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-42/feedback',
        expect.objectContaining({
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ correction: 'correct' }),
        })
      )
    })
  })

  it('clicking "Too advanced" sends correction=too_hard', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-hard' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('⬇️ Too advanced'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-hard/feedback',
        expect.objectContaining({
          body: JSON.stringify({ correction: 'too_hard' }),
        })
      )
    })
  })

  it('clicking "Too simple" sends correction=too_easy', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-easy' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('⬆️ Too simple'))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/analysis-events/evt-easy/feedback',
        expect.objectContaining({
          body: JSON.stringify({ correction: 'too_easy' }),
        })
      )
    })
  })

  it('collapses to thank-you state after a successful PATCH', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-ok' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(screen.getByText('Thanks — logged.')).toBeInTheDocument()
    })
    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
    expect(screen.queryByText('👍 Spot on')).not.toBeInTheDocument()
  })

  it('shows error message on PATCH failure and keeps buttons enabled', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-err' })
    )
    fetchMock.mockResolvedValueOnce(new Response('nope', { status: 500 }))

    render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })

    await waitFor(() => {
      expect(screen.getByText(/Couldn.t save/)).toBeInTheDocument()
    })

    const spotOn = screen.getByText('👍 Spot on').closest('button')!
    expect(spotOn).not.toBeDisabled()
    const tooAdv = screen.getByText('⬇️ Too advanced').closest('button')!
    expect(tooAdv).not.toBeDisabled()
  })

  it('resets feedbackState when feedback text clears (new analysis cycle)', async () => {
    mockPoseState.framesData = [{ frame_index: 0, timestamp_ms: 0, landmarks: [], joint_angles: {} }]

    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>
    fetchMock.mockResolvedValueOnce(
      streamingResponse([FIXTURE_THREE_SECTIONS], { 'X-Analysis-Event-Id': 'evt-reset' })
    )
    fetchMock.mockResolvedValueOnce(new Response('{"ok":true}', { status: 200 }))

    const { rerender } = render(<LLMCoachingPanel />)
    await act(async () => {
      fireEvent.click(screen.getByText('Analyze Swing'))
    })
    await act(async () => {
      fireEvent.click(screen.getByText('👍 Spot on'))
    })
    await waitFor(() => expect(screen.getByText('Thanks — logged.')).toBeInTheDocument())

    // Simulate a new analysis starting: feedback wiped, loading flips on.
    mockAnalysisState = {
      ...mockAnalysisState,
      feedback: '',
      loading: true,
    }
    rerender(<LLMCoachingPanel />)

    expect(screen.queryByText('Thanks — logged.')).not.toBeInTheDocument()
    expect(screen.queryByText('Was this coaching right for you?')).not.toBeInTheDocument()
  })
})
