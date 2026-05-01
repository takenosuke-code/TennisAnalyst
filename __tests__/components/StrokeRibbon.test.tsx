import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import StrokeRibbon from '@/components/StrokeRibbon'
import type {
  DetectedStroke,
  StrokeQualityResult,
  StrokeComparisonResult,
} from '@/lib/strokeAnalysis'

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function makeStroke(i: number): DetectedStroke {
  return {
    strokeId: `s-${i}`,
    startFrame: i * 30,
    endFrame: i * 30 + 20,
    peakFrame: i * 30 + 10,
    fps: 30,
  }
}

function makeQuality(
  i: number,
  score: number,
  opts: Partial<StrokeQualityResult> = {},
): StrokeQualityResult {
  return {
    strokeId: `s-${i}`,
    score,
    rejected: false,
    components: {
      peakWristSpeed: 1,
      kineticChainTimingError: 0.1,
      wristAngleVariance: 0.2,
    },
    ...opts,
  }
}

// requestAnimationFrame is needed for the keyboard-nav focus deferral.
// jsdom polyfills it, but make it synchronous so the focus assertion
// runs in the same tick.
beforeEach(() => {
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(
    (cb: FrameRequestCallback) => {
      cb(0)
      return 0
    },
  )
})

describe('StrokeRibbon', () => {
  // --- Empty state ---------------------------------------------------------

  it('renders the empty state when there are no strokes', () => {
    render(<StrokeRibbon strokes={[]} quality={[]} />)
    expect(screen.getByText(/No strokes detected/i)).toBeInTheDocument()
  })

  // --- One chip per stroke -------------------------------------------------

  it('renders one chip per stroke', () => {
    const strokes = [makeStroke(0), makeStroke(1), makeStroke(2)]
    const quality = [makeQuality(0, 0), makeQuality(1, 0), makeQuality(2, 0)]

    render(<StrokeRibbon strokes={strokes} quality={quality} />)

    const chips = screen.getAllByRole('button')
    expect(chips).toHaveLength(3)
    expect(screen.getByText('Stroke 1')).toBeInTheDocument()
    expect(screen.getByText('Stroke 2')).toBeInTheDocument()
    expect(screen.getByText('Stroke 3')).toBeInTheDocument()
  })

  // --- Color stripe maps to score ------------------------------------------

  it('color stripe maps correctly: worst chip clay, best chip green-3', () => {
    const strokes = [makeStroke(0), makeStroke(1), makeStroke(2)]
    const quality = [
      makeQuality(0, -1.5), // worst → clay
      makeQuality(1, 0.0), // neutral → cream-soft
      makeQuality(2, 1.5), // best → green-3
    ]

    const { container } = render(
      <StrokeRibbon strokes={strokes} quality={quality} />,
    )

    const stripes = container.querySelectorAll('[data-stripe-bucket]')
    expect(stripes).toHaveLength(3)
    expect(stripes[0].getAttribute('data-stripe-bucket')).toBe('worst')
    expect(stripes[0].classList.contains('bg-clay')).toBe(true)
    expect(stripes[1].getAttribute('data-stripe-bucket')).toBe('neutral')
    expect(stripes[2].getAttribute('data-stripe-bucket')).toBe('best')
    expect(stripes[2].classList.contains('bg-green-3')).toBe(true)
  })

  // --- Best / worst badges -------------------------------------------------

  it('shows BEST and WORST badges when comparison provides them', () => {
    const strokes = [makeStroke(0), makeStroke(1), makeStroke(2)]
    const quality = [
      makeQuality(0, -1.5),
      makeQuality(1, 0),
      makeQuality(2, 1.5),
    ]
    const comparison: StrokeComparisonResult = {
      best: { strokeId: 's-2', reasoning: '', citations: [] },
      worst: { strokeId: 's-0', reasoning: '', citations: [] },
      isConsistent: false,
    }

    render(
      <StrokeRibbon strokes={strokes} quality={quality} comparison={comparison} />,
    )

    expect(screen.getByText(/^Best$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Worst$/i)).toBeInTheDocument()
  })

  // --- Consistent-session banner ------------------------------------------

  it('renders consistent-session banner and hides best/worst when isConsistent=true', () => {
    const strokes = [makeStroke(0), makeStroke(1)]
    const quality = [makeQuality(0, 0.2), makeQuality(1, -0.1)]
    const comparison: StrokeComparisonResult = {
      best: { strokeId: 's-1', reasoning: '', citations: [] },
      worst: { strokeId: 's-0', reasoning: '', citations: [] },
      isConsistent: true,
      consistentCue: 'Hold the same hip rotation through every contact.',
    }

    render(
      <StrokeRibbon strokes={strokes} quality={quality} comparison={comparison} />,
    )

    expect(screen.getByText(/Consistent session/i)).toBeInTheDocument()
    expect(
      screen.getByText(/Hold the same hip rotation through every contact/),
    ).toBeInTheDocument()
    // No best/worst badges in the consistent state.
    expect(screen.queryByText(/^Best$/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^Worst$/i)).not.toBeInTheDocument()
  })

  // --- Rejected chips ------------------------------------------------------

  it('rejected chip is muted, struck through, and exposes reason via aria-describedby', () => {
    const strokes = [makeStroke(0), makeStroke(1)]
    const quality = [
      makeQuality(0, NaN, { rejected: true, rejectReason: 'low_visibility' }),
      makeQuality(1, 0.4),
    ]

    const { container } = render(
      <StrokeRibbon strokes={strokes} quality={quality} />,
    )

    // The rejected chip is opacity-50 (muted).
    const rejectedChip = screen.getByLabelText(/Stroke 1, score/i)
    expect(rejectedChip.className).toMatch(/opacity-50/)

    // The reject reason is referenced by aria-describedby and present
    // in DOM (not gated on hover).
    const describedBy = rejectedChip.getAttribute('aria-describedby')
    expect(describedBy).toBeTruthy()
    const reasonNode = container.querySelector(`#${CSS.escape(describedBy!)}`)
    expect(reasonNode).toBeTruthy()
    expect(reasonNode!.textContent).toMatch(/Low visibility/i)

    // The label "Stroke 1" inside the rejected chip is struck through.
    const label = rejectedChip.querySelector('.line-through')
    expect(label).toBeTruthy()
  })

  it('does not run a NaN-scored rejected chip through the worst/best thresholds', () => {
    const strokes = [makeStroke(0)]
    const quality = [
      makeQuality(0, NaN, { rejected: true, rejectReason: 'too_short' }),
    ]

    const { container } = render(
      <StrokeRibbon strokes={strokes} quality={quality} />,
    )

    const stripe = container.querySelector('[data-stripe-bucket]')
    expect(stripe?.getAttribute('data-stripe-bucket')).toBe('rejected')
  })

  // --- Tap behaviour -------------------------------------------------------

  it('clicking a chip fires onStrokeSelect with the strokeId', () => {
    const onSelect = vi.fn()
    const strokes = [makeStroke(0), makeStroke(1)]
    const quality = [makeQuality(0, 0), makeQuality(1, 0.5)]

    render(
      <StrokeRibbon
        strokes={strokes}
        quality={quality}
        onStrokeSelect={onSelect}
      />,
    )

    fireEvent.click(screen.getByLabelText(/Stroke 2,/i))
    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith('s-1')
  })

  // --- selectedStrokeId visual state --------------------------------------

  it('selectedStrokeId drives the selected visual state', () => {
    const strokes = [makeStroke(0), makeStroke(1)]
    const quality = [makeQuality(0, 0), makeQuality(1, 0.5)]

    render(
      <StrokeRibbon
        strokes={strokes}
        quality={quality}
        selectedStrokeId="s-1"
      />,
    )

    const chips = screen.getAllByRole('button')
    expect(chips[0].getAttribute('data-selected')).toBe('false')
    expect(chips[1].getAttribute('data-selected')).toBe('true')
    expect(chips[1].className).toMatch(/border-clay/)
    // Selected chip is in the tab order; the other is not (roving tabindex).
    expect(chips[1].getAttribute('tabindex')).toBe('0')
    expect(chips[0].getAttribute('tabindex')).toBe('-1')
  })

  // --- Keyboard navigation -------------------------------------------------

  it('arrow-key navigation moves selection through chips', () => {
    const onSelect = vi.fn()
    const strokes = [makeStroke(0), makeStroke(1), makeStroke(2)]
    const quality = [
      makeQuality(0, 0),
      makeQuality(1, 0),
      makeQuality(2, 0),
    ]

    render(
      <StrokeRibbon
        strokes={strokes}
        quality={quality}
        onStrokeSelect={onSelect}
        selectedStrokeId="s-0"
      />,
    )

    const toolbar = screen.getByRole('toolbar', { name: /Stroke timeline/i })

    // ArrowRight: 0 -> 1
    fireEvent.keyDown(toolbar, { key: 'ArrowRight' })
    expect(onSelect).toHaveBeenLastCalledWith('s-1')

    // ArrowLeft from index 0 stays at 0 — no callback fires (no change).
    onSelect.mockClear()
    fireEvent.keyDown(toolbar, { key: 'ArrowLeft' })
    // Note: parent hasn't updated selectedStrokeId, so internally
    // selectedIndex is still 0. ArrowLeft stays. No callback.
    expect(onSelect).not.toHaveBeenCalled()

    // End jumps to last.
    fireEvent.keyDown(toolbar, { key: 'End' })
    expect(onSelect).toHaveBeenLastCalledWith('s-2')
  })

  // --- aria container label ------------------------------------------------

  it('exposes the toolbar with aria-label="Stroke timeline"', () => {
    const strokes = [makeStroke(0)]
    const quality = [makeQuality(0, 0)]
    render(<StrokeRibbon strokes={strokes} quality={quality} />)
    expect(
      screen.getByRole('toolbar', { name: /Stroke timeline/i }),
    ).toBeInTheDocument()
  })

  // --- score pill formatting ----------------------------------------------

  it('formats positive scores with a leading + and negative with a typographic minus', () => {
    const strokes = [makeStroke(0), makeStroke(1)]
    const quality = [makeQuality(0, 0.83), makeQuality(1, -1.21)]

    render(<StrokeRibbon strokes={strokes} quality={quality} />)

    expect(screen.getByText('+0.8')).toBeInTheDocument()
    // U+2212 minus, NOT a hyphen-minus.
    expect(screen.getByText('\u22121.2')).toBeInTheDocument()
  })
})
