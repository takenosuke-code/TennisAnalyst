import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SwingSelector from '@/components/SwingSelector'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'
import { makeFrame } from '@/__tests__/helpers'

// Mock detectSwings from jointAngles
vi.mock('@/lib/jointAngles', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/jointAngles')>()
  return {
    ...actual,
    detectSwings: vi.fn(() => []),
  }
})

import { detectSwings } from '@/lib/jointAngles'
const mockedDetectSwings = vi.mocked(detectSwings)

function makeSegment(index: number, startMs: number, endMs: number, frameCount: number): SwingSegment {
  const frames: PoseFrame[] = Array.from({ length: frameCount }, (_, i) =>
    makeFrame(i, startMs + i * ((endMs - startMs) / Math.max(frameCount - 1, 1)), [])
  )
  return {
    index,
    startFrame: 0,
    endFrame: frameCount - 1,
    startMs,
    endMs,
    peakFrame: Math.floor(frameCount / 2),
    frames,
  }
}

describe('SwingSelector', () => {
  const mockOnSelect = vi.fn()
  const dummyFrames: PoseFrame[] = [makeFrame(0, 0, []), makeFrame(1, 100, [])]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns null when detectSwings returns only 1 segment', () => {
    mockedDetectSwings.mockReturnValue([makeSegment(1, 0, 1000, 10)])
    const { container } = render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={null} onSelect={mockOnSelect} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('returns null when detectSwings returns 0 segments', () => {
    mockedDetectSwings.mockReturnValue([])
    const { container } = render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={null} onSelect={mockOnSelect} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders buttons for each swing segment when multiple exist', () => {
    const segments = [
      makeSegment(1, 0, 2000, 20),
      makeSegment(2, 3000, 5000, 25),
      makeSegment(3, 6000, 8000, 30),
    ]
    mockedDetectSwings.mockReturnValue(segments)

    render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={null} onSelect={mockOnSelect} />
    )

    expect(screen.getByText('Swing 1')).toBeInTheDocument()
    expect(screen.getByText('Swing 2')).toBeInTheDocument()
    expect(screen.getByText('Swing 3')).toBeInTheDocument()
    expect(screen.getByText('3 swings detected')).toBeInTheDocument()
  })

  it('shows correct metadata (duration, start time, frame count)', () => {
    const segments = [
      makeSegment(1, 1000, 3500, 20),
      makeSegment(2, 5000, 7000, 15),
    ]
    mockedDetectSwings.mockReturnValue(segments)

    render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={null} onSelect={mockOnSelect} />
    )

    // Swing 1: startSec = 1.0s, duration = 2.5s, 20 frames
    // The metadata text uses middot entity which renders as a visible character
    // We need to check for the individual values
    expect(screen.getByText(/1\.0s/)).toBeInTheDocument()
    expect(screen.getByText(/2\.5s/)).toBeInTheDocument()
    expect(screen.getByText(/20f/)).toBeInTheDocument()

    // Swing 2: startSec = 5.0s, duration = 2.0s, 15 frames
    expect(screen.getByText(/5\.0s/)).toBeInTheDocument()
    expect(screen.getByText(/2\.0s/)).toBeInTheDocument()
    expect(screen.getByText(/15f/)).toBeInTheDocument()
  })

  it('clicking a swing button calls onSelect with correct segment', () => {
    const segments = [
      makeSegment(1, 0, 2000, 20),
      makeSegment(2, 3000, 5000, 25),
    ]
    mockedDetectSwings.mockReturnValue(segments)

    render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={null} onSelect={mockOnSelect} />
    )

    fireEvent.click(screen.getByText('Swing 2'))
    expect(mockOnSelect).toHaveBeenCalledWith(segments[1])
  })

  it('selected swing has active styling', () => {
    const segments = [
      makeSegment(1, 0, 2000, 20),
      makeSegment(2, 3000, 5000, 25),
    ]
    mockedDetectSwings.mockReturnValue(segments)

    render(
      <SwingSelector allFrames={dummyFrames} selectedIndex={2} onSelect={mockOnSelect} />
    )

    const swing2Btn = screen.getByText('Swing 2').closest('button')!
    const swing1Btn = screen.getByText('Swing 1').closest('button')!

    // Selected: 'bg-emerald-500 border-emerald-400 text-white'
    expect(swing2Btn.className).toContain('bg-emerald-500')
    // Unselected: 'bg-white/5 border-white/10 text-white/70'
    expect(swing1Btn.className).toContain('bg-white/5')
  })
})
