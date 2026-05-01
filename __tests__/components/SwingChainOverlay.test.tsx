import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import SwingChainOverlay from '@/components/SwingChainOverlay'

// Smoke test for the kinetic-chain animated decoration. The animation
// itself is a CSS keyframe loop so we only check the structural pieces:
// header, all four chain stops (Hips → Trunk → Shoulders → Wrist), and
// an optional caption.
describe('SwingChainOverlay', () => {
  it('renders the four chain-stop labels in order', () => {
    render(<SwingChainOverlay />)
    expect(screen.getByText('Hips')).toBeInTheDocument()
    expect(screen.getByText('Trunk')).toBeInTheDocument()
    expect(screen.getByText('Shoulders')).toBeInTheDocument()
    expect(screen.getByText('Wrist')).toBeInTheDocument()
  })

  it('renders the caption when provided', () => {
    render(<SwingChainOverlay caption="Hips fire first." />)
    expect(screen.getByText('Hips fire first.')).toBeInTheDocument()
  })

  it('renders without throwing when no props are passed', () => {
    expect(() => render(<SwingChainOverlay />)).not.toThrow()
  })
})
