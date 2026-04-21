import { describe, it, expect } from 'vitest'
import {
  ADVANCED_BASELINE_TEMPLATE,
  buildCoachingToolInput,
  countTipsInMarkdown,
  extractResponseMetrics,
  renderCoachingToolInputToMarkdown,
  type CoachingToolInput,
} from '@/lib/profile'

describe('extractResponseMetrics', () => {
  it('flags used_baseline_template for advanced tier when tool cues=[] and closing matches template', () => {
    const toolInput: CoachingToolInput = {
      strengths: [{ text: 'Clean kinetic chain across all takes.' }],
      cues: [],
      closing: ADVANCED_BASELINE_TEMPLATE,
    }
    const r = extractResponseMetrics({
      tier: 'advanced',
      toolInput,
      markdownText: ADVANCED_BASELINE_TEMPLATE,
      outputTokens: 42,
    })
    expect(r.response_tip_count).toBe(0)
    expect(r.used_baseline_template).toBe(true)
    expect(r.response_token_count).toBe(42)
    expect(r.response_char_count).toBe(ADVANCED_BASELINE_TEMPLATE.length)
  })

  it('does NOT flag used_baseline_template when advanced has one cue', () => {
    const toolInput: CoachingToolInput = {
      strengths: [{ text: 'Contact point is consistent.' }],
      cues: [{ title: 'Micro-refinement', body: 'Feel the racket swing out just a hair earlier.' }],
      closing: 'Groove it.',
    }
    const r = extractResponseMetrics({
      tier: 'advanced',
      toolInput,
      markdownText: 'rendered',
      outputTokens: 100,
    })
    expect(r.response_tip_count).toBe(1)
    expect(r.used_baseline_template).toBe(false)
  })

  it('counts beginner with 3 cues as tip_count=3', () => {
    const toolInput: CoachingToolInput = {
      strengths: [{ text: 'Nice relaxed grip.' }],
      cues: [
        { title: 'Push through', body: 'Drive the ball toward the far fence.' },
        { title: 'Finish high', body: 'Racket up by your ear.' },
        { title: 'Feel low-to-high', body: 'Let the racket swing up through contact.' },
      ],
      closing: 'Great work.',
    }
    const r = extractResponseMetrics({
      tier: 'beginner',
      toolInput,
      markdownText: 'rendered',
      outputTokens: 200,
    })
    expect(r.response_tip_count).toBe(3)
    expect(r.used_baseline_template).toBe(false)
  })

  it('falls back to counting tips in markdown when toolInput is null', () => {
    const md = [
      '## Your Coaching Cues',
      '',
      '**1. Foo**',
      'Body one.',
      '',
      '**2. Bar**',
      'Body two.',
      '',
      '**3. Baz**',
      'Body three.',
    ].join('\n')
    const r = extractResponseMetrics({
      tier: 'intermediate',
      toolInput: null,
      markdownText: md,
      outputTokens: 150,
    })
    expect(r.response_tip_count).toBe(3)
    expect(r.response_char_count).toBe(md.length)
    expect(r.used_baseline_template).toBe(false)
  })

  it('returns tip_count=null and char_count=0 for empty markdown with null toolInput', () => {
    const r = extractResponseMetrics({
      tier: 'intermediate',
      toolInput: null,
      markdownText: '',
      outputTokens: null,
    })
    expect(r.response_tip_count).toBeNull()
    expect(r.response_char_count).toBe(0)
    expect(r.used_baseline_template).toBe(false)
    expect(r.response_token_count).toBeNull()
  })

  it('flags used_baseline_template in the advanced fallback path when stripped markdown matches template', () => {
    const r = extractResponseMetrics({
      tier: 'advanced',
      toolInput: null,
      markdownText: `  ${ADVANCED_BASELINE_TEMPLATE}  `,
      outputTokens: 12,
    })
    expect(r.used_baseline_template).toBe(true)
  })

  it('does NOT flag used_baseline_template for advanced fallback when markdown is empty', () => {
    const r = extractResponseMetrics({
      tier: 'advanced',
      toolInput: null,
      markdownText: '',
      outputTokens: null,
    })
    expect(r.used_baseline_template).toBe(false)
  })
})

describe('countTipsInMarkdown', () => {
  it('counts a standard 3-tip block', () => {
    const md = '**1. Alpha**\nbody\n\n**2. Bravo**\nbody\n\n**3. Charlie**\nbody'
    expect(countTipsInMarkdown(md)).toBe(3)
  })

  it('returns 0 for empty input', () => {
    expect(countTipsInMarkdown('')).toBe(0)
  })

  it('collapses a skipped-sequence "**1. Foo** **4. Bar**" to 1', () => {
    const md = '**1. Foo**\n\n**4. Bar**'
    expect(countTipsInMarkdown(md)).toBe(1)
  })

  it('handles leading `N.` at line start without bold markers', () => {
    const md = '1. First\n2. Second'
    expect(countTipsInMarkdown(md)).toBe(2)
  })
})

describe('renderCoachingToolInputToMarkdown', () => {
  it('renders the full structured shape with headings + numbered bold cues', () => {
    const input: CoachingToolInput = {
      strengths: [{ text: 'Strong rotation' }, { text: 'Good racket prep' }],
      cues: [
        { title: 'Alpha', body: 'alpha body' },
        { title: 'Bravo', body: 'bravo body' },
      ],
      closing: 'Take it to the court.',
    }
    const md = renderCoachingToolInputToMarkdown(input, 'intermediate')
    expect(md).toContain("## What You're Doing Well")
    expect(md).toContain('Strong rotation')
    expect(md).toContain('Good racket prep')
    expect(md).toContain('## Your Coaching Cues')
    expect(md).toContain('**1. Alpha**')
    expect(md).toContain('alpha body')
    expect(md).toContain('**2. Bravo**')
    expect(md).toContain('bravo body')
    expect(md).toContain('## Practice Focus')
    expect(md).toContain('Take it to the court.')
  })

  it('emits ONLY the template (no headings) for advanced cues=[] with template closing', () => {
    const input: CoachingToolInput = {
      strengths: [{ text: 'Mechanics are refined.' }],
      cues: [],
      closing: ADVANCED_BASELINE_TEMPLATE,
    }
    const md = renderCoachingToolInputToMarkdown(input, 'advanced')
    expect(md).toBe(ADVANCED_BASELINE_TEMPLATE)
    expect(md).not.toContain('##')
    expect(md).not.toContain("What You're Doing Well")
  })
})

describe('buildCoachingToolInput', () => {
  it('returns a parsed object for valid input (trimmed strings)', () => {
    const raw = {
      strengths: [{ text: '  Good shoulder turn.  ' }],
      cues: [
        { title: ' Alpha ', body: ' alpha body ' },
        { title: 'Bravo', body: 'bravo body' },
      ],
      closing: '  Practice focus.  ',
    }
    const parsed = buildCoachingToolInput(raw, 'intermediate')
    expect(parsed).not.toBeNull()
    expect(parsed?.strengths[0].text).toBe('Good shoulder turn.')
    expect(parsed?.cues[0].title).toBe('Alpha')
    expect(parsed?.cues[0].body).toBe('alpha body')
    expect(parsed?.closing).toBe('Practice focus.')
  })

  it('returns null when cues field is missing', () => {
    const raw = {
      strengths: [{ text: 'Great work.' }],
      closing: 'Focus.',
    }
    expect(buildCoachingToolInput(raw, 'intermediate')).toBeNull()
  })

  it('returns null for beginner with 4 cues (violates maxItems 3)', () => {
    const raw = {
      strengths: [{ text: 'Nice grip.' }],
      cues: [
        { title: 'A', body: 'a' },
        { title: 'B', body: 'b' },
        { title: 'C', body: 'c' },
        { title: 'D', body: 'd' },
      ],
      closing: 'Onward.',
    }
    expect(buildCoachingToolInput(raw, 'beginner')).toBeNull()
  })

  it('falls back to intermediate bounds when tier is null (accepts 2–3 cues)', () => {
    const two = {
      strengths: [{ text: 'Solid prep.' }],
      cues: [
        { title: 'A', body: 'a' },
        { title: 'B', body: 'b' },
      ],
      closing: 'Go.',
    }
    const three = {
      strengths: [{ text: 'Solid prep.' }],
      cues: [
        { title: 'A', body: 'a' },
        { title: 'B', body: 'b' },
        { title: 'C', body: 'c' },
      ],
      closing: 'Go.',
    }
    const one = {
      strengths: [{ text: 'Solid prep.' }],
      cues: [{ title: 'A', body: 'a' }],
      closing: 'Go.',
    }
    expect(buildCoachingToolInput(two, null)).not.toBeNull()
    expect(buildCoachingToolInput(three, null)).not.toBeNull()
    // Intermediate requires minCues=2; 1 should reject.
    expect(buildCoachingToolInput(one, null)).toBeNull()
  })

  it('returns null when rawInput is not an object', () => {
    expect(buildCoachingToolInput(null, 'intermediate')).toBeNull()
    expect(buildCoachingToolInput('oops', 'intermediate')).toBeNull()
    expect(buildCoachingToolInput(42, 'intermediate')).toBeNull()
  })

  it('returns null when strengths is empty', () => {
    const raw = {
      strengths: [],
      cues: [
        { title: 'A', body: 'a' },
        { title: 'B', body: 'b' },
      ],
      closing: 'Go.',
    }
    expect(buildCoachingToolInput(raw, 'intermediate')).toBeNull()
  })
})
