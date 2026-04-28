import { describe, it, expect } from 'vitest'
import {
  buildLiveCoachingPrompt,
  LIVE_SYSTEM_PROMPT,
  type LivePromptSwing,
} from '@/lib/livePrompt'
import type { UserProfile } from '@/lib/profile'

const intermediateProfile: UserProfile = {
  skill_tier: 'intermediate',
  dominant_hand: 'right',
  backhand_style: 'two_handed',
  primary_goal: 'consistency',
  primary_goal_note: null,
  onboarded_at: '2026-04-20T00:00:00Z',
}

const advancedProfile: UserProfile = {
  skill_tier: 'advanced',
  dominant_hand: 'left',
  backhand_style: 'one_handed',
  primary_goal: 'other',
  primary_goal_note: 'reduce forehand pronation',
  onboarded_at: '2026-04-20T00:00:00Z',
}

const beginnerProfile: UserProfile = {
  skill_tier: 'beginner',
  dominant_hand: 'right',
  backhand_style: 'two_handed',
  primary_goal: 'fundamentals',
  primary_goal_note: null,
  onboarded_at: '2026-04-20T00:00:00Z',
}

function makeSwing(i: number): LivePromptSwing {
  return {
    angleSummary: `preparation: elbow_R=135° shoulder_R=100° ...\ncontact: elbow_R=160° ...`,
    startMs: i * 2000,
    endMs: i * 2000 + 600,
  }
}

describe('LIVE_SYSTEM_PROMPT', () => {
  it('enforces TTS-friendly brevity rules', () => {
    expect(LIVE_SYSTEM_PROMPT).toMatch(/ONE sentence/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/25 words/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/no markdown/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/no lists/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/external-focus/i)
  })

  it('authorizes silence as a first-class response', () => {
    // The previous prompt forbade silence; this version makes it primary.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/SILENCE IS A FIRST-CLASS RESPONSE/)
    // The four canonical affirmations called out in the spec must all be
    // present so the model has concrete patterns to mimic.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Clean — repeat that\./)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Trust that swing, do it again\./)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Same swing, again\./)
    // Empty response must be explicitly allowed.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/empty response/i)
  })

  it('enumerates the allowed observables (epistemic surface)', () => {
    expect(LIVE_SYSTEM_PROMPT).toMatch(/WHAT YOU CAN COMMENT ON/)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/angleSummary/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Peak frame timing/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/symmetry/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Hip rotation/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/trunk rotation/i)
  })

  it('forbids commenting on grip, footwork, target, opponent', () => {
    expect(LIVE_SYSTEM_PROMPT).toMatch(/WHAT YOU CANNOT COMMENT ON/)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/grip/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/pronation/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/footwork/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/target/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/opponent/i)
  })

  it('describes the soft "include one observed measurement" rule', () => {
    // Soft preference, NOT a requirement — the prompt must explicitly say
    // not every cue needs a number, otherwise the model produces gimmicky
    // number-stuffed output.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/EVIDENCE/)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/parens|parentheses/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/not required|not a requirement|do NOT force/i)
  })
})

describe('buildLiveCoachingPrompt', () => {
  it('returns both prompt text and a usedBaselineTemplate flag', () => {
    const result = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(typeof result.prompt).toBe('string')
    expect(typeof result.usedBaselineTemplate).toBe('boolean')
  })

  it('names the tier when profile is present', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2), makeSwing(3)],
    })
    expect(prompt).toMatch(/intermediate/i)
    expect(prompt).toMatch(/right-handed/i)
    expect(prompt).toMatch(/two-handed backhand/i)
    expect(prompt).toMatch(/consistency/i)
    expect(prompt).toMatch(/4 forehands in a row/i)
    expect(prompt).toMatch(/LAST 4 SWINGS/i)
    expect(prompt).toMatch(/SWING 1/i)
    expect(prompt).toMatch(/SWING 4/i)
  })

  it('handles the advanced tier and primary_goal_note', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: advancedProfile,
      skipped: false,
      shotType: 'backhand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(prompt).toMatch(/advanced/i)
    expect(prompt).toMatch(/left-handed/i)
    expect(prompt).toMatch(/one-handed backhand/i)
    expect(prompt).toMatch(/reduce forehand pronation/)
  })

  it('uses inferred-tier language for skipped users', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: null,
      skipped: true,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2)],
    })
    expect(prompt).toMatch(/skipped onboarding/i)
    expect(prompt).toMatch(/infer tier/i)
    expect(prompt).not.toMatch(/Player is/)
  })

  it('uses generic language when neither profile nor skipped flag is present', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: null,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(prompt).toMatch(/tier unknown/i)
    expect(prompt).toMatch(/broadly applicable/i)
  })

  it('includes the baseline block only when baselineSummary is provided', () => {
    const { prompt: withBaseline } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      baselineSummary: 'preparation: elbow_R=120° ...',
      baselineLabel: 'June 14 winner',
    })
    expect(withBaseline).toMatch(/BASELINE \(June 14 winner\)/)
    expect(withBaseline).toMatch(/elbow_R=120°/)

    const { prompt: withoutBaseline } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(withoutBaseline).not.toMatch(/BASELINE/)
  })

  it('asks for one cue OR silence at the end (non-advanced tiers)', () => {
    // Intermediate tier should NOT take the baseline-template path; the
    // closing directive must explicitly allow silence so the model knows
    // it's a real option, not a fallback.
    const { prompt, usedBaselineTemplate } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1), makeSwing(2)],
    })
    expect(usedBaselineTemplate).toBe(false)
    expect(prompt).toMatch(/ONE cue for the next ball/)
    expect(prompt).toMatch(/stay silent/i)
  })

  it('singular shot language when only one swing', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(prompt).toMatch(/hit 1 forehand in a row/i)
    expect(prompt).toMatch(/LAST 1 SWING:/)
  })
})

describe('buildLiveCoachingPrompt — silence / baseline-template path', () => {
  it('clean-swing scenario: prompt allows silence and the system prompt carries the affirmations', () => {
    // Concrete check that the rendered batch directive permits silence, not
    // just the system prompt's general framing.
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(prompt).toMatch(/stay silent if the batch is already clean/i)
    // System prompt is what the LLM sees as authoritative — the affirmation
    // examples must be there too.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Trust that swing, do it again\./)
  })

  it('baseline-template path: advanced tier flips usedBaselineTemplate=true and reframes the closing directive', () => {
    const result = buildLiveCoachingPrompt({
      profile: advancedProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
    })
    expect(result.usedBaselineTemplate).toBe(true)
    // Advanced default is silence-or-affirmation; the prompt should hand
    // the model the same example phrases the system prompt enumerates.
    expect(result.prompt).toMatch(/silence or one short affirmation/i)
    expect(result.prompt).toMatch(/Clean — repeat that\./)
  })

  it('non-advanced tiers do NOT take the baseline-template path', () => {
    const beginner = buildLiveCoachingPrompt({
      profile: beginnerProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(beginner.usedBaselineTemplate).toBe(false)

    const intermediate = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(intermediate.usedBaselineTemplate).toBe(false)

    const skipped = buildLiveCoachingPrompt({
      profile: null,
      skipped: true,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(skipped.usedBaselineTemplate).toBe(false)
  })
})

describe('buildLiveCoachingPrompt — recentCues (session memory)', () => {
  it('renders a RECENT CUES block when populated', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
      recentCues: [
        'Hip rotation looked closed. Open up earlier.',
        'Still seeing the early finish — let the racket pass your hip first.',
        'Trust that swing, do it again.',
      ],
    })
    expect(prompt).toMatch(/RECENT CUES/)
    expect(prompt).toMatch(/Hip rotation looked closed/)
    expect(prompt).toMatch(/early finish/)
    expect(prompt).toMatch(/Trust that swing, do it again\./)
    // Numbered (oldest -> newest) so the model knows the temporal order.
    expect(prompt).toMatch(/1\. Hip rotation/)
    expect(prompt).toMatch(/3\. Trust that swing/)
  })

  it('persistent-flaw scenario: system prompt instructs model to acknowledge persistence', () => {
    // The "persistence" framing lives in the system prompt — the user-message
    // body just exposes the prior cues. Combined, they should let the model
    // emit "still seeing the early finish, try X" instead of repeating verbatim.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/PRIOR CUES/)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/acknowledge persistence/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/Do not repeat a recent cue verbatim/i)

    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0), makeSwing(1)],
      recentCues: ['Open up the hips earlier.'],
    })
    expect(prompt).toMatch(/RECENT CUES/)
    expect(prompt).toMatch(/Open up the hips earlier\./)
  })

  it('omits the RECENT CUES block when recentCues is missing or empty', () => {
    const empty = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      recentCues: [],
    })
    expect(empty.prompt).not.toMatch(/RECENT CUES/)

    const missing = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
    })
    expect(missing.prompt).not.toMatch(/RECENT CUES/)

    const nullCues = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      recentCues: null,
    })
    expect(nullCues.prompt).not.toMatch(/RECENT CUES/)
  })

  it('caps recent cues at the last 3, dropping older entries', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      recentCues: [
        'Cue alpha (oldest, should be dropped).',
        'Cue beta (also dropped).',
        'Cue gamma.',
        'Cue delta.',
        'Cue epsilon (most recent).',
      ],
    })
    expect(prompt).toMatch(/RECENT CUES/)
    expect(prompt).not.toMatch(/Cue alpha/)
    expect(prompt).not.toMatch(/Cue beta/)
    expect(prompt).toMatch(/Cue gamma/)
    expect(prompt).toMatch(/Cue delta/)
    expect(prompt).toMatch(/Cue epsilon/)
    // Numbered fresh each time so the most recent is always #3.
    expect(prompt).toMatch(/3\. Cue epsilon/)
  })

  it('skips empty / whitespace-only / non-string entries when sanitizing', () => {
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      // Mixed garbage on purpose; sanitizer should pick out only the strings.
      recentCues: ['', '   ', 'Real cue here.', '\n\t', 'Another real cue.'] as unknown as string[],
    })
    expect(prompt).toMatch(/RECENT CUES/)
    expect(prompt).toMatch(/Real cue here\./)
    expect(prompt).toMatch(/Another real cue\./)
    // The whitespace-only entries shouldn't bump the numbering.
    expect(prompt).toMatch(/1\. Real cue here\./)
    expect(prompt).toMatch(/2\. Another real cue\./)
  })

  it('does not crash when recentCues contains non-string junk', () => {
    expect(() =>
      buildLiveCoachingPrompt({
        profile: intermediateProfile,
        skipped: false,
        shotType: 'forehand',
        swings: [makeSwing(0)],
        recentCues: [null, undefined, 42, { text: 'object' }] as unknown as string[],
      }),
    ).not.toThrow()
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [makeSwing(0)],
      recentCues: [null, undefined, 42, { text: 'object' }] as unknown as string[],
    })
    // Nothing valid -> no RECENT CUES block at all.
    expect(prompt).not.toMatch(/RECENT CUES/)
  })
})

describe('buildLiveCoachingPrompt — thin-signal / epistemic boundaries', () => {
  it('thin-signal scenario: prompt explicitly forbids commenting on grip and footwork', () => {
    // Even with a thin angleSummary, the system prompt ringfences the
    // observables; the user message hands the model the data and the
    // closing directive. Together they must keep the model off grip /
    // footwork claims.
    const { prompt } = buildLiveCoachingPrompt({
      profile: intermediateProfile,
      skipped: false,
      shotType: 'forehand',
      // One swing, sparse summary -> classic thin-signal case.
      swings: [
        {
          angleSummary: 'preparation: elbow_R=140°',
          startMs: 0,
          endMs: 500,
        },
      ],
    })

    // The system prompt is where the boundaries live; check both surfaces.
    expect(LIVE_SYSTEM_PROMPT).toMatch(/grip/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/footwork/i)
    expect(LIVE_SYSTEM_PROMPT).toMatch(/zero tolerance/i)

    // User message still permits silence so a thin signal can yield no cue.
    expect(prompt).toMatch(/stay silent if the batch is already clean/i)
  })

  it('thin-signal scenario advanced tier: prefers silence over invented cues', () => {
    const { prompt, usedBaselineTemplate } = buildLiveCoachingPrompt({
      profile: advancedProfile,
      skipped: false,
      shotType: 'forehand',
      swings: [
        {
          angleSummary: 'preparation: shoulder_R=95°',
          startMs: 0,
          endMs: 400,
        },
      ],
    })
    expect(usedBaselineTemplate).toBe(true)
    expect(prompt).toMatch(/silence or one short affirmation/i)
    expect(prompt).toMatch(/Default to silence/i)
  })
})
