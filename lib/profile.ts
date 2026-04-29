// User profile stored on Supabase `auth.users.user_metadata`.
//
// Written client-side via `supabase.auth.updateUser({ data: {...} })`.
// Read server-side via `supabase.auth.getUser()` then `parseProfile()`.
//
// Skill tier drives how much and what kind of coaching the LLM produces
// (see `app/api/analyze/route.ts`). Handedness + backhand style change
// which dominant-arm joints the prompt emphasizes. Primary goal weights
// which observations get surfaced first.

export type SkillTier =
  | 'beginner' // New to tennis
  | 'intermediate' // Recreational player who rallies consistently
  | 'competitive' // Plays matches, tournaments, strong club level
  | 'advanced' // Top club / college / pro

export type DominantHand = 'right' | 'left'
export type BackhandStyle = 'one_handed' | 'two_handed'

export type PrimaryGoal =
  | 'power'
  | 'consistency'
  | 'topspin'
  | 'slice'
  | 'learning'
  | 'other'

export interface UserProfile {
  skill_tier: SkillTier
  dominant_hand: DominantHand
  backhand_style: BackhandStyle
  primary_goal: PrimaryGoal
  // Only set when primary_goal === 'other'. Bounded to 120 chars to keep
  // the prompt budget sane.
  primary_goal_note: string | null
  onboarded_at: string // ISO 8601
}

const SKILL_TIERS: readonly SkillTier[] = ['beginner', 'intermediate', 'competitive', 'advanced']
const DOMINANT_HANDS: readonly DominantHand[] = ['right', 'left']
const BACKHAND_STYLES: readonly BackhandStyle[] = ['one_handed', 'two_handed']
const PRIMARY_GOALS: readonly PrimaryGoal[] = [
  'power',
  'consistency',
  'topspin',
  'slice',
  'learning',
  'other',
]

export const SKILL_TIER_LABELS: Record<SkillTier, string> = {
  beginner: 'New to tennis',
  intermediate: 'Intermediate',
  competitive: 'Competitive',
  advanced: 'Advanced or pro',
}

export const PRIMARY_GOAL_LABELS: Record<PrimaryGoal, string> = {
  power: 'More power',
  consistency: 'More consistency',
  topspin: 'Cleaner topspin',
  slice: 'Fix my slice',
  learning: 'Just learning',
  other: 'Something else',
}

// Parses raw user_metadata into a validated UserProfile, or null if any
// required field is missing or invalid. Callers use this to decide both
// "is the user onboarded?" and "do we have a profile to consume?"
export function parseProfile(
  metadata: Record<string, unknown> | null | undefined,
): UserProfile | null {
  if (!metadata || typeof metadata !== 'object') return null

  const skill_tier = metadata.skill_tier
  const dominant_hand = metadata.dominant_hand
  const backhand_style = metadata.backhand_style
  const primary_goal = metadata.primary_goal
  const primary_goal_note = metadata.primary_goal_note
  const onboarded_at = metadata.onboarded_at

  if (typeof skill_tier !== 'string' || !SKILL_TIERS.includes(skill_tier as SkillTier)) {
    return null
  }
  if (
    typeof dominant_hand !== 'string' ||
    !DOMINANT_HANDS.includes(dominant_hand as DominantHand)
  ) {
    return null
  }
  if (
    typeof backhand_style !== 'string' ||
    !BACKHAND_STYLES.includes(backhand_style as BackhandStyle)
  ) {
    return null
  }
  if (typeof primary_goal !== 'string' || !PRIMARY_GOALS.includes(primary_goal as PrimaryGoal)) {
    return null
  }
  if (typeof onboarded_at !== 'string' || !onboarded_at) {
    return null
  }

  // primary_goal_note is only meaningful when goal === 'other'; otherwise normalize to null
  let note: string | null = null
  if (primary_goal === 'other') {
    if (typeof primary_goal_note === 'string' && primary_goal_note.trim()) {
      note = primary_goal_note.trim().slice(0, 120)
    }
  }

  return {
    skill_tier: skill_tier as SkillTier,
    dominant_hand: dominant_hand as DominantHand,
    backhand_style: backhand_style as BackhandStyle,
    primary_goal: primary_goal as PrimaryGoal,
    primary_goal_note: note,
    onboarded_at,
  }
}

// Convenience for the routing gate: does the signed-in user have a
// complete profile? Treats malformed metadata as "not onboarded" so the
// gate heals broken entries by re-running onboarding.
export function isOnboarded(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  return parseProfile(metadata) !== null
}

// Did the user explicitly skip the onboarding form? Skipped users get a
// different prompt shape (LLM infers their tier from swing data) but still
// pass through the middleware gate so we don't nag them forever.
export function wasSkipped(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  if (!metadata || typeof metadata !== 'object') return false
  const t = (metadata as Record<string, unknown>).skipped_onboarding_at
  return typeof t === 'string' && t.length > 0
}

// Has the user made any onboarding decision — either completing or skipping?
// Used by middleware to decide whether to redirect to /onboarding. A true
// return here means "pass through"; false means "gate to onboarding".
export function hasCompletedOnboardingFlow(
  metadata: Record<string, unknown> | null | undefined,
): boolean {
  return isOnboarded(metadata) || wasSkipped(metadata)
}

// ---------------------------------------------------------------------------
// Server-side additions. Kept below the shared contract so the type/parser
// surface stays untouched.
// ---------------------------------------------------------------------------

// Minimal shape of the Supabase server client we need. Declared locally so
// this module stays decoupled from @supabase/ssr — the real client satisfies
// it, and tests can pass in a hand-rolled stub.
interface SupabaseLikeClient {
  auth: {
    getUser: () => Promise<{
      data: { user: { user_metadata?: Record<string, unknown> | null } | null }
      error?: unknown
    }>
  }
}

// Fetches the signed-in user and returns their parsed profile, or null if
// the caller is anonymous or their metadata is malformed. Swallowing the
// Supabase error is intentional: the LLM routes fall back to generic
// coaching when there's no profile, and we never want auth hiccups to take
// down the coaching stream.
export async function getProfile(
  client: SupabaseLikeClient,
): Promise<UserProfile | null> {
  try {
    const { data } = await client.auth.getUser()
    const metadata = data.user?.user_metadata
    return parseProfile(metadata)
  } catch {
    return null
  }
}

// Human label for the stated goal, including the free-text note for 'other'.
function describeGoal(profile: UserProfile): string {
  const label = PRIMARY_GOAL_LABELS[profile.primary_goal]
  if (profile.primary_goal === 'other' && profile.primary_goal_note) {
    return `${label} ("${profile.primary_goal_note}")`
  }
  return label
}

// Defanged reconcile rule used only for self-reported-tier players. Previously
// this told the LLM to downgrade mid-response if the swing "looked elementary",
// which produced the Djokovic regression: a single-camera pose summary is
// nowhere near enough signal to override what the player said about themselves.
// New contract: coach to the self-report, full stop. If pose numbers look
// inconsistent with the stated tier, assume camera geometry or a bad take.
const RECONCILE_RULE = `RECONCILE RULE: The self-reported tier is a CONTRACT. Coach to it. Do NOT downgrade the player mid-response because the pose data looks rougher than the stated tier. Single-camera pose estimates are noisy. A weird elbow angle or scattered hip rotation is almost always camera geometry, occlusion, or a single off-take, NOT evidence that the player misreported their level. Trust the self-report. If the numbers surprise you, assume the camera is at fault, pick the cleanest frames to anchor on, and coach them at the tier they told you they're at. Never imply they overestimated themselves. Never phrase feedback like "let's lock in the basics first" unless their tier is beginner. Meet them exactly where they said they are.`

// Rewritten 2026-04 after Park et al. (2024) DPO verbosity bias + MIT/CVPR 2025
// negation-handling work. Then re-tightened 2026-04 against the
// tennis-coaching pedagogy brief (Wulf 2013 external focus; McNevin 2003 +
// 2024 distance-effect meta-analysis; Liao & Masters 2001 analogy learning;
// ITF constraints-led practice). Rules:
//   1. No negations inside TIER_RULES strings themselves. Every directive
//      is a positive "do X" statement. The negation-bearing material
//      (anti-filler list, banned-jargon list) lives in adjacent rule
//      blocks injected by buildTierCoachingBlock. This split is enforced
//      by the TIER_RULES phrasing-invariants test.
//   2. Explicit word cap inside the rule string so the model anchors on it.
//   3. Beginner leads with a strength so the first thing they read is positive.
//   4. Advanced has a literal default baseline template (see ADVANCED_BASELINE_TEMPLATE)
//      — positive-framed, short, and the model copies it verbatim when there
//      is nothing substantive to refine. The advanced TIER_RULES string is
//      deliberately spare — anti-filler / specificity / examples are NOT
//      injected for the advanced tier (see buildTierCoachingBlock).
//   5. Cues across all detail-coached tiers are EXTERNAL-focus (Wulf 2013):
//      attention on the racket, the ball, or a court landmark, not on a
//      muscle. Beginner uses PROXIMAL external referents (landmarks on the
//      body or racket — "racket tip up by your ear"). Competitive can use
//      distal court landmarks ("aim three balls past the opposite service-
//      line corner") since the 2024 meta-analysis shows distance-effect
//      benefits scale with skill.
//   6. Each cue must satisfy at least TWO of: named landmark, named swing
//      phase, measurable target, external-focus referent. SPECIFICITY_RULE
//      states this explicitly with examples; the per-tier rules echo it
//      in tier-appropriate vocabulary (beginner: no biomech jargon;
//      intermediate: stroke-name vocabulary; competitive: phase-name
//      vocabulary).
export const TIER_RULES: Record<SkillTier, string> = {
  beginner: `TIER: Beginner (no serious play, returning casual, or kid). Lead with ONE sentence about a strength you see in their swing. Then give 2 or 3 external-focus coaching cues; each cue must reference a body or racket landmark a 12-year-old could find ("racket tip up by your ear", "step toward the far net post", "finish with your strings facing the back fence", "push the ball toward the cone past the service line"). Each cue pairs ONE landmark with ONE measurable check the player can run on the next swing. Use everyday words: turn sideways, racket up, swing low to high like brushing the ball up a window. Cap mechanical detail at ONE thing per cue. Stay under 120 words total. Close with the literal phrase "Next session, one feel:" followed by a single external-focus check at a named landmark — this is the one thing they carry to the court (e.g. "Next session, one feel: finish with your racket tip up by your ear on every swing").`,
  intermediate: `TIER: Intermediate (NTRP 3.0–4.0; consistent rallies, recognizable stroke shape). Give 2 or 3 coaching cues that mix one foundation tune-up with one refinement. Lead with what's working in two specific sentences (which phase, which landmark), then hand them the cues as numbered feel-based tips. Each cue ties a NAMED swing phase (prep, racket-drop, contact window, follow-through, recovery) to a body or racket landmark AND a measurable check ("contact a foot in front of your front hip; if the ball pulls your shoulder forward, contact was a beat late"). Stroke-name vocabulary is welcome (slice, topspin, open vs closed stance, split step). Stay under 180 words total. Close with the literal phrase "Next session, one feel:" followed by a single external-focus check naming the phase and the landmark (e.g. "Next session, one feel: at the contact window, butt-cap and forearm in one straight line").`,
  competitive: `TIER: Competitive match player (NTRP 4.5+, regular matchplay). Give exactly 3 execution cues focused on millimeter polish and matchplay leverage. Lead with what's working in two specific sentences naming phase and landmark. Each of the 3 cues must (a) name the specific stroke and the specific phase ("forehand contact window", "second-serve toss release", "backhand recovery step"), (b) state the measurable drift you observed against a clean version of that phase, and (c) tie the fix to a tactical pattern they hit on every match (cross-court angle, inside-out forehand, kick-serve location, return depth). Each cue is actionable on the very next ball. Technical vocabulary is welcome where it's accurate: kinetic chain, racket lag, pronation, racket-head speed, swing path, stance load. Stay under 220 words total. Close with the literal phrase "Next session, one feel:" followed by a single phase-named external-focus check tied to the highest-leverage cue above (e.g. "Next session, one feel: belt buckle facing the side fence at the loading turn before the racket drops").`,
  advanced: `TIER: Advanced. Their mechanics are refined. Default to reinforcing the swing: emit the single sentence "This is clean. Save it as your baseline." If, and only if, you see a concrete micro-refinement worth one cue, add ONE short positive tip after that sentence. Stay under 60 words total. Keep the frame positive throughout.`,
}

// Specificity contract — every cue body must do at least TWO of these four
// things. Lifted directly from the pedagogy brief (Wulf 2013; McNevin et al.
// 2003 + 2024 meta). Lives outside TIER_RULES so it can use whatever framing
// reads cleanest, including negations, without breaking the per-tier
// phrasing-invariants test. Injected into the system prompt by
// buildTierCoachingBlock for every detail-coached tier (skipped for
// advanced — that path is the baseline-template branch).
export const SPECIFICITY_RULE = `SPECIFICITY CONTRACT: Every cue body MUST do at least TWO of these four:
1. Name a body or court LANDMARK ("racket tip up by the ear", "ball lands inside the service-line tape", "contact a foot in front of the front hip", "belt buckle facing the side fence", "butt cap pointing back at the ball you just hit").
2. Reference a named swing PHASE ("at the loading turn", "as the back foot pushes", "through the contact window", "into the follow-through", "during the recovery step").
3. Give a MEASURABLE target ("three balls in a row past the service line", "strings still facing the target one full step past contact", "the back foot leaves the ground").
4. Use an EXTERNAL-focus referent — attention on the racket, the ball, or a court target, never on a muscle. For lower-skill players prefer PROXIMAL external focus (landmarks on the body or racket); for higher-skill players a DISTAL court landmark works better.

Vague mood words ("more fluid", "more explosive", "more dynamic"), abstract exhortations ("really focus", "trust it"), and unmeasurable degrees ("a touch more", "a little bit more") are NOT cues. Replace them with a landmark + a measurable check.

When the same fault has both a "don't" framing and a positive framing, ALWAYS pick the positive: "keep the racket-butt and forearm in one straight line through the contact window" instead of "don't break your wrist".`

// Filler / dead-phrase blacklist. The list is verbatim from the pedagogy
// brief's Section 2 ("Filler / dead phrases the LLM must NOT emit"). Negations
// are intentional and live outside TIER_RULES. The phrasing is "if a sentence
// works without one of these, omit it" so the rule survives prompt rewrites
// without becoming a list of magic incantations to avoid.
export const ANTI_FILLER_RULE = `ANTI-FILLER LIST: Strip these phrases from your output. If a sentence works without one of them, omit it; if it doesn't work without one, the sentence itself is filler and should be cut entirely.

1. "As a coach…" / "From a coaching perspective…" (preamble)
2. "That's a great question / great swing!" (throat-clearing)
3. "You might want to consider…" / "It could be helpful to…" (hedge)
4. "Really focus on…" / "Make sure you…" (exhortation, no information)
5. "Keep working hard, you've got this!" (generic encouragement)
6. "Now, moving on to the next point…" (redundant transition)
7. "Try to be more dynamic / fluid / explosive / powerful." (mood word, no referent)
8. "Don't think about your wrist." (negation; the brain renders the noun anyway)
9. "Just relax." (no actionable target)
10. "Feel the flow of the stroke." (vague kinesthetic)
11. "Use your whole body." (no landmark)
12. "Get into the zone." (mystical)
13. "Trust the process." (filler)
14. "A little bit more / a touch more X." (unmeasurable)
15. "Let's lock in the basics first." (condescending, and explicitly banned by the reconcile rule)

Also omit: medical or anatomical claims ("this protects your shoulder"), body-shaming framing ("you're too stiff"), cause-blaming the player ("you're being lazy with your feet"), comparisons to named pros ("swing like Sinner"), and negations as primary cues. Convert every "don't drop the elbow" into a positive "keep the elbow at shoulder height through contact".

SAFETY / SCOPE RAIL — never:
- Imply injury risk or medical concern ("be careful with that wrist", "you might hurt yourself", "that's hard on the elbow"). You are not a clinician.
- Compare unfavorably to a named pro ("Federer would never", "Sinner does this better"). Use anonymous "a clean version of the stroke" instead.
- Infer or comment on age, body type, weight, or physical limitations from a pose video. You can't see those, and guessing damages trust.
- Promise improvement timelines ("you'll see results in two weeks", "give it ten sessions and this clicks"). Players ground out at different paces; don't anchor expectations.`

// Drill-quality contract for the cue.exercises[] field. The structured tool
// schema requires 1–2 exercises per cue; this rule turns those slots into
// court-runnable drills with a constraint, a rep target, a feed type, and a
// success criterion. Lifted from the pedagogy brief Section 4 (ITF
// constraints-led practice). Negations OK because this lives outside
// TIER_RULES. Sized so it fits the 200-char per-exercise schema cap.
export const DRILL_QUALITY_RULE = `DRILL QUALITY: Every entry in a cue's exercises[] array must include (a) a CONSTRAINT that auto-corrects the fault (smaller target, lowered net, restricted stance, hand-feed only, towel under the back foot, cone at a target landing spot), (b) a specific REP COUNT, (c) a FEED TYPE (shadow swing, hand-feed, racket-feed slow, live ball, wall rally, ball machine), and (d) a SUCCESS CRITERION the player can self-score on court.

Good: "Shadow forehands with a cone at your contact point: 15 reps, swing only counts if the racket tip ends up by your ear and the cone is still standing."
Good: "Cross-court forehand wall rally, hand-feed start: 20 contacts in a row that land above the wall stripe. Reset to zero if one misses."
Bad: "Practice forehands until they feel better." (no constraint, no count, no success criterion)
Bad: "Work on your contact point." (no feed, no measurable target)

Beginners start at shadow or hand-feed. Intermediate moves to racket-feed and wall rallies. Competitive uses live ball with a constraint.`

// Cue priority rule. The structured tool schema lets the model emit 2–3 cues,
// but the synthesis flagged that not ranking them is a missed opportunity:
// the player only acts on what's at the top, and an unranked list pushes the
// "highest-leverage fix" into a coin flip with cosmetic refinements. Inject
// this into every detail-coached tier so cues[0] is always the load-bearing
// fix.
export const CUE_PRIORITY_RULE = `CUE PRIORITY: Order cues by leverage, not by where they appear in the swing. cues[0] is the ONE thing the player should focus on first — the highest-leverage fix on this clip, the fix that unlocks the most other improvements. cues[1] and cues[2] (if present) are secondary refinements. The "Next session, one feel:" closing line MUST anchor on cues[0], not on a later cue. Never list cues in chronological swing order if that ordering buries the highest-leverage fix at position 2 or 3.`

// Few-shot benchmark cues. Lifted verbatim from the pedagogy brief Section 6
// (Gold-standard cue examples). Each is a body-field exemplar that obeys the
// specificity contract: external focus, named landmark, named phase,
// measurable check, no hedges, no negations, no medical claims. Tier tags
// in parens so the model picks the right vocabulary register without us
// needing three copies of this block.
export const COACHING_EXAMPLES_BLOCK = `EXAMPLES of strong cue bodies (imitate this register, do not copy the words):

1. Wrist breaks at contact (intermediate). "Lock the butt-cap and your forearm into one straight line as the strings meet the ball, and hold that line until the racket clears your front hip. The check: your strings should still be facing the target one full step past contact. If they twist early, the wrist let go."

2. Weak hip rotation on the forehand (competitive). "Start the swing with your belt buckle facing the side fence and finish it with the buckle facing the opposite net post. Push the back hip toward the incoming ball as the racket drops below it. Three reps in a row where the back foot ends up off the ground means the hips, not the arm, drove the ball."

3. Pushy follow-through (intermediate). "Throw the racket head up and over your opposite shoulder so the butt cap finishes pointing back at the ball you just hit. The strings should brush up the back of the ball like wiping a window from low to high. If the racket stops in front of you, the hand decelerated before contact cleared."

4. Late contact point on the forehand (beginner). "Catch the ball out in front of your lead hip, close enough that you could shake hands with it. Finish with the racket tip up by your ear on the opposite side. If the ball pulls your shoulder forward as you hit, you waited a beat too long."

5. Serve toss drifting behind the head (competitive). "Release the toss when your tossing arm is straight up alongside your front ear, and let the ball land a foot inside the baseline and a foot to the right of your front foot. Mark that spot with a cone for the first basket of serves. If the ball lands behind the baseline, the release was late."`

// Literal string the advanced tier defaults to. When the tool_use output
// emits this exact sentence (and nothing else substantive), we flag the
// telemetry row with used_baseline_template=true. Exported so both the
// metrics extractor and integration tests reference one source of truth.
export const ADVANCED_BASELINE_TEMPLATE = 'This is clean. Save it as your baseline.'

// Per-tier token ceiling passed to Anthropic's max_tokens. Bumped twice:
// once for per-cue exercises, again for per-cue keyPoints (bullet
// summaries). Each cue now carries body + keyPoints + exercises, so the
// per-response budget needed another ~15% headroom on top of the
// post-exercises bump. Ordering still reflects the research invariant:
// advanced produces the least, competitive the most. Monotonic.
export const TIER_MAX_TOKENS: Record<SkillTier, number> = {
  advanced: 400,
  beginner: 900,
  intermediate: 1200,
  competitive: 1500,
}

// Fallback used when no profile is available (generic / skipped path).
// Between intermediate and competitive so the skipped-path inference has
// room to pick any tier.
export const DEFAULT_MAX_TOKENS = 1400

// Tool-use contract. Each tier forces the model through a JSON schema that
// mechanically bounds the number of cues it can emit. COACHING_TOOL_NAME is
// the only tool we expose; the caller forces
// tool_choice = { type: "tool", name: COACHING_TOOL_NAME } so the model must
// call it and cannot freeform past max_tokens.
export const COACHING_TOOL_NAME = 'emit_coaching' as const

export type CoachingToolInput = {
  strengths: Array<{ text: string }>
  // Each cue is one coaching point.
  //   `body` is the conversational coach-voice paragraph with analogies.
  //   `keyPoints` is the same advice as 1-3 short bullets (no analogies,
  //     no feel cues, just the literal mechanic) for users who want a
  //     quick read.
  //   `exercises` is 1-2 short court-runnable drills that fix the issue.
  // exercises and keyPoints are optional so the advanced-baseline path
  // (cues=[]) still validates.
  cues: Array<{
    title: string
    body: string
    keyPoints?: Array<string>
    exercises?: Array<string>
  }>
  closing: string
}

// Per-tier count + length limits. Used both to build the schema and to
// validate parsed tool input server-side (defense in depth).
const TIER_CUE_BOUNDS: Record<SkillTier, { minCues: number; maxCues: number; maxCueBodyChars: number }> = {
  beginner:     { minCues: 2, maxCues: 3, maxCueBodyChars: 280 },
  intermediate: { minCues: 2, maxCues: 3, maxCueBodyChars: 350 },
  competitive:  { minCues: 3, maxCues: 3, maxCueBodyChars: 350 },
  advanced:     { minCues: 0, maxCues: 1, maxCueBodyChars: 210 },
}

export const COACHING_TOOL_SCHEMAS: Record<SkillTier, {
  name: typeof COACHING_TOOL_NAME
  description: string
  input_schema: Record<string, unknown>
}> = Object.fromEntries(
  (Object.keys(TIER_CUE_BOUNDS) as SkillTier[]).map((tier) => {
    const { minCues, maxCues, maxCueBodyChars } = TIER_CUE_BOUNDS[tier]
    return [
      tier,
      {
        name: COACHING_TOOL_NAME,
        description: `Emit tennis coaching for a ${tier} player. Follow the schema exactly — the player only sees what you put in these fields.`,
        input_schema: {
          type: 'object',
          additionalProperties: false,
          required: ['strengths', 'cues', 'closing'],
          properties: {
            strengths: {
              type: 'array',
              minItems: 1,
              maxItems: 2,
              items: {
                type: 'object',
                additionalProperties: false,
                required: ['text'],
                properties: {
                  text: { type: 'string', minLength: 4, maxLength: 200 },
                },
              },
            },
            cues: {
              type: 'array',
              minItems: minCues,
              maxItems: maxCues,
              items: {
                type: 'object',
                additionalProperties: false,
                // keyPoints + exercises are required when cues are
                // emitted (every recommendation should come with both a
                // quick-read summary and 1-2 drills). The advanced-
                // baseline path emits cues=[] entirely, so this only
                // bites when there's something to fix.
                required: ['title', 'body', 'keyPoints', 'exercises'],
                properties: {
                  title: { type: 'string', minLength: 2, maxLength: 60 },
                  body:  { type: 'string', minLength: 2, maxLength: maxCueBodyChars },
                  keyPoints: {
                    type: 'array',
                    minItems: 1,
                    maxItems: 3,
                    items: { type: 'string', minLength: 4, maxLength: 90 },
                  },
                  exercises: {
                    type: 'array',
                    minItems: 1,
                    maxItems: 2,
                    items: { type: 'string', minLength: 8, maxLength: 200 },
                  },
                },
              },
            },
            closing: { type: 'string', minLength: 2, maxLength: 200 },
          },
        },
      },
    ]
  }),
) as unknown as Record<SkillTier, { name: typeof COACHING_TOOL_NAME; description: string; input_schema: Record<string, unknown> }>

// Parses a raw tool_use input blob into a validated CoachingToolInput.
// Returns null on any shape mismatch — callers fall back to the plain-text
// streaming path. Enforces TIER_CUE_BOUNDS at the TS layer too (Anthropic's
// schema validation is best-effort; we defense-in-depth guard tier=null by
// falling back to intermediate bounds).
export function buildCoachingToolInput(
  rawInput: unknown,
  tier: SkillTier | null,
): CoachingToolInput | null {
  if (!rawInput || typeof rawInput !== 'object') return null
  const obj = rawInput as Record<string, unknown>

  const rawStrengths = obj.strengths
  if (!Array.isArray(rawStrengths)) return null
  if (rawStrengths.length < 1 || rawStrengths.length > 2) return null
  const strengths: Array<{ text: string }> = []
  for (const s of rawStrengths) {
    if (!s || typeof s !== 'object') return null
    const text = (s as Record<string, unknown>).text
    if (typeof text !== 'string') return null
    const trimmed = text.trim()
    if (!trimmed) return null
    strengths.push({ text: trimmed })
  }

  const bounds = TIER_CUE_BOUNDS[tier ?? 'intermediate']
  const rawCues = obj.cues
  if (!Array.isArray(rawCues)) return null
  if (rawCues.length < bounds.minCues || rawCues.length > bounds.maxCues) return null
  const cues: Array<{
    title: string
    body: string
    keyPoints?: Array<string>
    exercises?: Array<string>
  }> = []
  // Generic helper for the parallel keyPoints + exercises validation:
  // both are arrays of trimmed non-empty strings within length bounds.
  // Tolerate absent (older persisted rows) by returning undefined.
  const parseStringArray = (
    raw: unknown,
    maxItems: number,
  ): { ok: true; value: string[] | undefined } | { ok: false } => {
    if (raw === undefined) return { ok: true, value: undefined }
    if (!Array.isArray(raw)) return { ok: false }
    if (raw.length > maxItems) return { ok: false }
    const cleaned: string[] = []
    for (const item of raw) {
      if (typeof item !== 'string') return { ok: false }
      const trimmed = item.trim()
      if (!trimmed) return { ok: false }
      cleaned.push(trimmed)
    }
    return { ok: true, value: cleaned.length > 0 ? cleaned : undefined }
  }
  for (const c of rawCues) {
    if (!c || typeof c !== 'object') return null
    const title = (c as Record<string, unknown>).title
    const body = (c as Record<string, unknown>).body
    if (typeof title !== 'string' || typeof body !== 'string') return null
    const trimmedTitle = title.trim()
    const trimmedBody = body.trim()
    if (!trimmedTitle || !trimmedBody) return null

    const kp = parseStringArray((c as Record<string, unknown>).keyPoints, 3)
    if (!kp.ok) return null
    const ex = parseStringArray((c as Record<string, unknown>).exercises, 2)
    if (!ex.ok) return null

    const cue: {
      title: string
      body: string
      keyPoints?: Array<string>
      exercises?: Array<string>
    } = { title: trimmedTitle, body: trimmedBody }
    if (kp.value) cue.keyPoints = kp.value
    if (ex.value) cue.exercises = ex.value
    cues.push(cue)
  }

  const rawClosing = obj.closing
  if (typeof rawClosing !== 'string') return null
  const closing = rawClosing.trim()
  if (!closing || closing.length > 200) return null

  return { strengths, cues, closing }
}

// Normalize a string for template-match comparison: lowercase, collapse
// whitespace + punctuation (periods, commas, em-dashes, en-dashes, hyphens)
// to a single space, trim. Used to detect ADVANCED_BASELINE_TEMPLATE even
// when the model echoes back minor punctuation variants.
function normalizeForTemplateCompare(s: string): string {
  return s
    .toLowerCase()
    .replace(/[.,\u2014\u2013\-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function matchesBaselineTemplate(s: string): boolean {
  if (!s) return false
  return normalizeForTemplateCompare(s) === normalizeForTemplateCompare(ADVANCED_BASELINE_TEMPLATE)
}

// Deterministic renderer: CoachingToolInput -> markdown for LLMCoachingPanel's
// MarkdownText (understands `## heading` + `**bold**` only).
//
// Default shape:
//   ## What You're Doing Well
//   <strengths[0].text>
//   <strengths[1].text>   (if present)
//
//   ## Quick Read           (omitted when no cue has keyPoints)
//   **1. <cues[0].title>**
//   - <kp> ...
//
//   ## Detailed Read        (omitted when cues.length === 0)
//   **1. <cues[0].title>**
//   <cues[0].body>
//   *Try this:* - <ex>...
//
//   ## Practice Focus
//   <closing>
//
// Advanced-baseline special case: when tier === 'advanced' AND cues.length === 0
// AND closing matches ADVANCED_BASELINE_TEMPLATE (via normalized compare),
// render ONLY the literal template string with no headings.
export function renderCoachingToolInputToMarkdown(
  input: CoachingToolInput,
  tier: SkillTier | null,
): string {
  if (
    tier === 'advanced' &&
    input.cues.length === 0 &&
    matchesBaselineTemplate(input.closing)
  ) {
    return ADVANCED_BASELINE_TEMPLATE
  }

  const parts: string[] = []
  parts.push(`## What You're Doing Well`)
  for (const s of input.strengths) {
    parts.push(s.text)
  }

  if (input.cues.length > 0) {
    // Quick Read section first: bullet summaries up top so a skimmer
    // gets the takeaways immediately. Each cue's bullets are nested
    // under its title so the skimmer still knows which point each
    // bullet belongs to. Only emitted when at least one cue has
    // keyPoints — older persisted rows without keyPoints fall through
    // to the detailed section only.
    const hasAnyKeyPoints = input.cues.some(
      (c) => c.keyPoints && c.keyPoints.length > 0,
    )
    if (hasAnyKeyPoints) {
      parts.push('')
      parts.push('## Quick Read')
      input.cues.forEach((cue, idx) => {
        if (!cue.keyPoints || cue.keyPoints.length === 0) return
        parts.push('')
        parts.push(`**${idx + 1}. ${cue.title}**`)
        for (const kp of cue.keyPoints) {
          parts.push(`- ${kp}`)
        }
      })
    }

    // Detailed Read: coach-voice paragraph + court-runnable drills for
    // the reader who wants the full reasoning behind each Quick Read
    // bullet.
    parts.push('')
    parts.push('## Detailed Read')
    input.cues.forEach((cue, idx) => {
      parts.push('')
      parts.push(`**${idx + 1}. ${cue.title}**`)
      parts.push(cue.body)
      if (cue.exercises && cue.exercises.length > 0) {
        parts.push('')
        parts.push('*Try this:*')
        for (const ex of cue.exercises) {
          parts.push(`- ${ex}`)
        }
      }
    })
  }

  parts.push('')
  parts.push('## Practice Focus')
  parts.push(input.closing)

  return parts.join('\n')
}

// Counts occurrences of `**N. <title>**` or leading `N.` at line start.
// Dedupe by the captured integer, return the count of distinct 1-indexed
// integers that appear in a plausible ascending sequence (1; or 1,2;
// or 1,2,3). Malformed/skipped sequences collapse to the count of the
// first in-sequence run. Exported for unit testing.
export function countTipsInMarkdown(md: string): number {
  if (!md) return 0
  const re = /^\s*(?:\*\*)?\s*(\d+)\.\s/gm
  const seen = new Set<number>()
  const ordered: number[] = []
  for (const m of md.matchAll(re)) {
    const n = parseInt(m[1], 10)
    if (!Number.isFinite(n)) continue
    if (seen.has(n)) continue
    seen.add(n)
    ordered.push(n)
  }
  if (ordered.length === 0) return 0
  // Count the longest prefix starting at 1 where each next is +1.
  if (ordered[0] !== 1) return 0
  let count = 1
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i] === ordered[i - 1] + 1) count++
    else break
  }
  return count
}

// Central metrics extractor used by both analyze routes. Handles both the
// structured (tool_use) path and the markdown fallback.
export function extractResponseMetrics(args: {
  tier: SkillTier | null
  toolInput: CoachingToolInput | null
  markdownText: string
  outputTokens: number | null
}): {
  response_token_count: number | null
  response_tip_count: number | null
  response_char_count: number | null
  used_baseline_template: boolean
} {
  const { tier, toolInput, markdownText, outputTokens } = args

  const response_token_count = outputTokens
  const response_char_count = typeof markdownText === 'string' ? markdownText.length : 0

  let response_tip_count: number | null
  if (toolInput) {
    response_tip_count = toolInput.cues.length
  } else if (markdownText) {
    response_tip_count = countTipsInMarkdown(markdownText)
  } else {
    response_tip_count = null
  }

  let used_baseline_template = false
  if (tier === 'advanced') {
    if (toolInput) {
      used_baseline_template =
        toolInput.cues.length === 0 && matchesBaselineTemplate(toolInput.closing)
    } else if (markdownText) {
      used_baseline_template = matchesBaselineTemplate(markdownText.trim())
    }
  }

  return {
    response_token_count,
    response_tip_count,
    response_char_count,
    used_baseline_template,
  }
}

// Trailing instruction appended to every prompt branch so we can collect
// telemetry on what tier the LLM actually thought the swing was, even when the
// defanged RECONCILE_RULE prevents it from acting on that belief. The server
// parses and strips this line before the client sees the response.
const TIER_ASSESSMENT_TRAILER = `At the very END of your response, on its own final line, emit EXACTLY:
[TIER_ASSESSMENT: <beginner|intermediate|competitive|advanced|unknown>]
Do not explain this line. The server parses and strips it before the player sees it.`

// Exported so the analyze routes can parse + strip the trailer after the
// stream buffers. Unanchored (global) + case-insensitive — if the LLM appends
// a period/emoji/extra paragraph after the tag, we still match + strip it
// rather than leaking the raw `[TIER_ASSESSMENT: ...]` line to the user.
export const TIER_ASSESSMENT_REGEX =
  /\[TIER_ASSESSMENT:\s*(beginner|intermediate|competitive|advanced|unknown)\s*\]/gi

// Parse the trailer out of a buffered response. Returns the assessed tier
// (lowercase) and the response with the trailer stripped. If no trailer is
// found, returns null + the original text. We also strip any trailing blank
// lines left behind after the trailer is removed so the coaching text doesn't
// look truncated. Uses the LAST match in the buffer — the model occasionally
// quotes the format earlier in coaching, and we only care about the real tag.
export function parseTierAssessmentTrailer(buffered: string): {
  assessedTier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | 'unknown' | null
  stripped: string
} {
  const matches = Array.from(buffered.matchAll(TIER_ASSESSMENT_REGEX))
  if (matches.length === 0) {
    return { assessedTier: null, stripped: buffered }
  }
  const last = matches[matches.length - 1]
  const tier = last[1].toLowerCase() as
    | 'beginner'
    | 'intermediate'
    | 'competitive'
    | 'advanced'
    | 'unknown'
  // Strip EVERY trailer instance (defensive — the model occasionally emits
  // more than one), then clean up trailing whitespace.
  const stripped = buffered.replace(TIER_ASSESSMENT_REGEX, '').replace(/\s+$/, '')
  return { assessedTier: tier, stripped }
}

// Ordering used to decide whether the LLM's assessed tier represents a
// downgrade from the self-reported tier. 'unknown' is not a real tier; callers
// treat a null assessed tier as "no downgrade detected".
const TIER_RANK: Record<SkillTier, number> = {
  beginner: 0,
  intermediate: 1,
  competitive: 2,
  advanced: 3,
}

export function tierRank(tier: SkillTier): number {
  return TIER_RANK[tier]
}

// True when the LLM's assessed tier is strictly lower than the tier we coached
// to. Returns false whenever either side is null/unknown — missing data is not
// a downgrade signal.
export function isTierDowngrade(
  coachedTier: SkillTier | null,
  assessedTier: 'beginner' | 'intermediate' | 'competitive' | 'advanced' | 'unknown' | null,
): boolean {
  if (!coachedTier || !assessedTier || assessedTier === 'unknown') return false
  return TIER_RANK[assessedTier] < TIER_RANK[coachedTier]
}

// Fallback block when no profile is available (legacy users, anonymous
// sessions). Keeps the prompt coherent without assuming a tier — mirrors the
// old SKILL_CALIBRATION behavior. Trailer instruction appended so we still
// capture the assessed tier for anon sessions.
const GENERIC_CALIBRATION = `READ THE SKILL LEVEL FIRST:
Before giving any advice, look at the joint angles and phase timing. Judge how refined this swing already is.
- Polished / near-pro mechanics: give SUBTLE refinements, not rebuilds.
- Solid intermediate: point to the 2 or 3 biggest gaps and give practical drills.
- Still developing: focus on foundations, and pick the ONE thing that unlocks everything else.
Match the advice to what the swing actually needs. Never default to generic cues.

${TIER_ASSESSMENT_TRAILER}`

// Returns the prompt fragment that tier-specific coaching is built from.
// Designed to be interpolated into the route prompts — contains the tier
// rule, reconcile rule, handedness context, and goal weighting.
//
// For the three detail-coached tiers (beginner / intermediate / competitive)
// we also inject the SPECIFICITY_RULE, ANTI_FILLER_RULE, DRILL_QUALITY_RULE,
// and COACHING_EXAMPLES_BLOCK so the LLM has the concrete contract + few-
// shot exemplars in front of it. The advanced tier is deliberately spared:
// its baseline-template path is meant to be terse and drift-focused, and
// loading three pages of specificity rules into a "save this as your
// baseline" response would defeat that.
export function buildTierCoachingBlock(profile: UserProfile | null): string {
  if (!profile) return GENERIC_CALIBRATION

  const tierRule = TIER_RULES[profile.skill_tier]
  const handednessLabel = profile.dominant_hand === 'right' ? 'right-handed' : 'left-handed'
  // Dominant-side references are what the LLM uses to pick which joints to
  // name in its feedback; inverted for lefties.
  const dominantSide = profile.dominant_hand === 'right' ? 'right' : 'left'
  const nonDominantSide = profile.dominant_hand === 'right' ? 'left' : 'right'
  const backhandLabel =
    profile.backhand_style === 'one_handed' ? 'one-handed backhand' : 'two-handed backhand'
  const goalLabel = describeGoal(profile)

  // Advanced stays minimal. The three detail-coached tiers get the full
  // specificity / anti-filler / drill-quality / examples scaffolding.
  const detailBlocks =
    profile.skill_tier === 'advanced'
      ? ''
      : `\n\n${SPECIFICITY_RULE}\n\n${CUE_PRIORITY_RULE}\n\n${ANTI_FILLER_RULE}\n\n${DRILL_QUALITY_RULE}\n\n${COACHING_EXAMPLES_BLOCK}`

  return `${tierRule}

${RECONCILE_RULE}

PLAYER CONTEXT:
- Handedness: ${handednessLabel}. Their dominant arm is the ${dominantSide} one; reference "${dominantSide} shoulder / elbow / wrist" for swing-side cues and "${nonDominantSide}" for the off-hand.
- Backhand style: ${backhandLabel}. When discussing the backhand side, tailor cues to this grip.

GOAL WEIGHTING:
- The player's stated priority is: ${goalLabel}. Prioritize observations that move the needle on this goal and mention it explicitly in at least one cue.${detailBlocks}

${TIER_ASSESSMENT_TRAILER}`
}

// Prompt block for users who skipped onboarding. The LLM is told there is no
// self-reported tier, asked to classify the swing from the data, name its
// inferred tier in a short italic header, then coach using that tier's rules.
//
// Unlike the self-reported path (where RECONCILE_RULE is defanged into a
// "coach to the contract" instruction), the skipped path keeps the three-
// signal override gate: the LLM is allowed to shift tier mid-response only if
// multiple independent signals agree the swing is different from the initial
// guess. Without a self-report, there is no contract to preserve.
export function buildInferredTierCoachingBlock(): string {
  return `INFERRED TIER MODE: The player declined to self-report their skill level. Classify their swing from the joint angle and phase data into ONE of these four tiers, then coach to that tier:

- beginner: wild inconsistency across frames, arming the ball with little trunk rotation, phase timing scattered, no clear kinetic chain.
- intermediate: rallies consistently, recognizable stroke shape, but timing and rotation still need work.
- competitive: solid fundamentals, clean kinetic chain, mostly about execution polish and matchplay refinements.
- advanced: clean mechanics end to end, only micro-refinements left, groove the baseline rather than rebuild anything.

NAME YOUR INFERRED TIER: At the very top of your response, on its own line, emit the tier you picked in italic parentheses. Keep it short and non-intrusive, for example:
*(coaching you as intermediate, set your profile to recalibrate)*
Do this once, then move straight into the normal coaching sections.

TIER RULES (use the one matching the tier you picked):
- beginner: ${TIER_RULES.beginner}
- intermediate: ${TIER_RULES.intermediate}
- competitive: ${TIER_RULES.competitive}
- advanced: ${TIER_RULES.advanced}

RECONCILE RULE (three-signal override): Because there is no self-report to anchor on, you are allowed to shift your inferred tier mid-response if, and only if, at least THREE independent signals agree with the shift. Examples of independent signals: kinetic chain sequencing, trunk rotation magnitude, contact-point consistency across frames, racket-drop depth, phase-timing regularity. One weird number is not a signal. One off-frame is not a signal. If you shift, be explicit: "updating my read to <tier>" and continue from there. If only one or two signals disagree with your first guess, assume noise and stay the course.

When you pick beginner, intermediate, or competitive, the rules below also apply. When you pick advanced, ignore them and use the baseline-template path described above.

${SPECIFICITY_RULE}

${CUE_PRIORITY_RULE}

${ANTI_FILLER_RULE}

${DRILL_QUALITY_RULE}

${COACHING_EXAMPLES_BLOCK}

${TIER_ASSESSMENT_TRAILER}`
}

// One-shot fetch of everything the coaching routes need from auth: the parsed
// profile plus whether the user explicitly skipped onboarding. Keeps the
// routes to a single getUser() round trip instead of calling getProfile()
// and a separate skipped-check.
export async function getCoachingContext(
  client: SupabaseLikeClient,
): Promise<{ profile: UserProfile | null; skipped: boolean }> {
  try {
    const { data } = await client.auth.getUser()
    const metadata = data.user?.user_metadata ?? null
    return {
      profile: parseProfile(metadata),
      skipped: wasSkipped(metadata),
    }
  } catch {
    return { profile: null, skipped: false }
  }
}
