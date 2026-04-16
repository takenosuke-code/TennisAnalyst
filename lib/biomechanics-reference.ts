/**
 * Tennis Biomechanics Reference Data
 *
 * Structured reference for LLM prompt injection. Provides ideal joint angle
 * ranges, common amateur mistakes detectable via pose data, coaching cue
 * language, and comparison metrics for when no pro reference exists.
 *
 * Angle conventions match computeJointAngles() in jointAngles.ts:
 *   - elbow_R/L: interior angle at elbow (shoulder-elbow-wrist). 180 = full extension.
 *   - shoulder_R/L: angle at shoulder (cross-shoulder to elbow). ~90 = arm horizontal.
 *   - knee_R/L: interior angle at knee (hip-knee-ankle). 180 = straight leg.
 *   - hip_rotation: angle of hip line relative to horizontal (camera-facing). 0 = square to camera.
 *   - trunk_rotation: angle of shoulder line relative to horizontal. 0 = square to camera.
 *
 * Sources:
 *   Elliott (2006) "Biomechanics and tennis" - British Journal of Sports Medicine
 *   Landlinger et al. (2010) "Key factors in the tennis forehand" - JSSM
 *   Whiteside et al. (2013) "Mechanics of the tennis forehand" - JSSM
 *   Abrams et al. (2011) "8-Stage model for evaluating the tennis serve" - JSSM
 *   Reid et al. (2024) "Kinematics of key points during tennis serve" - Frontiers
 *   Knudson & Blackwell (2000) "Trunk muscle activation in the tennis forehand"
 *   OnCourtAI analysis metrics (2024)
 */

// ---------------------------------------------------------------------------
// 1. FOREHAND BIOMECHANICS
// ---------------------------------------------------------------------------

export const FOREHAND_REFERENCE = `
## FOREHAND BIOMECHANICS -Ideal Joint Angles by Phase

### Preparation (Unit Turn)
- Shoulder rotation (trunk_rot): ~90 degrees from baseline (shoulders nearly perpendicular to net)
- Hip rotation (hip_rot): ~45-60 degrees from baseline (hips turn less than shoulders)
- Hip-shoulder separation angle: 20-30 degrees (shoulders rotated farther than hips)
- Dominant elbow: relaxed, ~90-120 degrees
- Knees: slightly flexed, ~140-160 degrees (athletic ready position)
- Non-dominant arm: extended forward for balance and to help coil the torso

### Loading (Backswing complete, weight on back foot)
- Shoulder rotation: 90-110 degrees from baseline (maximum coil)
- Hip rotation: 45-70 degrees from baseline
- Hip-shoulder separation: 20-30 degrees (key power source -elastic energy storage)
- Dominant knee (back leg): 130-150 degrees (loaded, not locked)
- Wrist: laid back ~90 degrees relative to forearm (racket lag position)
- Weight: primarily on back foot

### Forward Swing / Contact
- Elbow at contact:
  - Eastern grip: ~130 degrees (more extended)
  - Semi-western grip: ~110-120 degrees
  - Western grip: ~100 degrees (more flexed for low-to-high path)
  - General optimal range: 100-140 degrees depending on grip
- Shoulder: actively internally rotating at 800+ degrees/sec in advanced players
- Hip rotation: hips have fired first -now approximately square to baseline (belly button toward target)
- Trunk rotation: shoulders catching up to hips, approaching square at contact
- Knee (front leg): extending to drive upward, ~150-170 degrees
- Contact point: arm extended forward 20-40cm in front of body

### Follow-Through
- Shoulder: continues internal rotation past contact
- Elbow: extends slightly then flexes as arm wraps
- Trunk: fully rotated past square position
- "3x90 rule" at finish: shoulder abduction ~90 degrees, elbow ~90 degrees, wrist variable

### Follow-Through / Finish
- Racket finishes over opposite shoulder or wraps around body
- Weight fully transferred to front foot
- Shoulders have rotated a full ~180 degrees from preparation
- Hips facing the net or slightly past

### Racket Path Angles (vertical)
- Flat drive: ~20 degrees above horizontal
- Moderate topspin: ~30-40 degrees above horizontal
- Heavy topspin: ~40-50+ degrees above horizontal
- Topspin lob: ~70 degrees above horizontal
`;

// ---------------------------------------------------------------------------
// 2. BACKHAND BIOMECHANICS
// ---------------------------------------------------------------------------

export const BACKHAND_REFERENCE = `
## BACKHAND BIOMECHANICS -Ideal Joint Angles by Phase

### ONE-HANDED BACKHAND

#### Preparation
- Shoulder rotation: ~90 degrees (full turn, dominant shoulder pointing toward net)
- Hip-shoulder separation angle: ~30 degrees
- Non-dominant hand on throat of racket helping guide the turn
- Knees: flexed ~140-150 degrees

#### Loading
- Shoulders turned >90 degrees from baseline
- Hips turned ~60-70 degrees
- Dominant arm: elbow slightly bent ~150-160 degrees
- Wrist: hyperextended (firm, locked-back position)
- Back knee: loaded at 130-150 degrees

#### Contact
- Arm extension: 160-175 degrees (nearly full extension for reach and leverage)
- Contact point: 25-40cm in front of front hip
- Wrist: hyperextended and firm (NOT flexed -flexion at contact = tennis elbow risk)
- Skilled players: wrist extending at ~4 rad/s through contact
- Front knee: firm, ~160-170 degrees
- Shoulders approximately square to net

#### Follow-Through
- Arm continues upward and across (50-80cm follow-through distance)
- Non-dominant arm extends behind for counterbalance
- Full body rotation through the shot

### TWO-HANDED BACKHAND

#### Preparation
- Shoulder rotation: ~80-90 degrees from baseline
- Hip-shoulder separation angle: ~20 degrees (less than one-handed)
- Both hands on racket, compact turn

#### Loading
- Trunk rotation: ~80-90 degrees
- Hip rotation: ~60-70 degrees
- Knees: flexed ~140-150 degrees
- Back leg loaded

#### Contact
- Arm extension: 140-160 degrees (less extended than one-handed -acceptable due to second hand)
- Non-dominant arm provides additional force
- Greater trunk rotation contribution than one-handed
- Contact point: 15-30cm in front of body (slightly closer than one-handed)

#### Follow-Through
- Both arms extend together initially
- Non-dominant hand may release after contact
- Follow-through distance: 40-70cm
- Greater axial rotation of pelvis and shoulders compared to one-handed

### Backhand Racket Path Angles
- Heavy topspin: 25-45 degrees upward
- Moderate topspin: 15-25 degrees upward
- Flat drive: 5-15 degrees upward
- Slice: negative angle (high to low)
`;

// ---------------------------------------------------------------------------
// 3. SERVE BIOMECHANICS
// ---------------------------------------------------------------------------

export const SERVE_REFERENCE = `
## SERVE BIOMECHANICS -Ideal Joint Angles by Phase

### Preparation (Stance and Toss)
- Feet: shoulder-width apart, front foot angled ~45 degrees toward net post
- Knees: slightly flexed ~160-170 degrees
- Weight beginning to shift forward
- Toss height: 30-60cm above intended contact point

### Loading (Trophy Position)
- Front knee flexion: ~55-75 degrees (key power metric)
  - Greater flexion (65-75 degrees) produces 32% higher knee extension velocity
  - Insufficient flexion (>80 degrees) eliminates half of potential power
  - Recreational players often stay >90 degrees (too straight) -major power loss
- Trunk inclination: ~25 degrees (lateral lean toward hitting side)
- Shoulder abduction: ~100 degrees (arm approximately horizontal)
- Elbow angle: ~104 degrees (>90 degrees optimal for power and shoulder safety)
- Tossing arm: still extended upward
- Shoulder-pelvis lateral tilt for energy storage

### Cocking (Racket Drop / Maximum Shoulder Layback)
- Shoulder external rotation: 170-172 degrees (maximum "layback")
  - This extreme external rotation stores elastic energy in shoulder rotators
  - Similar to elite baseball pitchers (175-185 degrees)
- Shoulder lateral rotation at racket low point: ~130 degrees
- Elbow flexion: ~104 degrees (maintaining roughly right angle)
- Wrist extension: ~66 degrees
- Knees beginning to extend explosively

### Acceleration
- Lead knee extension velocity: ~800 degrees/sec (explosive leg drive)
- Shoulder rapidly internally rotating (fastest human movement)
- Trunk extending upward and rotating
- Elbow extending rapidly toward contact

### Contact
- Shoulder elevation: ~110 degrees (arm slightly above horizontal)
- Shoulder abduction: ~100 degrees (minimizes joint loading while maximizing velocity)
- Elbow flexion: ~20-30 degrees (nearly full extension)
- Wrist extension: ~15 degrees (about to flex through contact)
- Front knee: ~24 degrees of remaining flexion (nearly straight from extension)
- Trunk tilt: ~48 degrees above horizontal
- Contact height: 22-35cm above player's head at full reach

### Follow-Through / Deceleration
- Shoulder continues internal rotation and decelerates
- Elbow flexes as arm comes down
- Trunk flexes forward
- Landing on front foot with controlled deceleration
- Posterior rotator cuff muscles activate heavily to decelerate

### Force Contributions in Power Serve
- Upper arm internal rotation: 40%
- Hand/wrist flexion: 30%
- Upper arm horizontal flexion: 15%
- Shoulder: 10%
- Forearm pronation: 5%
Note: These exclude the ~54% of total force from legs and trunk

### Serve Knee Bend Benchmarks
- Excellent: 130-150 degrees (deep, powerful loading)
- Good: 150-160 degrees (functional range)
- Needs improvement: >160 degrees (insufficient bend, major power loss)
- Too deep: <120 degrees (balance compromise)
`;

// ---------------------------------------------------------------------------
// 4. COMMON AMATEUR MISTAKES DETECTABLE VIA JOINT ANGLES
// ---------------------------------------------------------------------------

export const AMATEUR_MISTAKES_REFERENCE = `
## COMMON AMATEUR MISTAKES -Detectable from Pose Data

### 1. Insufficient Hip Rotation
- Detection: hip_rot stays flat (<20 degrees change from preparation to contact)
- Ideal: hip_rot changes 40-60+ degrees through the swing
- Impact: massive power loss; forces arm-only compensation
- 90% of recreational players have hips out of sync on forehands

### 2. "Arm-Only" Swing (No Trunk Rotation)
- Detection: trunk_rot changes <15 degrees between loading and contact
- Ideal: trunk_rot changes 60-100+ degrees
- Detection: hip_rot and trunk_rot are nearly identical (no separation angle)
- Impact: poor power, inconsistency, shoulder/elbow injury risk
- A 20% decrease in energy from hip/trunk requires 34% MORE shoulder rotational velocity

### 3. Locked Knees During Loading
- Detection: knee angle >170 degrees during loading phase (forehand/backhand)
- Ideal: 130-160 degrees during loading
- For serve: knee angle >160 degrees at trophy position
- Ideal serve loading: 130-150 degrees
- Impact: eliminates leg drive, reduces power by up to 50% on serve

### 4. Late Contact Point / Cramped Elbow
- Detection: dominant elbow too bent at contact (<90 degrees on forehand)
- Ideal forehand: 100-140 degrees depending on grip
- Detection: contact appears to happen beside or behind the body
- Impact: loss of leverage, reduced racket head speed, "pushing" the ball

### 5. Over-Extended Arm at Contact
- Detection: elbow >175 degrees (locked straight) at contact
- Impact: loss of control, elbow injury risk, reduced ability to generate spin

### 6. Poor Follow-Through
- Detection: shoulder angle and trunk_rot stop rotating at contact instead of continuing
- Detection: minimal change in trunk_rot from contact to finish (<15 degrees)
- Ideal: trunk continues rotating 30-50+ degrees past contact
- Impact: reduced power transfer, deceleration injuries

### 7. Wrist Flipping vs. Proper Lag
- Detection: on forehand, rapid wrist angle change happens before contact rather than at/after contact
- Ideal: wrist maintains ~90 degrees of lag (forearm-to-racket angle) until just before contact
- Forearm-wrist lag ranges:
  - Excellent: 30-50 degrees of lag
  - Good: 20-30 degrees
  - Minimal: 10-20 degrees
  - No lag: <10 degrees
- Impact: premature wrist release = inconsistency, loss of racket head acceleration

### 8. No Hip-Shoulder Separation
- Detection: hip_rot and trunk_rot values are nearly identical throughout backswing
- Ideal separation: 20-30 degrees (forehand), ~30 degrees (one-handed backhand), ~20 degrees (two-handed backhand)
- Impact: cannot store elastic energy in torso, massive power loss

### 9. Insufficient Shoulder Turn in Preparation
- Detection: trunk_rot <45 degrees during preparation/loading
- Ideal: ~90 degrees of shoulder turn from baseline
- Impact: shortened backswing, reduced coil, less potential energy

### 10. Open Stance Without Rotation
- Detection: hip_rot stays <30 degrees throughout entire swing
- Common in beginners who stand facing the net and just swing their arm
- Impact: no ground-reaction force transfer, pure arm swing
`;

// ---------------------------------------------------------------------------
// 5. SWING QUALITY METRICS (NO PRO REFERENCE NEEDED)
// ---------------------------------------------------------------------------

export const SWING_QUALITY_METRICS = `
## SWING QUALITY ASSESSMENT -Metrics Without a Pro Reference

### Kinetic Chain Sequencing
The single most important quality metric. Proper sequence: hips rotate first, then trunk, then arm.
- Detection: peak hip_rot change should occur BEFORE peak trunk_rot change, which should occur BEFORE peak elbow angle change
- Elite players: pelvis peak ~75ms before contact, trunk peak ~57ms before, shoulder ~45ms before
- High-performance: pelvis peak ~93ms before, trunk ~75ms before, shoulder ~61ms before
- Key: it is the TIMING that separates elite from high-performance, not the magnitude
- If trunk_rot peaks before or simultaneously with hip_rot: broken kinetic chain

### Consistency Across Repeated Swings
When multiple swings are available:
- Elite players: joint angle coefficient of variation <6% at contact
- Amateur players: coefficient of variation 15-30% at contact
- Key joints to measure: elbow at contact, trunk_rot at contact, knee at contact
- Higher consistency in end position (contact) is more important than consistency in movement path
- Advanced players actually show MORE variability in joint velocity but LESS variability in final position

### Range of Motion Indicators
Greater ROM generally indicates better technique (up to anatomical limits):
- Shoulder rotation range (preparation to contact): ideally 90-110 degrees of total rotation
- Hip rotation range: ideally 40-70 degrees total
- Knee flexion range on serve: 30-50 degrees (from trophy position to extension)

### Balance Indicators
- Hip symmetry: left and right hip vertical positions should stay relatively level during groundstrokes
- Minimal lateral sway during preparation and loading
- Weight transfer: hip_rot and trunk_rot should shift directionally (not oscillate)
- At finish: body should be balanced, not falling sideways

### Power Indicators (Derivable from Angles)
- Greater hip-shoulder separation = more stored elastic energy
- Deeper knee bend during loading = more potential leg drive
- Larger trunk_rot change from loading to contact = more rotational velocity
- Wrist lag maintained until just before contact = more racket head acceleration

### Quality Rating Framework (Per Phase)
For each phase, assess against the ranges in the stroke-specific sections above:
- 90-100: Within ideal range, proper sequencing
- 70-89: Minor deviations from ideal, one joint slightly off
- 50-69: Noticeable deviations, likely affecting power or consistency
- 30-49: Multiple joints outside ideal ranges, significant technique issues
- 0-29: Fundamental technique problems across multiple joints
`;

// ---------------------------------------------------------------------------
// 6. COACHING CUE LANGUAGE
// ---------------------------------------------------------------------------

export const COACHING_CUES = `
## COACHING CUE LANGUAGE -How to Sound Like a Real Tennis Coach

### Voice and Tone Principles
You are talking to a player standing on court with a racket in their hand. Be the coach they trust.
- Be WARM and DIRECT. No jargon dumps. No spreadsheet voice.
- Always lead with what they do WELL -find the genuine positive before you correct.
- Use metaphors and feel-based cues, not clinical measurements. Angles support your point; they never ARE the point.
- Give ONE actionable drill or feel-based tip per correction. The player should know exactly what to try on the next ball.
- Talk about what the body FEELS like, not what the numbers say. "You're standing too tall through the shot" not "Your knee angle is 172 degrees."

### Metaphors and Feel Cues -Use These Freely

#### Hip and Trunk Rotation
- "Turn your pocket toward the back fence during your backswing"
- "Lead with your hips -feel your belt buckle turning toward the target before your shoulders follow"
- "Think of your torso as a spring -you're coiling it up, and the swing is what happens when you let go"
- "Your hips should beat your shoulders to the ball -hips first, then everything else follows like cracking a whip"
- "Imagine you're standing in a revolving door -your hips push it open, your shoulders ride through"

#### Knee Bend and Leg Drive
- "Load into your back leg -sit down into the shot like you're settling into a low chair"
- "Push up and through the ball -your legs are the engine, your arm is just the steering wheel"
- "On your serve, think 'sit and explode' -deep knee bend, then launch up to the ball like you're trying to dunk a basketball"
- "Feel the ground push you into the shot -all the power starts down there"
- "Straight legs are dead legs. Bend and you'll feel the shot come alive"

#### Shoulder Turn and Preparation
- "Show your back to your opponent during preparation -if they can read the logo on your shirt, you haven't turned enough"
- "Turn your shoulders farther than your hips -that stretch between them is where all your free power lives"
- "Think 'unit turn' -your whole upper body coils as one piece, the racket just goes along for the ride"
- "Get your shoulders turned before the ball even bounces on your side -early preparation buys you time for everything else"

#### Contact Point and Arm
- "Meet the ball out front -reach out like you're shaking someone's hand"
- "Keep that elbow soft, almost like you're cradling the ball into the shot -a locked arm robs you of all your whip"
- "Let the racket head lag behind your hand like the tip of a whip -don't force it forward early"
- "Your arm rides with your chest rotation -you're not muscling the ball, you're letting the body sling it"
- "Imagine you're throwing a frisbee -that natural arm bend and release? That's what we want at contact"

#### Follow-Through
- "Don't stop at the ball -swing THROUGH it and let the racket wrap around to your opposite shoulder"
- "Think 'hit and hold' -freeze your finish position and check: is the racket over your shoulder? Good."
- "The follow-through isn't decoration -it's proof that you accelerated through contact, not into it"

#### Serve-Specific
- "Reach for the sky at contact -like you're grabbing an apple off the highest branch"
- "Think 'scratch your back' as the racket drops behind you, then snap up like you're throwing it over a tall wall"
- "Sit and explode -load those legs deep, then drive up like you're jumping to touch the ceiling"
- "Land inside the court -if you're falling backwards, you didn't commit to the shot"

### How to Weave in Angle Data (IMPORTANT)
GOOD: "You're hitting with a pretty straight arm -around 170° -when the ideal is to keep that elbow soft around 115°. That's robbing you of whip and spin."
BAD: "Your right elbow reaches 171° at contact. The ideal is 110-120°."

GOOD: "Your knees barely bend during the loading phase -you're at about 168° when the pros are sitting into it down around 140°. That's a lot of free power you're leaving on the table."
BAD: "Knee angle is 168° during loading. Ideal range is 130-150°."

The number supports the coaching insight. The coaching insight is always the lead.

### Encouragement Framing -Always Use This
- Instead of "You're not rotating enough" say "Let's get more turn -try showing your back to the net on the next one"
- Instead of "Your knees are too straight" say "Let's sit into the shot more -load those legs and feel the ground push you up into the ball"
- Instead of "You're hitting late" say "Let's meet the ball earlier -reach out to it, don't wait for it to come to you"
- Instead of "Your follow-through is bad" say "Keep swinging through -let the racket travel all the way to your shoulder"
- Frame every correction as an opportunity, not a deficiency. The player should feel motivated, not judged.

### Drill and Tip Examples (Give One Per Correction)
- "Try this: hit 10 forehands where you exaggerate the hip turn -feel your belt buckle point at the target before your arm even starts forward."
- "Here's a good drill: serve with a basketball. You can't muscle a basketball -your body has to do the work. That's the feeling we want."
- "Next time you practice, try shadow swings where you freeze at contact. Check: is your elbow in front of your hip? Are your legs pushing up? That's your checkpoint."
- "Grab a towel instead of a racket. If you can snap it and hear the crack, your kinetic chain is firing in the right order."
`;

// ---------------------------------------------------------------------------
// 7. COMPARISON FRAMEWORK (TWO AMATEURS / SINGLE VIDEO)
// ---------------------------------------------------------------------------

export const COMPARISON_FRAMEWORK = `
## COMPARISON FRAMEWORK -Rating Without a Pro Reference

### Single Video Analysis
When analyzing a single player with no reference:
1. Compare each joint angle against the ideal ranges above
2. Check for kinetic chain sequencing (hips before shoulders before arm)
3. Identify the largest deviation from ideal as the primary coaching point
4. Rate each phase 0-100 based on how close angles are to ideal ranges
5. Prioritize corrections by impact: rotation > knee bend > contact point > follow-through

### Two-Amateur Comparison
When comparing two amateur swings:
1. Calculate which player is closer to ideal ranges for each joint at each phase
2. Identify where Player A is better and where Player B is better
3. Check kinetic chain sequencing for both -better sequencing = better player
4. Compare consistency if multiple swings available (lower variance = better)
5. Compare hip-shoulder separation (more separation = more stored power)
6. Compare knee bend during loading (deeper and within range = better foundation)
7. Do NOT assume the "better" player is correct in all aspects -both may have issues

### Priority Ordering for Corrections
When multiple issues exist, address in this order (highest impact first):
1. Kinetic chain sequencing (broken chain = fundamental issue)
2. Hip-shoulder separation (drives power generation)
3. Knee bend and leg drive (foundation of all power)
4. Shoulder/trunk rotation range (coil and uncoil)
5. Elbow angle at contact (leverage and timing)
6. Wrist position and lag (racket head speed)
7. Follow-through completion (deceleration safety and confirmation of proper mechanics)
8. Balance and recovery (injury prevention and readiness for next shot)
`;

// ---------------------------------------------------------------------------
// Assembled prompt injection
// ---------------------------------------------------------------------------

/**
 * Returns the full biomechanics reference as a single string suitable for
 * injection into an LLM system/user prompt. Optionally filter by stroke type.
 */
export function getBiomechanicsReference(
  strokeType?: 'forehand' | 'backhand' | 'serve' | 'all'
): string {
  const sections: string[] = []

  const type = strokeType ?? 'all'

  if (type === 'forehand' || type === 'all') {
    sections.push(FOREHAND_REFERENCE)
  }
  if (type === 'backhand' || type === 'all') {
    sections.push(BACKHAND_REFERENCE)
  }
  if (type === 'serve' || type === 'all') {
    sections.push(SERVE_REFERENCE)
  }

  // Always include these -they apply regardless of stroke type
  sections.push(AMATEUR_MISTAKES_REFERENCE)
  sections.push(SWING_QUALITY_METRICS)
  sections.push(COACHING_CUES)
  sections.push(COMPARISON_FRAMEWORK)

  return sections.join('\n')
}
