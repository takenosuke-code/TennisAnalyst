import type { Landmark, JointAngles, PoseFrame } from './supabase'
import { detectStrokes } from './swingDetect'
import { scoreStrokes } from './strokeQuality'

// MediaPipe Pose landmark indices
export const LANDMARK_INDICES = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
}

// IDs that are actually populated by the RTMPose -> BlazePose-33 mapping
// (see lib/cocoToBlazepose.ts:COCO17_TO_BLAZEPOSE33). The other 16 of 33
// BlazePose slots ride along at {x:0, y:0, visibility:0} on every Railway
// extraction. Visibility-based gating MUST average over only these IDs;
// folding the unfilled slots into the mean drops a clean clip to ~0.40
// even when every real landmark is high-visibility, which then collapses
// the multiplicative confidence gate downstream.
export const RTMPOSE_FILLED_LANDMARK_IDS: readonly number[] = Object.freeze([
  0,                                       // nose
  2, 5, 7, 8,                              // eyes + ears
  11, 12, 13, 14, 15, 16,                  // shoulders, elbows, wrists
  23, 24, 25, 26, 27, 28,                  // hips, knees, ankles
])

// Joint groups for toggle visibility
export const JOINT_GROUPS = {
  shoulders: [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.RIGHT_SHOULDER],
  elbows: [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.RIGHT_ELBOW],
  wrists: [LANDMARK_INDICES.LEFT_WRIST, LANDMARK_INDICES.RIGHT_WRIST],
  hips: [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.RIGHT_HIP],
  knees: [LANDMARK_INDICES.LEFT_KNEE, LANDMARK_INDICES.RIGHT_KNEE],
  ankles: [LANDMARK_INDICES.LEFT_ANKLE, LANDMARK_INDICES.RIGHT_ANKLE],
} as const

export type JointGroup = keyof typeof JOINT_GROUPS

// Skeleton connections for drawing lines between joints
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Torso
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.RIGHT_SHOULDER],
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.LEFT_HIP],
  [LANDMARK_INDICES.RIGHT_SHOULDER, LANDMARK_INDICES.RIGHT_HIP],
  [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.RIGHT_HIP],
  // Left arm
  [LANDMARK_INDICES.LEFT_SHOULDER, LANDMARK_INDICES.LEFT_ELBOW],
  [LANDMARK_INDICES.LEFT_ELBOW, LANDMARK_INDICES.LEFT_WRIST],
  // Right arm
  [LANDMARK_INDICES.RIGHT_SHOULDER, LANDMARK_INDICES.RIGHT_ELBOW],
  [LANDMARK_INDICES.RIGHT_ELBOW, LANDMARK_INDICES.RIGHT_WRIST],
  // Left leg
  [LANDMARK_INDICES.LEFT_HIP, LANDMARK_INDICES.LEFT_KNEE],
  [LANDMARK_INDICES.LEFT_KNEE, LANDMARK_INDICES.LEFT_ANKLE],
  // Right leg
  [LANDMARK_INDICES.RIGHT_HIP, LANDMARK_INDICES.RIGHT_KNEE],
  [LANDMARK_INDICES.RIGHT_KNEE, LANDMARK_INDICES.RIGHT_ANKLE],
]

function vec2(a: Landmark, b: Landmark): [number, number] {
  return [b.x - a.x, b.y - a.y]
}

function angleBetween(v1: [number, number], v2: [number, number]): number {
  const dot = v1[0] * v2[0] + v1[1] * v2[1]
  const mag1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2)
  const mag2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2)
  // Degenerate landmarks (NaN, overlapping joints, missing keypoints) used
  // to return 0 here, which downstream `min()` then picked up as a real
  // measurement of "arm folded shut at 0°". Returning NaN lets the
  // Number.isFinite filters in coachingObservations skip the bad frame.
  if (mag1 === 0 || mag2 === 0) return NaN
  const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)))
  return (Math.acos(cosAngle) * 180) / Math.PI
}

function getLandmark(landmarks: Landmark[], id: number): Landmark | undefined {
  return landmarks.find((l) => l.id === id)
}

export function computeJointAngles(landmarks: Landmark[]): JointAngles {
  const lShoulder = getLandmark(landmarks, LANDMARK_INDICES.LEFT_SHOULDER)
  const rShoulder = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_SHOULDER)
  const lElbow = getLandmark(landmarks, LANDMARK_INDICES.LEFT_ELBOW)
  const rElbow = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ELBOW)
  const lWrist = getLandmark(landmarks, LANDMARK_INDICES.LEFT_WRIST)
  const rWrist = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_WRIST)
  const lIndex = getLandmark(landmarks, LANDMARK_INDICES.LEFT_INDEX)
  const rIndex = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_INDEX)
  const lHip = getLandmark(landmarks, LANDMARK_INDICES.LEFT_HIP)
  const rHip = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_HIP)
  const lKnee = getLandmark(landmarks, LANDMARK_INDICES.LEFT_KNEE)
  const rKnee = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_KNEE)
  const lAnkle = getLandmark(landmarks, LANDMARK_INDICES.LEFT_ANKLE)
  const rAnkle = getLandmark(landmarks, LANDMARK_INDICES.RIGHT_ANKLE)

  const angles: JointAngles = {}

  if (rShoulder && rElbow && rWrist) {
    angles.right_elbow = angleBetween(
      vec2(rElbow, rShoulder),
      vec2(rElbow, rWrist)
    )
  }
  if (lShoulder && lElbow && lWrist) {
    angles.left_elbow = angleBetween(
      vec2(lElbow, lShoulder),
      vec2(lElbow, lWrist)
    )
  }
  if (lShoulder && rShoulder && rElbow) {
    angles.right_shoulder = angleBetween(
      vec2(rShoulder, lShoulder),
      vec2(rShoulder, rElbow)
    )
  }
  if (lShoulder && rShoulder && lElbow) {
    angles.left_shoulder = angleBetween(
      vec2(lShoulder, rShoulder),
      vec2(lShoulder, lElbow)
    )
  }
  // Wrist flexion: elbow → wrist → index. 180° = straight, smaller = flexed.
  // RTMPose-2D doesn't predict the LEFT_INDEX / RIGHT_INDEX (ids 19/20)
  // landmarks; cocoToBlazepose leaves them at {x:0, y:0, visibility:0}.
  // Truthiness checks pass because getLandmark finds the slot, so we
  // also gate on visibility ≥ 0.5 — without this the wrist angle is
  // computed against a vector pointing at the image origin, which
  // poisons every downstream consumer (kineticChain timing, cue rules,
  // strokeQuality wrist-variance).
  const INDEX_VIS_FLOOR = 0.5
  if (rElbow && rWrist && rIndex && (rIndex.visibility ?? 1) >= INDEX_VIS_FLOOR) {
    angles.right_wrist = angleBetween(
      vec2(rWrist, rElbow),
      vec2(rWrist, rIndex)
    )
  }
  if (lElbow && lWrist && lIndex && (lIndex.visibility ?? 1) >= INDEX_VIS_FLOOR) {
    angles.left_wrist = angleBetween(
      vec2(lWrist, lElbow),
      vec2(lWrist, lIndex)
    )
  }
  if (rHip && rKnee && rAnkle) {
    angles.right_knee = angleBetween(
      vec2(rKnee, rHip),
      vec2(rKnee, rAnkle)
    )
  }
  if (lHip && lKnee && lAnkle) {
    angles.left_knee = angleBetween(
      vec2(lKnee, lHip),
      vec2(lKnee, lAnkle)
    )
  }
  // Hip rotation: signed angle of the hip line in screen-space.
  // Range (-180, 180]. Returning the SIGNED angle (was Math.abs(...))
  // is required for excursion to be correct — a hip line that tilts
  // +10° (loading) -> 0° (square) -> -10° (follow-through) has true
  // arc 20°, but Math.abs collapses the negative side and reports
  // max=10°, min=0°, excursion=10°. Downstream consumers (excursion,
  // truncated_followthrough) now use signed math.
  if (lHip && rHip) {
    const hipVec = vec2(lHip, rHip)
    angles.hip_rotation = (Math.atan2(hipVec[1], hipVec[0]) * 180) / Math.PI
  }
  // Trunk rotation: signed shoulder-line angle. Same reasoning as hip.
  if (lShoulder && rShoulder) {
    const shoulderVec = vec2(lShoulder, rShoulder)
    angles.trunk_rotation = (Math.atan2(shoulderVec[1], shoulderVec[0]) * 180) / Math.PI
  }

  return angles
}

// Detected swing segment within a longer video
export type SwingSegment = {
  index: number           // 1-based swing number
  startFrame: number      // index into frames array
  endFrame: number
  startMs: number
  endMs: number
  peakFrame: number       // frame with max activity
  frames: PoseFrame[]
}

/**
 * Detect individual swings in a sequence of frames.
 *
 * Backwards-compatible wrapper around `detectStrokes()` (lib/swingDetect.ts).
 * The underlying algorithm is the wrist-velocity peak-pick rewrite — see
 * the doc comment in lib/swingDetect.ts for design notes. This wrapper
 * preserves the long-standing `SwingSegment[]` shape that downstream UI
 * code (SwingSelector, SwingBaselineGrid, baseline/compare,
 * coachingObservations, analyze/page) was written against.
 *
 * The legacy hysteresis-tuning options (`restThreshold`, `enterThreshold`,
 * `exitThreshold`, `mergeGapFrames`) are accepted-and-ignored to keep the
 * call-site contract intact. They have no analogue in the new algorithm —
 * peak-picking + refractory + biomechanical padding replaces hysteresis +
 * region-merging + fixed pre/post pad.
 *
 * `minSwingFrames` is still honored: short clips below the floor return a
 * single whole-clip segment so callers built around "always have something
 * to render" don't crash on micro-clips.
 */
export function detectSwings(
  allFrames: PoseFrame[],
  opts: {
    minSwingFrames?: number
    /** @deprecated Hysteresis-era knob; ignored under the peak-pick algorithm. */
    restThreshold?: number
    /** @deprecated Hysteresis-era knob; ignored under the peak-pick algorithm. */
    enterThreshold?: number
    /** @deprecated Hysteresis-era knob; ignored under the peak-pick algorithm. */
    exitThreshold?: number
    /** @deprecated Hysteresis-era knob; ignored under the peak-pick algorithm. */
    mergeGapFrames?: number
    /** Override fps when timestamps are unreliable. */
    fps?: number
    /** Profile dominant-hand override forwarded to detectStrokes. */
    dominantHand?: 'left' | 'right' | null
    /**
     * When true, runs `scoreStrokes` and filters out strokes flagged
     * `rejected: true` (low visibility, camera pan/zoom, missing
     * kinematic chain data). Default false — preserves the legacy
     * "always return everything detectStrokes saw" contract that
     * synthetic test fixtures rely on. Set true at the analyze /
     * baseline UI surface so phantom video cards from too-noisy
     * strokes don't reach users.
     */
    dropRejected?: boolean
    /**
     * When true, runs the swing-shape verifier on each candidate and
     * drops those whose joint trajectories don't match a real swing
     * (insufficient hip rotation, no proximal-to-distal kinetic
     * chain). Forwarded to detectStrokes. Default false.
     */
    verifyShape?: boolean
  } = {}
): SwingSegment[] {
  const { minSwingFrames = 12, dropRejected = false, verifyShape = false } = opts

  // Whole-clip fallback for very short inputs — preserves the legacy
  // "always return something" behavior callers rely on.
  if (allFrames.length < minSwingFrames) {
    return [{
      index: 1,
      startFrame: 0,
      endFrame: Math.max(0, allFrames.length - 1),
      startMs: allFrames[0]?.timestamp_ms ?? 0,
      endMs: allFrames[allFrames.length - 1]?.timestamp_ms ?? 0,
      peakFrame: Math.floor(allFrames.length / 2),
      frames: allFrames,
    }]
  }

  // Short-clip pass-through: if the whole input is under 5 seconds
  // AND the caller isn't asking us to filter rejected strokes, skip
  // Voronoi slicing and return the entire clip as a single swing.
  // The alternative (Voronoi-bound the detected peaks) was producing
  // tightly-trimmed slices on short single-swing clips when the
  // detector picked up grip adjustments / footwork as fake additional
  // peaks — the LLM then saw a partial swing and reported "1° hip
  // rotation," etc. Below 5s the user is almost always uploading a
  // single swing anyway.
  //
  // We still run detectStrokes to find the actual wrist-velocity peak
  // — the rules need an accurate peakFrame to identify the contact
  // moment. The output is the full clip with the strongest peak's
  // peakFrame stamped on it.
  //
  // Gated on !dropRejected because that flag asks the rule layer to
  // throw out low-quality strokes; bypassing the detection there
  // would let phantom whole-clip strokes through the quality filter.
  if (!dropRejected && allFrames.length >= 2) {
    const firstTs = allFrames[0]?.timestamp_ms ?? 0
    const lastTs = allFrames[allFrames.length - 1]?.timestamp_ms ?? 0
    if (lastTs - firstTs < 5000) {
      const peaks = detectStrokes(allFrames, {
        fps: opts.fps,
        dominantHand: opts.dominantHand,
        verifyShape,
      })
      const primaryPeakFrame =
        peaks.length > 0
          ? peaks[0].peakFrame
          : Math.floor(allFrames.length / 2)
      return [{
        index: 1,
        startFrame: 0,
        endFrame: allFrames.length - 1,
        startMs: firstTs,
        endMs: lastTs,
        peakFrame: primaryPeakFrame,
        frames: allFrames,
      }]
    }
  }

  let strokes = detectStrokes(allFrames, {
    fps: opts.fps,
    dominantHand: opts.dominantHand,
    verifyShape,
  })

  if (dropRejected && strokes.length > 0) {
    const scores = scoreStrokes(strokes, allFrames)
    if (scores.length === 0) {
      // scoreStrokes returns [] when EVERY stroke was rejected. The
      // earlier `scores.length > 0` guard silently let those phantom
      // strokes through — exactly the bug this option was added to
      // prevent. Treat empty as "all rejected" and clear the list.
      strokes = []
    } else {
      const surviving = new Set<string>()
      for (const s of scores) {
        if (!s.rejected) surviving.add(s.strokeId)
      }
      strokes = strokes.filter((s) => surviving.has(s.strokeId))
    }
  }

  if (strokes.length === 0) {
    // When the caller asked us to drop rejected strokes, return an
    // empty list rather than the whole-clip fallback — the fallback
    // exists for synthetic-fixture compatibility, not for clips where
    // every detected stroke failed quality. Returning the whole clip
    // would produce a single phantom card spanning the full video.
    if (dropRejected) return []
    // No peaks survived — fall through to the whole-clip segment so the
    // analyze / baseline UIs don't render an empty state for clips
    // where the fixture is genuinely flat (synthetic tests, very still
    // setups). Real-world rallies always have peaks.
    return [{
      index: 1,
      startFrame: 0,
      endFrame: allFrames.length - 1,
      startMs: allFrames[0]?.timestamp_ms ?? 0,
      endMs: allFrames[allFrames.length - 1]?.timestamp_ms ?? 0,
      peakFrame: Math.floor(allFrames.length / 2),
      frames: allFrames,
    }]
  }

  // Adapt DetectedStroke[] to SwingSegment[]. peakFrame from
  // DetectedStroke is an absolute frame index — same semantics as
  // SwingSegment.peakFrame. Consumers like SwingBaselineGrid use
  // `swing.frames[swing.peakFrame - swing.startFrame]` so we preserve
  // that invariant by carrying the absolute index through.
  return strokes.map((s, i) => ({
    index: i + 1, // 1-based for legacy consumers
    startFrame: s.startFrame,
    endFrame: s.endFrame,
    startMs: allFrames[s.startFrame]?.timestamp_ms ?? 0,
    endMs: allFrames[s.endFrame]?.timestamp_ms ?? 0,
    peakFrame: s.peakFrame,
    frames: allFrames.slice(s.startFrame, s.endFrame + 1),
  }))
}

// Re-export the new stroke detector + types so callers wanting the
// strict contract can import them from the same module.
export { detectStrokes, deriveFps } from './swingDetect'
export type { DetectedStroke, DetectStrokesOptions } from './swingDetect'

// Sample N evenly-spaced frames from a keypoints sequence
export function sampleKeyFrames(frames: PoseFrame[], count = 5): PoseFrame[] {
  if (count <= 1) return frames.slice(0, 1)
  if (frames.length <= count) return frames
  const step = (frames.length - 1) / (count - 1)
  return Array.from({ length: count }, (_, i) =>
    frames[Math.round(i * step)]
  )
}

// Build a compact prompt-safe summary of angles (no raw x/y coords)
export function buildAngleSummary(
  frames: PoseFrame[],
  phaseNames: string[] = ['preparation', 'loading', 'contact', 'follow-through', 'finish']
): string {
  const sampled = sampleKeyFrames(frames, 5)
  return sampled
    .map((frame, i) => {
      const a = frame.joint_angles
      const phase = phaseNames[i] ?? `frame_${frame.frame_index}`
      return `${phase}: elbow_R=${a.right_elbow?.toFixed(0) ?? 'N/A'}° elbow_L=${a.left_elbow?.toFixed(0) ?? 'N/A'}° shoulder_R=${a.right_shoulder?.toFixed(0) ?? 'N/A'}° knee_R=${a.right_knee?.toFixed(0) ?? 'N/A'}° hip_rot=${a.hip_rotation?.toFixed(0) ?? 'N/A'}° trunk_rot=${a.trunk_rotation?.toFixed(0) ?? 'N/A'}°`
    })
    .join('\n')
}

// ---------------------------------------------------------------------------
// Camera-robust helpers (re-exported from `lib/cameraNormalization`).
//
// These are siblings of the camera-frame helpers above, factored out so
// callers can opt into camera-invariant comparisons without reaching
// across modules. The implementations live in `lib/cameraNormalization.ts`
// — kept there because they share private constants (visibility floor,
// hip-line minimum) with the other camera-normalization utilities. We
// re-export here so `import { computeRotationExcursion } from
// '@/lib/jointAngles'` is the natural ergonomic path for code already
// pulling other angle helpers from this module.
// ---------------------------------------------------------------------------

export {
  computeBodyFrameAngles,
  computeRotationExcursion,
  computeCameraSimilarity,
} from './cameraNormalization'
export type { CameraClass } from './cameraNormalization'
