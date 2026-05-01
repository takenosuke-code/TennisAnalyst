// ---------------------------------------------------------------------------
// Cross-task stroke analysis types — single source of truth.
//
// These four interfaces flow across the stroke-detection / scoring /
// comparison / rendering pipeline. They previously lived as duplicated
// declarations across lib/swingDetect.ts, lib/strokeQuality.ts,
// app/api/compare-strokes/route.ts and components/StrokeRibbon.tsx.
// Co-locating them here gives the compiler one canonical contract so a
// future edit to the producer can't silently drift from the consumer.
//
// Don't add fields here that aren't shared by every consumer — module-
// local refinements (e.g. StrokeQualityComponents in strokeQuality.ts)
// stay where they're used.
// ---------------------------------------------------------------------------

export interface DetectedStroke {
  /** `stroke_${index}` where index is zero-based in peakFrame ascending order. */
  strokeId: string
  /** Inclusive start frame index after pre-padding (clamped to ≥ 0). */
  startFrame: number
  /** Inclusive end frame index after post-padding (clamped to ≤ totalFrames-1). */
  endFrame: number
  /** Frame index of the wrist-speed peak that anchored this stroke. */
  peakFrame: number
  /** Effective fps used for ms ↔ frame conversion. */
  fps: number
}

/**
 * Reasons a detected stroke can be hard-rejected by the quality gate.
 * Kept as a closed string union so consumers (e.g. the StrokeRibbon
 * REJECT_LABEL map) can use `Record<StrokeRejectReason, …>` to enforce
 * exhaustiveness at compile time.
 */
export type StrokeRejectReason =
  | 'low_visibility'
  | 'camera_pan'
  | 'camera_zoom'
  | 'too_short'
  | 'missing_data'

export interface StrokeQualityResult {
  strokeId: string
  /** z-scored across non-rejected strokes; higher = better; NaN if rejected. */
  score: number
  rejected: boolean
  rejectReason?: StrokeRejectReason
  components: {
    /** Raw peak wrist speed (linear), not z-scored. Units = coord/sec. */
    peakWristSpeed: number
    /** |actual_lag - session_median_lag|, where actual_lag = shoulder_peak_t - hip_peak_t (ms). */
    kineticChainTimingError: number
    /** Variance of dominant-arm elbow joint angle over the contact window. */
    wristAngleVariance: number
  }
}

export interface StrokeComparisonResult {
  best: { strokeId: string; reasoning: string; citations: string[] } | null
  worst: { strokeId: string; reasoning: string; citations: string[] } | null
  isConsistent: boolean
  consistentCue?: string
}
