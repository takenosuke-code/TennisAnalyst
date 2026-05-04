import { create } from 'zustand'

export type LiveStatus =
  | 'idle'
  | 'preflight'
  | 'recording'
  | 'uploading'
  | 'complete'
  | 'error'

export type LivePermissionState = 'unknown' | 'granted' | 'denied'

export type LiveTranscriptEntry = {
  id: string
  text: string
  // Time (ms since recording start) when this line was produced by the LLM
  sessionMs: number
  // X-Analysis-Event-Id from /api/live-coach — used at stop time to backfill
  // the session_id / segment_id on the corresponding analysis_events row
  eventId: string | null
  // How many swings this line summarized
  swingCount: number
}

// Dev-only visibility into what the live coach is "thinking" each batch.
// Captured per-batch by useLiveCoach and rendered by CoachDebugPanel below
// the live video so the player can sanity-check that the right inputs are
// flowing in and the right cues are coming out during a demo.
export type CoachDebugEntry = {
  id: string
  // Time (ms since recording start) when this batch fired
  sessionMs: number
  // Number of swings sent in this batch
  swingCount: number
  // The exact angleSummary strings that were shipped to /api/live-coach
  angleSummaries: string[]
  // The 6-word cue that came back, or a marker for non-spoken outcomes
  outcome:
    | { kind: 'cue'; text: string }
    | { kind: 'silence' }
    | { kind: 'error' }
  // Round-trip latency for the (final, post-retry) /api/live-coach request
  latencyMs: number
}

type LiveShotType = 'forehand' | 'backhand' | 'serve' | 'volley'

interface LiveStore {
  status: LiveStatus
  errorMessage: string | null

  shotType: LiveShotType
  cameraPermission: LivePermissionState
  micPermission: LivePermissionState

  swingCount: number
  lastBatchAtMs: number | null
  sessionStartedAtMs: number | null

  transcript: LiveTranscriptEntry[]
  // Dev visibility log; never spoken, never persisted to analysis_events.
  // Capped to the last N entries inside appendCoachDebugEntry so a long
  // demo doesn't grow the array unboundedly.
  coachDebugLog: CoachDebugEntry[]
  ttsEnabled: boolean
  ttsAvailable: boolean
  // Set when the most recent /api/live-coach request failed (network /
  // server error). Cleared when the next batch succeeds OR the model
  // deliberately returned silence. Surfaces as a visible inline error in
  // the transcript header so the player isn't left wondering why nothing
  // is happening.
  coachingError: string | null

  // True while a coach request is awaiting a response. Phase 3
  // visibility surface — drives the "Listening to your last few
  // swings…" pulse on the transcript header. Mirrored from
  // useLiveCoach.isRequestInFlight() by LiveCapturePanel; we don't
  // mutate it from inside useLiveCoach so the batching path stays
  // untouched.
  coachRequestInFlight: boolean

  setStatus: (status: LiveStatus) => void
  setErrorMessage: (msg: string | null) => void
  setShotType: (t: LiveShotType) => void
  setCameraPermission: (p: LivePermissionState) => void
  setMicPermission: (p: LivePermissionState) => void
  setSwingCount: (n: number) => void
  setLastBatchAtMs: (ms: number | null) => void
  setSessionStartedAtMs: (ms: number | null) => void
  appendTranscriptEntry: (entry: LiveTranscriptEntry) => void
  clearTranscript: () => void
  appendCoachDebugEntry: (entry: CoachDebugEntry) => void
  clearCoachDebugLog: () => void
  setTtsEnabled: (v: boolean) => void
  setTtsAvailable: (v: boolean) => void
  setCoachingError: (msg: string | null) => void
  setCoachRequestInFlight: (v: boolean) => void
  resetSession: () => void
}

const INITIAL: Pick<
  LiveStore,
  | 'status'
  | 'errorMessage'
  | 'shotType'
  | 'cameraPermission'
  | 'micPermission'
  | 'swingCount'
  | 'lastBatchAtMs'
  | 'sessionStartedAtMs'
  | 'transcript'
  | 'coachDebugLog'
  | 'ttsEnabled'
  | 'ttsAvailable'
  | 'coachingError'
  | 'coachRequestInFlight'
> = {
  status: 'idle',
  errorMessage: null,
  shotType: 'forehand',
  cameraPermission: 'unknown',
  micPermission: 'unknown',
  swingCount: 0,
  lastBatchAtMs: null,
  sessionStartedAtMs: null,
  transcript: [],
  coachDebugLog: [],
  ttsEnabled: true,
  ttsAvailable: false,
  coachingError: null,
  coachRequestInFlight: false,
}

const COACH_DEBUG_LOG_MAX = 30

export const useLiveStore = create<LiveStore>((set) => ({
  ...INITIAL,
  setStatus: (status) => set({ status }),
  setErrorMessage: (errorMessage) => set({ errorMessage }),
  setShotType: (shotType) => set({ shotType }),
  setCameraPermission: (cameraPermission) => set({ cameraPermission }),
  setMicPermission: (micPermission) => set({ micPermission }),
  setSwingCount: (swingCount) => set({ swingCount }),
  setLastBatchAtMs: (lastBatchAtMs) => set({ lastBatchAtMs }),
  setSessionStartedAtMs: (sessionStartedAtMs) => set({ sessionStartedAtMs }),
  appendTranscriptEntry: (entry) =>
    set((state) => ({ transcript: [...state.transcript, entry] })),
  clearTranscript: () => set({ transcript: [] }),
  appendCoachDebugEntry: (entry) =>
    set((state) => {
      const next = [...state.coachDebugLog, entry]
      const trimmed =
        next.length > COACH_DEBUG_LOG_MAX
          ? next.slice(next.length - COACH_DEBUG_LOG_MAX)
          : next
      return { coachDebugLog: trimmed }
    }),
  clearCoachDebugLog: () => set({ coachDebugLog: [] }),
  setTtsEnabled: (ttsEnabled) => set({ ttsEnabled }),
  setTtsAvailable: (ttsAvailable) => set({ ttsAvailable }),
  setCoachingError: (coachingError) => set({ coachingError }),
  setCoachRequestInFlight: (coachRequestInFlight) => set({ coachRequestInFlight }),
  resetSession: () =>
    set({
      status: 'idle',
      errorMessage: null,
      swingCount: 0,
      lastBatchAtMs: null,
      sessionStartedAtMs: null,
      transcript: [],
      coachDebugLog: [],
      coachingError: null,
      coachRequestInFlight: false,
    }),
}))
