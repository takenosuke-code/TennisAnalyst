# Phase E spike findings (Worker E1)

Three deliverables that gate Phase E (Modal-during-live):
1. fMP4 MediaRecorder spike page (real-device test required).
2. /api/extract short-clip audit (desk check, done).
3. Modal warm-up ping shape (proposal, done).

This doc is a hand-off to E2/E3/E4. Section 4 captures everything I noticed
that is out of scope for E1 but worth knowing about.

---

## Section 1 — fMP4 spike page

**Path:** `app/dev/fmp4-spike/page.tsx` (a single client-component page; no
sibling files needed).

**How to run.** Start `next dev`, then on a phone visit
`http://<dev-host>:3000/dev/fmp4-spike`. The page is dev-only and not linked
from the main nav. It needs `getUserMedia` (HTTPS or localhost only — for
remote phone testing use a tunnel like ngrok or `next dev --experimental-https`).

**What the page does:**
- Logs `navigator.userAgent` at the top so you can confirm what device /
  browser you ran it on without DevTools.
- Probes `MediaRecorder.isTypeSupported` for five MIME candidates:
  `video/mp4;codecs=avc1`, `video/mp4`, `video/webm;codecs=vp9`,
  `video/webm;codecs=vp8`, `video/webm`. Renders a YES/no list.
- Start/Stop buttons drive `getUserMedia({ video: { facingMode: 'environment' }, audio: false })`
  + `MediaRecorder.start(1000)` (1s timeslice) using the highest-priority
  supported MIME.
- Each `dataavailable` chunk is rendered as a row showing `#index · bytes ·
  mimeType` plus a "Play this chunk alone" button. The button wraps the
  chunk in a fresh `Blob([chunk], { type: 'video/mp4' })`, makes an object
  URL, points a `<video>` at it, and waits for the `playing` event (4s
  budget). Result rendered inline:
  - **green `ok`** = chunk has its own moov atom and fMP4 chunked playback works.
  - **red `error`** = chunk depends on a prior init segment (classic MP4
    behavior); fragmented mode isn't actually fragmenting on this browser.
- "Upload last 3s to /api/extract" button concatenates the most recent 3
  chunks (`new Blob(chunks, { type: containerMime })`), uploads via
  `@vercel/blob/client`'s `upload()` against `handleUploadUrl: '/api/upload'`
  (mirrors `components/LiveCapturePanel.tsx:680-684`), then POSTs the blob
  URL + a fresh `crypto.randomUUID()` to `/api/extract`. Renders the JSON
  response, total elapsed ms, and error if any.

**What the user should look for, on phone:**
- The MIME-support list. Plan assumes Chrome Android + iOS Safari 17+ both
  return YES for `video/mp4;codecs=avc1`. **Pending real-device check.**
- Whether each chunk plays standalone (green vs. red). The plan's option (1)
  ("send one fMP4 sub-clip to Modal") only works if the answer is "every
  chunk after #0 plays standalone." If chunks #1+ all fail, fall back to
  plan's option (2) (multi-URL request) or use a JS muxer.
  **Pending real-device check.**
- The /api/extract response. Look for `{ status: 'queued' }` (good — the
  3s clip was accepted by Railway). 502/503 means the proxy or Railway
  rejected; investigate logs. **Pending real-device check.**

**Out of scope (intentional).** The page does NOT poll for keypoints. The
sessionId is a fresh uuid that doesn't have a row in `user_sessions`, so
Railway's background task will silently no-op the write-back. The
synchronous queue ack is the only signal we need to gate Phase E. End-to-end
keypoint round-trip belongs to E2/E3 once the real session row is wired.

---

## Section 2 — /api/extract short-clip audit

**Question A: Is there a minimum-duration check that would reject a 3–15s clip?**

**Answer: No.**

I grep'd the four target files and the rest of `railway-service/` for
`min_duration`, `min_seconds`, `too short`, etc. The only "minimum" hits
are:
- `railway-service/extraction.py:194 MOTION_PREPASS_MIN_DURATION_SEC = 90.0`
  — this is the *upper* threshold above which the motion pre-pass kicks in.
  It's a "skip the optimization on short clips" gate, not a rejection gate.
- `railway-service/scene_detector.py:41` (irrelevant — YouTube pipeline).
- `__tests__` files (test fixtures, not runtime gates).

Walking the code:
- `app/api/extract/route.ts:23-86` validates only `sessionId` + `blobUrl`
  presence and forwards to Railway. No duration check.
- `railway-service/main.py:95-122` (`@app.post("/extract")`) validates
  the URL allowlist and queues the background task. No duration check.
- `railway-service/main.py:154-272` (`_run_extraction`) downloads the video
  and calls `extract_keypoints_from_video` — no duration gate.
- `railway-service/extraction.py:656-664` (`extract_keypoints_from_video`)
  delegates to `_extract_with_rtmpose` — no duration gate.
- `railway-service/modal_inference.py:251-386` (`extract_pose`) validates
  the URL allowlist and forwards. No duration gate.

**Question B: Does adaptive 8fps sampling kick in below 2 minutes?**

**Answer: No — confirmed.** `railway-service/extraction.py:412-419`:

```py
if duration_sec_est > 120:
    effective_sample_fps = min(sample_fps, 8)
elif duration_sec_est > 60:
    effective_sample_fps = min(sample_fps, 12)
elif duration_sec_est > 30:
    effective_sample_fps = min(sample_fps, 18)
else:
    effective_sample_fps = sample_fps
```

For a 3–15s clip, `duration_sec_est <= 30` so `effective_sample_fps = 30`
(the requested default). The plan's claim is correct. A 4-swing batch
(~10–15s) gets full 30fps server extraction.

**Question C: Will a 3–15s clip just work, or does it need a code change?**

**Answer: it should just work.** No min-duration gate, full 30fps sampling,
motion pre-pass disabled (it would have been a problem if it kicked in —
the percentile threshold + 0.5s minimum window length would risk dropping a
short clip with a single swing). Both Modal and Railway-local paths are
clean.

**Question D: If a change is needed, name the file + scope.**

None required. (Marker for E2: if real-device testing surfaces a problem,
the candidate site would be `railway-service/extraction.py:_extract_with_rtmpose`
since that's where every duration-related branch lives — but I don't expect
a change to be needed.)

One adjacent observation worth flagging (not blocking): for very short
clips, `motion_skipped` and `racket_yolo_skipped` telemetry will read 0
which is a tiny diagnostic wart but harmless.

---

## Section 3 — Modal warm-up ping shape

**Constraint.** `railway-service/modal_inference.py:196-205` `_is_url_allowed`
rejects anything not on the allowlist (`blob.vercel-storage.com`,
`public.blob.vercel-storage.com`, `supabase.co`, `supabase.com`,
`storage.googleapis.com`). Plus the function expects a video URL it can
`httpx.get()` and feed through `extract_keypoints_from_video`. A
zero-payload "ping" can't reach the model code without first passing the
URL allowlist *and* downloading a real video.

The /api/extract proxy on Vercel adds a second wrinkle: it requires
`sessionId` + `blobUrl`, and Railway requires `Bearer EXTRACT_API_KEY`
which only the server route holds.

**Three options I considered.**

### Option A — tiny static MP4 hosted on Vercel Blob, called via `/api/extract`

Upload a ~10KB single-frame mp4 once (manually or via a one-shot script) to
Vercel Blob. The URL is permanent. At session start, the browser POSTs
`{ blobUrl: <warmup-url>, sessionId: <fake-warmup-id> }` to `/api/extract`,
fire-and-forget.

- **Pros:** zero changes to Railway or Modal. Reuses the entire existing
  pipeline. The allowlist is satisfied (Vercel Blob host). Testing surface
  is the same as production.
- **Cons:** runs full pipeline on the warm-up frame (~1s of inference even
  warm). Wastes a tiny amount of cycles vs. a true no-op, but it warms
  the *exact* code path the first real swing will hit, which is actually a
  feature: if there's a regression in YOLO/RTMPose loading, the warm-up
  request surfaces it before the player swings. Cold-start cost (5–10s) is
  paid by the warm-up request, which is what we want.
- **Side effect:** Railway's `_run_extraction` will try to `supabase.update`
  a row keyed by the fake sessionId. That's a silent no-op (UPDATE with no
  matching rows = 0 rows affected). Verified by reading
  `railway-service/main.py:239-243`: there's no "row not found" branch.
  Acceptable but should be documented in the warm-up helper.

### Option B — new GET endpoint on the Modal function

Add a `@modal.fastapi_endpoint(method="GET")` `warm_up()` that imports the
extraction modules and returns `{status: "warm"}`. Forces a redeploy.

- **Pros:** truly zero-cost ping. Zero side effects.
- **Cons:** requires Modal redeploy (and the `modal-deploy.md` runbook
  flagged risks of URL rotation in the past — see modal_inference.py
  module docstring). Adds a new public Modal endpoint. Needs a separate
  Vercel proxy (`/api/modal-warmup`?) since the browser can't hit Modal
  directly (the Modal URL is server-only by design).

### Option C — call /api/extract with an obviously-bogus URL just to wake the proxy

Wouldn't reach Modal — Railway rejects on the URL allowlist before forwarding.
**Not viable.** (Documented for completeness so E2 doesn't reinvent it.)

### Recommendation: **Option A.**

Same code path as production, no Modal redeploy, no new endpoints. The
small inference cost on warm-up is a bonus correctness check.

**Implementation sketch (for E2/E3 to write — owned files for that worker):**

```ts
// lib/liveModalWarmup.ts (NEW — in E2 scope, not E1)
//
// Fire-and-forget warm-up ping. Hits /api/extract with a tiny static
// blob URL so the Modal container is hot before the first real swing.
//
// The static asset is a 1-frame mp4 we upload once to Vercel Blob.
// URL is pinned in env (NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL) so we can
// rotate without a code deploy. Falls back silently when the env is
// unset — warm-up is best-effort.

const WARMUP_URL = process.env.NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL

export function fireModalWarmup(): void {
  if (!WARMUP_URL) return
  const sessionId = `warmup-${crypto.randomUUID()}`
  void fetch('/api/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blobUrl: WARMUP_URL, sessionId }),
    keepalive: true,
  }).catch(() => {/* warm-up failures are non-fatal */})
}
```

Call site: in `useLiveCapture.ts`, immediately after camera permission
grant, before the recorder starts. One-shot per session. Optional: a
periodic re-ping every 25s while idle (Modal's scaledown is 30s) — but
that's an enhancement, not gate-blocking.

**Asset prep (one-time, can be a manual step):**
1. Encode a ~1s, 64x64, single-color mp4 (`ffmpeg -f lavfi -i color=c=black:s=64x64:d=1 -c:v libx264 -t 1 warmup.mp4`).
2. Upload to Vercel Blob via the dashboard or a one-shot script.
3. Set `NEXT_PUBLIC_MODAL_WARMUP_BLOB_URL` in Vercel env.

Total inference cost per warm-up: a single 64x64 frame at 30fps → ~30 frames.
Probably <1s on warm Modal. Cold-start tax (~5–10s) is what we want to absorb.

---

## Section 4 — Open questions / risks for E2/E3/E4

### Open questions

1. **Is `video/mp4;codecs=avc1` actually fragmented on iOS Safari 17+?**
   This is the gating real-device check. If chunks #1+ fail standalone
   playback on iOS but pass on Android Chrome, the plan's option (1)
   "concat sub-clip to one mp4" needs a JS muxer, OR option (2)
   "multi-URL POST" needs `/api/extract` to accept an array.

2. **Latency from Stop → /api/extract queued response on cellular.**
   The spike measures elapsed time including the Vercel Blob upload of the
   3-chunk bundle (~1–2MB). Worth noting in the report so E3 can budget
   for it in the "10–15s coaching latency" target.

3. **Warm-up ping side-effect on Supabase logs.** Each fire-and-forget
   warm-up will hit Railway → Modal → an UPDATE-zero-rows on user_sessions.
   No data corruption, but Railway logs will show `[extract] timing=...`
   lines for warm-up requests. Worth filtering or tagging if log noise
   becomes a problem (`session_id: 'warmup-...'` prefix makes them grep-able).

### Risks worth surfacing

1. **`/api/extract` response is fire-and-poll.** It returns
   `{ status: 'queued' }` synchronously and writes keypoints back to
   Supabase later (`railway-service/main.py:240-243`). The spike test
   doesn't poll; E3 will need to wire the live path through
   `lib/poseExtractionRailway.ts` (or a swing-batch variant) which polls
   `/api/sessions/[id]` until the row flips to `complete`. That polling
   layer assumes a real session row exists — for the per-batch live path,
   E3 needs to decide whether each batch creates a transient session row,
   reuses the parent live session row, or invents a new "extract jobs"
   table. The plan's `lib/liveSwingBatchExtractor.ts` line item implies a
   new extractor — confirm the row-shape decision before E3 starts.

2. **Modal cache-key collision risk.** `modal_inference.py:301-302` keys
   the per-clip cache on `sha256(content[:256KB]) + ":fps=" + sample_fps`.
   For a 3-chunk fMP4 bundle, the first 256KB will *very often* be
   identical across batches in the same session (same MOOV header, same
   first few seconds of stream init). False cache hits would return wrong
   keypoints. Worth E3 testing: record two distinct 3-swing batches in one
   session, confirm the second batch returns *different* keypoints. If it
   doesn't, expand the cache key (e.g. include `sessionId + batchIndex`,
   or extend the hash window past the init segment).

3. **MOTION_PREPASS_MIN_DURATION_SEC = 90s is irrelevant to live batches**
   (well below threshold) but the threshold check uses
   `(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / video_fps`. fMP4 chunked
   blobs sometimes report frame_count=0 to OpenCV until decoded. If
   `duration_sec_est == 0` the prepass is skipped — fine for our case —
   but worth knowing the path.

4. **MediaRecorder mimeType priority.** `hooks/useLiveCapture.ts:118-125`
   already prioritizes `video/mp4;codecs=avc1` first. If real-device
   testing shows iOS prefers a different ordering or rejects the codec
   string, the same change needs to land in both useLiveCapture and any
   new spike code paths. The spike page uses an independent
   `CANDIDATE_MIMES` constant on purpose (so it can probe broader
   support); E2 should keep them in sync once the winning MIME is
   confirmed.

5. **Privacy / consent.** Plan's "Tradeoffs" section flags this. Live
   video frames going to Modal during recording (rather than only after
   Stop) is a posture change. The /live consent flow currently only
   warns about post-Stop upload. E4 should update copy before ship.

### Out of scope for E1 but noticed

- `app/api/extract/route.ts` doesn't validate that `sessionId` is a uuid
  shape (pre-existing — see commit e3dcc9d "Audit fixes: uuid validation"
  for context — that audit added validation elsewhere but not here).
  Probably worth adding when E3 touches the route, defense in depth.
- Railway's `_run_extraction` swallows all exceptions into a
  `supabase.update(status='error')`. For warm-up requests with no matching
  row, the error path also no-ops silently. Might be worth a tiny
  diagnostic counter so we can tell warm-ups apart from real failures in
  Railway logs — but explicitly out of scope for E1.
