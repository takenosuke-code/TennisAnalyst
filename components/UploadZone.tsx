'use client'

import { useRef, useState, useCallback, useEffect } from 'react'
import { usePoseStore } from '@/store'
import { getPoseLandmarker, recordTimestamp } from '@/lib/mediapipe'
import { computeJointAngles } from '@/lib/jointAngles'
import { isFrameConfident, smoothFrames } from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley'] as const

interface UploadZoneProps {
  onComplete?: (blobUrl: string, frames: PoseFrame[]) => void
}

export default function UploadZone({ onComplete }: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const generationRef = useRef(0) // incremented on each new processVideo call to cancel stale runs
  const [dragging, setDragging] = useState(false)
  const [shotType, setShotType] = useState<string>('forehand')
  const [statusMsg, setStatusMsg] = useState('')

  const { setFramesData, setBlobUrl, setLocalVideoUrl, setProcessing, setProgress, isProcessing, reset, setShotType: persistShotType } =
    usePoseStore()

  // Reset only the processing flag when UploadZone mounts fresh (e.g. after back-button).
  // Don't clear framesData/blobUrl/localVideoUrl - those are needed by VideoCanvas.
  useEffect(() => {
    if (isProcessing) {
      setProcessing(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const processVideo = useCallback(
    async (file: File) => {
      // Increment generation; stale loops check this and abort
      const generation = ++generationRef.current
      setProcessing(true)
      setProgress(0)
      setStatusMsg('Uploading video...')

      // 1. Upload to Vercel Blob via API
      const formData = new FormData()
      formData.append('video', file)
      formData.append('shot_type', shotType)

      let blobUrl: string
      try {
        const uploadRes = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        })
        if (!uploadRes.ok) throw new Error('Upload failed')
        const uploadData = await uploadRes.json()
        blobUrl = uploadData.blobUrl
        setBlobUrl(blobUrl)
        if (uploadData.sessionId) usePoseStore.getState().setSessionId(uploadData.sessionId)
      } catch {
        setStatusMsg('Upload failed. Please try again.')
        setProcessing(false)
        return
      }

      setProgress(15)
      setStatusMsg('Loading pose model...')

      // 2. Initialize MediaPipe in browser
      let poseLandmarker: Awaited<ReturnType<typeof getPoseLandmarker>>
      try {
        poseLandmarker = await getPoseLandmarker()
      } catch {
        setStatusMsg('Failed to load pose model. Check your connection.')
        setProcessing(false)
        return
      }

      setProgress(25)
      setStatusMsg('Analyzing pose from video...')

      // 3. Process video frame by frame using a hidden video element
      const videoEl = videoRef.current!
      const objectUrl = URL.createObjectURL(file)
      videoEl.src = objectUrl

      await new Promise<void>((resolve) => {
        videoEl.onloadedmetadata = () => resolve()
      })

      // Wait until the browser has decoded enough data to render a frame.
      // loadedmetadata only guarantees dimensions/duration - not decodable frames.
      if (videoEl.readyState < 3) {
        await new Promise<void>((resolve) => {
          videoEl.oncanplay = () => {
            videoEl.oncanplay = null
            resolve()
          }
        })
      }

      const duration = videoEl.duration
      const fps = 30
      const frameInterval = 1 / fps
      const frames: PoseFrame[] = []
      let frameIndex = 0

      const canvas = document.createElement('canvas')
      canvas.width = videoEl.videoWidth || 640
      canvas.height = videoEl.videoHeight || 360
      const ctx = canvas.getContext('2d')!

      // Seek video to a timestamp and wait for onseeked, with 3s timeout per frame
      const seekToFrame = (time: number): Promise<boolean> =>
        new Promise((resolve) => {
          let settled = false
          const settle = (success: boolean) => {
            if (settled) return
            settled = true
            videoEl.onseeked = null
            resolve(success)
          }
          const timeout = setTimeout(() => settle(false), 3000)
          videoEl.onseeked = () => {
            clearTimeout(timeout)
            settle(true)
          }
          videoEl.currentTime = time
        })

      // Process all frames sequentially; abort if superseded by a new upload
      while (frameIndex * frameInterval <= duration) {
        if (generationRef.current !== generation) break // new upload started

        const currentTime = frameIndex * frameInterval
        const seeked = await seekToFrame(currentTime)

        if (!seeked) {
          // Browser failed to seek to this frame - skip it
          frameIndex++
          continue
        }

        // Wait one animation frame to ensure the decoded video frame is
        // fully composited and ready for canvas drawImage.
        await new Promise<void>((r) => requestAnimationFrame(() => r()))

        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)

        try {
          const ts = currentTime * 1000
          recordTimestamp(ts)
          const result = poseLandmarker.detectForVideo(canvas, ts)
          if (result.landmarks?.[0]?.length) {
            const rawLandmarks = result.landmarks[0]
            const landmarks: Landmark[] = rawLandmarks.map(
              (lm: { x: number; y: number; z?: number; visibility?: number }, id: number) => ({
                id,
                name: `landmark_${id}`,
                x: lm.x,
                y: lm.y,
                z: lm.z ?? 0,
                visibility: lm.visibility ?? 1,
              })
            )

            // Skip frames where the detection is low confidence (bad
            // visibility or tiny bounding box - common during warm-up)
            if (!isFrameConfident(landmarks)) {
              frameIndex++
              continue
            }

            const joint_angles = computeJointAngles(landmarks)
            frames.push({
              frame_index: frameIndex,
              timestamp_ms: currentTime * 1000,
              landmarks,
              joint_angles,
            })
          }
        } catch {
          // Skip frames where detection fails
        }

        frameIndex++
        const pct = 25 + (currentTime / duration) * 65
        setProgress(Math.round(pct))
      }

      // If superseded by a newer upload, bail out and release the processing lock
      if (generationRef.current !== generation) {
        URL.revokeObjectURL(objectUrl)
        setProcessing(false)
        return
      }

      // Post-process: discard warm-up frames and apply EMA smoothing to
      // stabilize jittery landmarks from the first few detections.
      const smoothedFrames = smoothFrames(frames)

      if (smoothedFrames.length === 0) {
        URL.revokeObjectURL(objectUrl)
        setStatusMsg('No pose detected in video. Try a clearer angle with your full body visible.')
        setProcessing(false)
        return
      }

      setProgress(95)
      setStatusMsg('Saving analysis...')

      // 4. Persist keypoints to DB
      const keypointsJson = {
        fps_sampled: fps,
        frame_count: smoothedFrames.length,
        frames: smoothedFrames,
      }

      try {
        const sessionId = usePoseStore.getState().sessionId
        const sessRes = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, blobUrl, shotType, keypointsJson }),
        })
        if (!sessRes.ok) {
          console.error('Failed to save session:', sessRes.status, await sessRes.text())
          setStatusMsg('Warning: analysis complete but failed to save session.')
        }
      } catch (err) {
        console.error('Failed to save session:', err)
        setStatusMsg('Warning: analysis complete but failed to save session.')
      }

      setLocalVideoUrl(objectUrl)
      setFramesData(smoothedFrames)
      persistShotType(shotType)
      setProgress(100)
      setStatusMsg(`Done! Analyzed ${smoothedFrames.length} frames.`)
      setProcessing(false)
      onComplete?.(blobUrl, smoothedFrames)
    },
    [shotType, setFramesData, setBlobUrl, setLocalVideoUrl, setProcessing, setProgress, persistShotType, onComplete]
  )

  const handleFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      setStatusMsg('Please upload a video file.')
      return
    }
    processVideo(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const { progress } = usePoseStore()

  return (
    <div className="space-y-4">
      {/* Shot type selector */}
      <div className="flex gap-2 flex-wrap">
        {SHOT_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => setShotType(type)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium capitalize transition-all ${
              shotType === type
                ? 'bg-emerald-500 text-white'
                : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
            }`}
          >
            {type}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-emerald-400 bg-emerald-500/10'
            : 'border-white/20 hover:border-white/40 hover:bg-white/5'
        } ${isProcessing ? 'cursor-not-allowed opacity-80' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
          }}
        />

        {isProcessing ? (
          <div className="space-y-4">
            <div className="text-4xl">🎾</div>
            <p className="text-white font-medium">{statusMsg}</p>
            <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-white/50 text-sm">{progress}%</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-5xl">🎬</div>
            <p className="text-white text-lg font-medium">
              Drop your swing video here
            </p>
            <p className="text-white/50 text-sm">
              MP4, MOV, WebM · Max 200MB · Select shot type above first
            </p>
            {statusMsg && (
              <p className={`${/fail|error|please/i.test(statusMsg) ? 'text-red-400' : 'text-emerald-400'} text-sm font-medium`}>{statusMsg}</p>
            )}
          </div>
        )}
      </div>

      {/* Hidden video for frame extraction */}
      <video
        ref={videoRef}
        className="hidden"
        muted
        playsInline
        preload="auto"
      />
    </div>
  )
}
