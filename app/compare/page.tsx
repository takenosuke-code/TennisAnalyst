'use client'

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useState, useRef } from 'react'
import JointTogglePanel from '@/components/JointTogglePanel'
import ProSelector from '@/components/ProSelector'
import { usePoseStore, useJointStore, useComparisonStore } from '@/store'
import { getPoseLandmarker, getMonotonicTimestamp } from '@/lib/mediapipe'
import { computeJointAngles } from '@/lib/jointAngles'
import { isFrameConfident, smoothFrames } from '@/lib/poseSmoothing'
import type { PoseFrame, Landmark } from '@/lib/supabase'
import { getProVideoUrl } from '@/lib/proVideoUrl'

const ComparisonLayout = dynamic(() => import('@/components/ComparisonLayout'), { ssr: false })
const MetricsComparison = dynamic(() => import('@/components/MetricsComparison'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function ComparePage() {
  const { framesData, blobUrl, localVideoUrl } = usePoseStore()
  const { visible, showSkeleton, showTrail } = useJointStore()
  const { activeProSwing, secondaryBlobUrl, secondaryFramesData, setSecondaryBlobUrl, setSecondaryFramesData } =
    useComparisonStore()

  const [compareMode, setCompareMode] = useState<'pro' | 'custom'>('pro')
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const hiddenVideoRef = useRef<HTMLVideoElement>(null)
  const secondaryGenerationRef = useRef(0)

  const userVideoSrc = localVideoUrl || blobUrl
  const hasUserVideo = userVideoSrc && framesData.length > 0
  const hasProData = activeProSwing && (activeProSwing.keypoints_json?.frames?.length ?? 0) > 0
  const hasSecondary = secondaryBlobUrl && secondaryFramesData.length > 0

  const canCompare =
    hasUserVideo &&
    ((compareMode === 'pro' && hasProData) || (compareMode === 'custom' && hasSecondary))

  const handleSecondaryFile = (file: File) => {
    if (!file.type.startsWith('video/')) return
    processSecondaryVideo(file)
  }

  const handleSecondaryDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleSecondaryFile(file)
  }

  const processSecondaryVideo = async (file: File) => {
    const generation = ++secondaryGenerationRef.current
    setProcessing(true)
    setProgress(0)

    // Capture the previous URL so we can revoke it after the new one is ready
    const previousBlobUrl = secondaryBlobUrl

    const objectUrl = URL.createObjectURL(file)
    const videoEl = hiddenVideoRef.current!
    videoEl.src = objectUrl

    await new Promise<void>((r) => { videoEl.onloadedmetadata = () => r() })

    if (videoEl.readyState < 3) {
      await new Promise<void>((resolve) => {
        videoEl.oncanplay = () => { videoEl.oncanplay = null; resolve() }
      })
    }

    const poseLandmarker = await getPoseLandmarker()
    const canvas = document.createElement('canvas')
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 360
    const ctx = canvas.getContext('2d')!

    const duration = videoEl.duration
    const fps = 30
    const frames: PoseFrame[] = []
    let frameIndex = 0

    // Seek helper with 3s timeout so a bad frame doesn't hang forever
    const seekTo = (time: number): Promise<boolean> =>
      new Promise((resolve) => {
        let settled = false
        const settle = (ok: boolean) => {
          if (settled) return
          settled = true
          videoEl.onseeked = null
          resolve(ok)
        }
        const t = setTimeout(() => settle(false), 3000)
        videoEl.onseeked = () => { clearTimeout(t); settle(true) }
        videoEl.currentTime = time
      })

    while (frameIndex * (1 / fps) <= duration) {
      if (secondaryGenerationRef.current !== generation) {
        URL.revokeObjectURL(objectUrl)
        setProcessing(false)
        return
      }

      const currentTime = frameIndex * (1 / fps)
      const seeked = await seekTo(currentTime)
      if (!seeked) { frameIndex++; continue }

      await new Promise<void>((r) => requestAnimationFrame(() => r()))
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)
      try {
        // Use monotonic timestamp so the shared PoseLandmarker singleton
        // always sees strictly increasing values, even if it was previously
        // used on the analyze page with higher timestamps.
        const monoTs = getMonotonicTimestamp(currentTime * 1000)
        const result = poseLandmarker.detectForVideo(canvas, monoTs)
        if (result.landmarks?.[0]?.length) {
          const landmarks: Landmark[] = result.landmarks[0].map((lm: { x: number; y: number; z?: number; visibility?: number }, id: number) => ({
            id, name: `landmark_${id}`,
            x: lm.x, y: lm.y, z: lm.z ?? 0, visibility: lm.visibility ?? 1,
          }))

          // Skip low-confidence detections (warm-up artifacts)
          if (!isFrameConfident(landmarks)) {
            frameIndex++
            continue
          }

          frames.push({
            frame_index: frameIndex,
            timestamp_ms: currentTime * 1000,
            landmarks,
            joint_angles: computeJointAngles(landmarks),
          })
        }
      } catch { /* skip */ }

      frameIndex++
      setProgress(Math.round((currentTime / duration) * 100))
    }

    // Post-process: discard warm-up frames and apply EMA smoothing
    const smoothedFrames = smoothFrames(frames)

    // Use the same objectUrl for playback (not a second one)
    setSecondaryBlobUrl(objectUrl)
    setSecondaryFramesData(smoothedFrames)
    setProcessing(false)
    setProgress(100)

    // Revoke previous URL now that the new one is set
    if (previousBlobUrl) URL.revokeObjectURL(previousBlobUrl)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Compare Swings</h1>
        <p className="text-white/50">
          Overlay or side-by-side compare your swing against a pro or another video.
        </p>
      </div>

      {!hasUserVideo && (
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center mb-8">
          <div className="text-4xl mb-3">🎾</div>
          <p className="text-white font-medium mb-2">No video analyzed yet</p>
          <p className="text-white/40 text-sm mb-6">
            Go to the Analyze page first to upload and process your swing.
          </p>
          <Link
            href="/analyze"
            className="inline-block px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-xl transition-colors"
          >
            Analyze My Swing
          </Link>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main comparison area */}
        <div className="lg:col-span-2 space-y-6">
          {/* Compare mode toggle */}
          <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit">
            <button
              onClick={() => setCompareMode('pro')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                compareMode === 'pro' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
              }`}
            >
              vs Pro Player
            </button>
            <button
              onClick={() => setCompareMode('custom')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                compareMode === 'custom' ? 'bg-white text-black' : 'text-white/50 hover:text-white'
              }`}
            >
              vs Custom Video
            </button>
          </div>

          {/* Custom video upload */}
          {compareMode === 'custom' && !hasSecondary && (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleSecondaryDrop}
              onClick={() => !processing && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
                dragging
                  ? 'border-emerald-400 bg-emerald-500/10'
                  : 'border-white/20 hover:border-white/40 hover:bg-white/5'
              } ${processing ? 'cursor-not-allowed opacity-80' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) processSecondaryVideo(file)
                }}
              />
              {processing ? (
                <div className="space-y-3">
                  <p className="text-white">Processing second video... {progress}%</p>
                  <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-emerald-400 rounded-full transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-4xl mb-3">🎬</div>
                  <p className="text-white font-medium">Upload second video to compare</p>
                  <p className="text-white/40 text-sm mt-1">MP4, MOV, WebM</p>
                </div>
              )}
            </div>
          )}

          {/* Comparison view */}
          {canCompare && hasUserVideo && (
            <div className="rounded-2xl border border-white/10 overflow-hidden bg-black">
              <div className="p-3 border-b border-white/5">
                <p className="text-sm text-white/60">
                  {compareMode === 'pro'
                    ? `Your swing vs ${activeProSwing?.pros?.name ?? 'Pro'} · ${activeProSwing?.shot_type}`
                    : 'Side by Side Comparison'}
                </p>
              </div>
              <div className="p-3">
                <ComparisonLayout
                  userBlobUrl={userVideoSrc!}
                  userFrames={framesData}
                  proFrames={
                    compareMode === 'pro'
                      ? (activeProSwing?.keypoints_json?.frames ?? [])
                      : secondaryFramesData
                  }
                  proVideoUrl={
                    compareMode === 'pro'
                      ? (getProVideoUrl(activeProSwing) ?? '')
                      : (secondaryBlobUrl ?? '')
                  }
                  proName={
                    compareMode === 'pro'
                      ? (activeProSwing?.pros?.name ?? 'Pro')
                      : 'Video 2'
                  }
                />
              </div>
            </div>
          )}

          {/* Key metrics (kinetic chain, mistakes, coaching cues) */}
          {canCompare && hasUserVideo && (
            <MetricsComparison
              userFrames={framesData}
              proFrames={
                compareMode === 'pro'
                  ? (activeProSwing?.keypoints_json?.frames ?? [])
                  : secondaryFramesData
              }
              shotType={
                compareMode === 'pro'
                  ? (activeProSwing?.shot_type ?? null)
                  : null
              }
            />
          )}

          {/* AI Coach chat */}
          {hasUserVideo && (
            <LLMCoachingPanel
              proSwing={compareMode === 'pro' ? activeProSwing : null}
              compareMode={compareMode}
            />
          )}
        </div>

        {/* Right sidebar */}
        <div className="space-y-4">
          <JointTogglePanel />

          {compareMode === 'pro' && (
            <div className="rounded-xl bg-white/5 border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Select Pro</h3>
              <ProSelector />
            </div>
          )}
        </div>
      </div>

      {/* Hidden video for secondary processing */}
      <video ref={hiddenVideoRef} className="hidden" muted playsInline preload="auto" />
    </div>
  )
}
