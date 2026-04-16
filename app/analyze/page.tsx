'use client'

import { useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import JointTogglePanel from '@/components/JointTogglePanel'
import ProSelector from '@/components/ProSelector'
import SwingSelector from '@/components/SwingSelector'
import { usePoseStore, useJointStore, useComparisonStore } from '@/store'
import { detectSwings } from '@/lib/jointAngles'
import type { SwingSegment } from '@/lib/jointAngles'
import type { PoseFrame } from '@/lib/supabase'

// Dynamic import to avoid SSR issues with MediaPipe WASM
const UploadZone = dynamic(() => import('@/components/UploadZone'), { ssr: false })
const VideoCanvas = dynamic(() => import('@/components/VideoCanvas'), { ssr: false })
const LLMCoachingPanel = dynamic(() => import('@/components/LLMCoachingPanel'), { ssr: false })

export default function AnalyzePage() {
  const { framesData, blobUrl, localVideoUrl } = usePoseStore()
  const { visible, showSkeleton, showTrail } = useJointStore()
  const { activeProSwing } = useComparisonStore()
  const [done, setDone] = useState(false)
  const [allFrames, setAllFrames] = useState<PoseFrame[]>([])
  const [selectedSwing, setSelectedSwing] = useState<number | null>(null)

  const swings = useMemo(() => detectSwings(allFrames), [allFrames])
  const hasMultipleSwings = swings.length > 1

  const handleComplete = (_url: string, frames: PoseFrame[]) => {
    setAllFrames(frames)
    const detected = detectSwings(frames)
    if (detected.length > 1) {
      setSelectedSwing(1)
    } else {
      setSelectedSwing(null)
    }
    setDone(true)
  }

  const handleSwingSelect = (seg: SwingSegment) => {
    setSelectedSwing(seg.index)
  }

  // The frames to send to the LLM - either the selected swing or all frames
  const analysisFrames = selectedSwing && swings[selectedSwing - 1]
    ? swings[selectedSwing - 1].frames
    : framesData

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Analyze Your Swing</h1>
        <p className="text-white/50">
          Upload a video to extract pose landmarks and get AI coaching feedback.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: video + upload */}
        <div className="lg:col-span-2 space-y-6">
          {!done ? (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Upload Video</h2>
              <UploadZone onComplete={handleComplete} />
            </div>
          ) : (
            blobUrl && (
              <div className="rounded-2xl border border-white/10 bg-black overflow-hidden">
                <div className="p-3 border-b border-white/5 flex items-center justify-between">
                  <span className="text-sm font-medium text-white">Your Swing</span>
                  <button
                    onClick={() => setDone(false)}
                    className="text-xs text-white/40 hover:text-white"
                  >
                    Upload different video
                  </button>
                </div>
                <VideoCanvas
                  src={localVideoUrl || blobUrl}
                  framesData={framesData}
                  visible={visible}
                  showSkeleton={showSkeleton}
                  showTrail={showTrail}
                  showControls
                  className="p-2"
                />
              </div>
            )
          )}

          {/* Swing selector for longer videos */}
          {done && hasMultipleSwings && (
            <SwingSelector
              allFrames={allFrames}
              selectedIndex={selectedSwing}
              onSelect={handleSwingSelect}
            />
          )}

          {/* LLM coaching */}
          {done && (
            <LLMCoachingPanel proSwing={activeProSwing} frames={analysisFrames} />
          )}

          {/* Compare CTA */}
          {done && (
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 flex items-center justify-between">
              <div>
                <p className="text-white font-medium text-sm">Ready to compare?</p>
                <p className="text-white/50 text-xs">
                  Select a pro below, then go to the Compare page for overlay view.
                </p>
              </div>
              <Link
                href="/compare"
                className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-semibold rounded-lg transition-colors flex-shrink-0"
              >
                Compare →
              </Link>
            </div>
          )}
        </div>

        {/* Right: controls */}
        <div className="space-y-4">
          <JointTogglePanel />

          <div className="rounded-xl bg-white/5 border border-white/10 p-4">
            <h3 className="text-sm font-semibold text-white mb-3">Compare Against</h3>
            <ProSelector />
          </div>

          {framesData.length > 0 && (
            <div className="rounded-xl bg-white/5 border border-white/10 p-4">
              <h3 className="text-sm font-semibold text-white mb-2">Analysis Stats</h3>
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between text-white/60">
                  <span>Frames analyzed</span>
                  <span className="text-white font-mono">{framesData.length}</span>
                </div>
                <div className="flex justify-between text-white/60">
                  <span>Duration</span>
                  <span className="text-white font-mono">
                    {((framesData[framesData.length - 1]?.timestamp_ms ?? 0) / 1000).toFixed(1)}s
                  </span>
                </div>
                <div className="flex justify-between text-white/60">
                  <span>Joint data</span>
                  <span className="text-emerald-400 font-medium">Ready</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
