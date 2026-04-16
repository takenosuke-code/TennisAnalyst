'use client'

import { useEffect, useRef } from 'react'
import dynamic from 'next/dynamic'
import { useProLibraryStore } from '@/store'

const ProBrowser = dynamic(() => import('@/components/ProBrowser'), { ssr: false })
const ProSwingViewer = dynamic(() => import('@/components/ProSwingViewer'), { ssr: false })
const ProSwingAnalysis = dynamic(() => import('@/components/ProSwingAnalysis'), { ssr: false })
const ProCoachChat = dynamic(() => import('@/components/ProCoachChat'), { ssr: false })

export default function ProsPage() {
  const selectedSwing = useProLibraryStore((s) => s.selectedSwing)
  const viewerColRef = useRef<HTMLDivElement>(null)

  // Scroll the viewer column to top whenever a new swing is selected
  useEffect(() => {
    if (viewerColRef.current) {
      viewerColRef.current.scrollTop = 0
    }
  }, [selectedSwing?.id])

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white mb-2">Pro Library</h1>
        <p className="text-white/50">
          Study professional technique frame-by-frame and get AI coaching insights.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Col 1: Pro Browser — scrolls independently */}
        <div className="lg:h-[calc(100vh-180px)] lg:overflow-y-auto">
          <ProBrowser />
        </div>

        {/* Col 2-3: Swing Viewer + Analysis — scrolls independently */}
        <div ref={viewerColRef} className="lg:col-span-2 lg:h-[calc(100vh-180px)] lg:overflow-y-auto space-y-6">
          {selectedSwing ? (
            <>
              <ProSwingViewer />
              <ProSwingAnalysis />
              <ProCoachChat />
            </>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
              <div className="text-4xl mb-3">🎾</div>
              <p className="text-white font-medium mb-2">Select a pro swing to start studying</p>
              <p className="text-white/40 text-sm">
                Browse the pros on the left and pick a swing to view.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
