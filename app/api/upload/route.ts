import { handleUpload, type HandleUploadBody } from '@vercel/blob/client'
import { NextRequest, NextResponse } from 'next/server'

// Vercel serverless functions cap request bodies at 4.5MB on Hobby / ~100MB on
// Pro, which is smaller than a real swing clip. So the browser uploads the
// file directly to Vercel Blob; this route only hands out a short-lived signed
// token. The user_sessions row is created later by /api/sessions once the
// client has extracted keypoints.
export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: HandleUploadBody
  try {
    body = (await request.json()) as HandleUploadBody
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: ['video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska'],
        maximumSizeInBytes: 200 * 1024 * 1024,
        addRandomSuffix: true,
      }),
      onUploadCompleted: async () => {
        // No DB write here. Blob's completion webhook can't be awaited by the
        // browser anyway, so session creation is deferred to /api/sessions
        // which the client calls after running MediaPipe on the clip.
      },
    })
    return NextResponse.json(jsonResponse)
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Upload token failed'
    return NextResponse.json({ error: msg }, { status: 400 })
  }
}
