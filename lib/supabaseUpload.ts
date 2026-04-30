'use client'

import * as tus from 'tus-js-client'
import { createClient } from '@/lib/supabase/client'

/*
 * Resumable video upload to Supabase Storage via tus.
 *
 * Why this exists: the old Vercel-Blob path used in-process multipart
 * which lost state when the tab closed. Tus persists upload progress
 * to localStorage (via tus-js-client's URL-storage default) so a user
 * who switches apps, locks their phone, or kills the tab mid-upload
 * can resume from the last completed 6 MB chunk on return. That's the
 * actual UX win — the bandwidth cost doesn't change.
 *
 * Public URL contract: callers get back a plain https URL pointing at
 * the public-read bucket, which slots straight into the existing
 * blob_url plumbing (Railway extractor, video playback, comparison
 * page) without any other changes.
 */

const BUCKET = 'videos'
// Required by Supabase tus implementation — anything other than 6 MB
// is rejected. See https://supabase.com/docs/guides/storage/uploads/resumable-uploads
const CHUNK_SIZE = 6 * 1024 * 1024

export interface UploadCallbacks {
  // Called with a 0..1 fraction. The caller maps this onto whatever
  // progress UI band makes sense.
  onProgress?: (fraction: number) => void
  // Resolves the public URL once the upload finishes.
  onSuccess?: (publicUrl: string) => void
  onError?: (err: Error) => void
}

export interface UploadHandle {
  // Pauses the upload; resume() picks up from the last chunk.
  abort: () => void
  // Fire-and-forget reference to the underlying tus.Upload so
  // callers can call .start() / .abort() / .findPreviousUploads()
  // if they need finer control.
  upload: tus.Upload
}

/**
 * Upload a File to Supabase Storage using tus. Returns a handle so the
 * caller can abort. The promise resolves with the public URL.
 *
 * Auth: tus-js-client sends the user's Supabase access_token (or the
 * anon key when the user is signed out — public bucket allows anon
 * writes per migration 011). Either path satisfies the bucket RLS.
 */
export async function uploadVideoResumable(
  file: File,
  objectPath: string,
  callbacks: UploadCallbacks = {},
): Promise<UploadHandle> {
  const supabase = createClient()
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

  // Use the user's session token when signed in so RLS sees an
  // authenticated principal; fall back to the anon key (still
  // permitted by the videos bucket policy) for guests on /analyze.
  const { data: { session } } = await supabase.auth.getSession()
  const accessToken = session?.access_token ?? anonKey

  return new Promise<UploadHandle>((resolve, reject) => {
    const upload = new tus.Upload(file, {
      endpoint: `${supabaseUrl}/storage/v1/upload/resumable`,
      retryDelays: [0, 1500, 4000, 10000, 20000],
      headers: {
        authorization: `Bearer ${accessToken}`,
        // x-upsert lets a re-upload at the same path overwrite cleanly.
        // Without this a retry of the same fingerprint after a server
        // glitch would 409 forever.
        'x-upsert': 'true',
      },
      uploadDataDuringCreation: true,
      removeFingerprintOnSuccess: true,
      metadata: {
        bucketName: BUCKET,
        objectName: objectPath,
        contentType: file.type || 'video/mp4',
        cacheControl: '3600',
      },
      chunkSize: CHUNK_SIZE,
      onError: (err) => {
        callbacks.onError?.(err as Error)
        reject(err as Error)
      },
      onProgress: (sent, total) => {
        if (total > 0) callbacks.onProgress?.(sent / total)
      },
      onSuccess: () => {
        const publicUrl =
          `${supabaseUrl}/storage/v1/object/public/${BUCKET}/${objectPath}`
        callbacks.onSuccess?.(publicUrl)
      },
    })

    // Resume any prior in-flight upload for the same File fingerprint
    // before kicking off a fresh one. The fingerprint is content-based
    // (file size + name + mtime) so reopening the same file after a
    // tab close picks up where we left off.
    upload.findPreviousUploads().then((prevs) => {
      if (prevs.length > 0) {
        upload.resumeFromPreviousUpload(prevs[0])
      }
      upload.start()
      resolve({
        abort: () => upload.abort(),
        upload,
      })
    }).catch((err) => {
      // Fingerprint lookup failed — proceed with a fresh upload.
      upload.start()
      resolve({
        abort: () => upload.abort(),
        upload,
      })
      void err
    })
  })
}

/**
 * Build a sanitized object path for a fresh upload. Mirrors the prior
 * Vercel Blob layout (`videos/<timestamp>-<safe-filename>`) so any
 * downstream URL parsing that keyed off the path still works.
 */
export function buildObjectPath(filename: string): string {
  const safe = filename
    .split(/[\\/]/)
    .pop()!
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .slice(0, 100) || 'upload'
  return `${Date.now()}-${safe}`
}
