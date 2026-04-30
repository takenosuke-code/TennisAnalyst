-- Public videos bucket for the analyze upload flow.
--
-- Why: the previous flow used Vercel Blob, which doesn't speak the tus
-- resumable upload protocol — large phone uploads on cellular died
-- mid-flight whenever the user closed the tab. Supabase Storage speaks
-- tus natively, so we can survive tab close, network drops, and the
-- 9-minute "user walked home from the courts" window.
--
-- This bucket is public-read so the URL plumbing matches the
-- preexisting blob_url contract (sessions / Railway extractor / video
-- playback all just `fetch()` the URL — anything https works). Writes
-- are open to anon + authenticated because /analyze works for guests
-- (sign-in is only required to *save* a baseline, not to analyze a
-- swing).
--
-- File size limit raised to 500 MB so a raw 4K60 phone clip can land
-- before any client-side transcode pipeline lands. Once the WebCodecs
-- transcode is in place this could come down to ~150 MB.

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'videos',
  'videos',
  true,
  524288000, -- 500 MB
  ARRAY['video/mp4','video/quicktime','video/webm','video/x-matroska']
)
ON CONFLICT (id) DO UPDATE SET
  public = EXCLUDED.public,
  file_size_limit = EXCLUDED.file_size_limit,
  allowed_mime_types = EXCLUDED.allowed_mime_types;

-- INSERT = create the row when tus starts the upload.
-- UPDATE = required for tus PATCH-chunk writes against the existing row.
-- SELECT = required because tus client fetches HEAD on the object during
--           resume. Public-read is also required for the consumers
--           (Railway extractor, browser video playback) to fetch the
--           file by URL.
DROP POLICY IF EXISTS "videos_insert" ON storage.objects;
CREATE POLICY "videos_insert" ON storage.objects
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (bucket_id = 'videos');

DROP POLICY IF EXISTS "videos_update" ON storage.objects;
CREATE POLICY "videos_update" ON storage.objects
  FOR UPDATE
  TO anon, authenticated
  USING (bucket_id = 'videos')
  WITH CHECK (bucket_id = 'videos');

DROP POLICY IF EXISTS "videos_select" ON storage.objects;
CREATE POLICY "videos_select" ON storage.objects
  FOR SELECT
  TO anon, authenticated
  USING (bucket_id = 'videos');
