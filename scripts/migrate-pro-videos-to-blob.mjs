#!/usr/bin/env node
// One-off migration: upload every pro_swing whose video_url points at
// /pro-videos/<file>.mp4 (local only) to Vercel Blob, then update the DB
// row with the blob URL so the videos play on any device.
//
//   node scripts/migrate-pro-videos-to-blob.mjs              # dry run
//   node scripts/migrate-pro-videos-to-blob.mjs --apply      # commit
//
// Requires .env (in project root) to contain:
//   NEXT_PUBLIC_SUPABASE_URL
//   SUPABASE_SERVICE_KEY    ← service-role key, bypasses RLS on pro_swings
//   BLOB_READ_WRITE_TOKEN   ← must be the NEW public store's token

import { createClient } from '@supabase/supabase-js'
import { put } from '@vercel/blob'
import { readFileSync, existsSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'

// Track which source ultimately wins for each env var so we can tell the user.
const envSource = {}

function loadEnvFile(path, label) {
  if (!existsSync(path)) return
  const raw = readFileSync(path, 'utf8')
  for (const line of raw.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/)
    if (!m) continue
    const key = m[1]
    let val = m[2]
    if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1)
    // File is authoritative — always override so edits actually take effect
    // even if the shell has an older value exported.
    process.env[key] = val
    envSource[key] = label
  }
}

// Capture shell-provided values before .env overwrites them (for reporting).
for (const k of ['BLOB_READ_WRITE_TOKEN', 'NEXT_PUBLIC_SUPABASE_URL', 'SUPABASE_SERVICE_KEY']) {
  if (process.env[k]) envSource[k] = 'shell/export'
}

loadEnvFile(join(process.cwd(), '.env'), '.env')

const apply = process.argv.includes('--apply')

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.SUPABASE_SERVICE_KEY
const blobToken = process.env.BLOB_READ_WRITE_TOKEN
if (!supabaseUrl || !supabaseKey) {
  console.error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_KEY')
  process.exit(1)
}
if (!blobToken) {
  console.error('Missing BLOB_READ_WRITE_TOKEN — needed to upload to Vercel Blob')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseKey)
const PROJECT_ROOT = process.cwd()

// Non-reversible hash of the token so the user can sanity-check which one
// the script picked up without the script ever printing the value.
function fingerprint(s) {
  if (!s) return '(empty)'
  const hash = createHash('sha256').update(s).digest('hex').slice(0, 8)
  return `len=${s.length}  sha256:${hash}`
}

async function main() {
  console.log(`\nMigrating local pro-videos → Vercel Blob  (${apply ? 'APPLY' : 'dry run'})`)
  console.log(`  BLOB_READ_WRITE_TOKEN  source=${envSource.BLOB_READ_WRITE_TOKEN ?? '(none)'}  ${fingerprint(blobToken)}\n`)

  const { data: swings, error } = await supabase
    .from('pro_swings')
    .select('id, video_url, pros(name)')
  if (error) throw error

  if (!swings || swings.length === 0) {
    console.log('No pro_swings rows at all. Nothing to migrate.')
    return
  }

  // Match by filename suffix. video_url might be:
  //   /pro-videos/<file>.mp4                (local)
  //   https://<old>.private.blob.vercel-storage.com/pro-videos/<file>.mp4
  //   https://<old>.public.blob.vercel-storage.com/pro-videos/<file>.mp4
  // In every case, the filename is the last path segment.
  const rows = swings
    .map((s) => {
      const url = s.video_url ?? ''
      const filename = url.split('?')[0].split('/').pop() ?? ''
      return { ...s, filename }
    })
    .filter((s) => s.filename && s.filename.endsWith('.mp4'))

  console.log(`Found ${rows.length} swing(s) with mp4 filenames.\n`)

  let ok = 0
  let skipped = 0
  let failed = 0

  for (const swing of rows) {
    const filename = swing.filename
    const localPath = join(PROJECT_ROOT, 'public', 'pro-videos', filename)
    const proName = swing.pros?.name ?? '(unknown)'
    const label = `${swing.id}  ${proName}  ${filename}`

    if ((swing.video_url ?? '').includes('.public.blob.vercel-storage.com')) {
      console.log(`  SKIP  ${label}  — already on a public blob store`)
      skipped++
      continue
    }

    if (!existsSync(localPath)) {
      console.log(`  SKIP  ${label}  — local file missing: ${localPath}`)
      skipped++
      continue
    }

    if (!apply) {
      console.log(`  PLAN  ${label}  — would upload (${sizeOf(localPath)})`)
      ok++
      continue
    }

    try {
      const buf = readFileSync(localPath)
      const blob = await put(`pro-videos/${filename}`, buf, {
        access: 'public',
        contentType: 'video/mp4',
        token: blobToken,
        // If we re-run the migration, don't collide on name.
        addRandomSuffix: false,
        allowOverwrite: true,
      })
      const { error: updErr } = await supabase
        .from('pro_swings')
        .update({ video_url: blob.url })
        .eq('id', swing.id)
      if (updErr) throw updErr
      console.log(`  OK    ${label}  → ${blob.url}`)
      ok++
    } catch (err) {
      console.error(`  FAIL  ${label}  — ${err.message ?? err}`)
      failed++
    }
  }

  console.log(`\nDone. ok=${ok}  skipped=${skipped}  failed=${failed}`)
  if (!apply) console.log('This was a dry run. Re-run with --apply to upload.')
}

function sizeOf(path) {
  const size = statSync(path).size
  return size > 1_000_000 ? `${(size / 1_000_000).toFixed(1)}MB` : `${(size / 1000).toFixed(0)}KB`
}

main().catch((err) => {
  console.error('\nMigration failed:', err.message ?? err)
  process.exit(1)
})
