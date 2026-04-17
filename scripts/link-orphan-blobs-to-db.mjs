#!/usr/bin/env node
// Backfill: find pro-video blobs in Vercel Blob that have no matching
// pro_swings row, parse their filenames, and insert the missing rows.
//
//   node scripts/link-orphan-blobs-to-db.mjs              # dry run
//   node scripts/link-orphan-blobs-to-db.mjs --apply      # commit
//
// Filename convention (produced by /admin/tag-clips):
//   <pro_slug>_<shot_type>_<camera_angle>_<timestamp>.mp4
//   e.g. carlos_alcaraz_forehand_behind_1776374463938.mp4
//        novak_djokovic_backhand_court_level_1776383254038.mp4
//
// Requires .env to contain:
//   NEXT_PUBLIC_SUPABASE_URL
//   SUPABASE_SERVICE_KEY    ← service-role key, bypasses RLS on pro_swings
//   BLOB_READ_WRITE_TOKEN   ← the NEW public store's token

import { createClient } from '@supabase/supabase-js'
import { list } from '@vercel/blob'
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

function loadEnvFile(path) {
  if (!existsSync(path)) return
  const raw = readFileSync(path, 'utf8')
  for (const line of raw.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/)
    if (!m) continue
    let val = m[2]
    if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1)
    process.env[m[1]] = val
  }
}
loadEnvFile(join(process.cwd(), '.env'))

const apply = process.argv.includes('--apply')

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.SUPABASE_SERVICE_KEY
const blobToken = process.env.BLOB_READ_WRITE_TOKEN
if (!supabaseUrl || !supabaseKey) {
  console.error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_KEY')
  process.exit(1)
}
if (!blobToken) {
  console.error('Missing BLOB_READ_WRITE_TOKEN — needed to list the blob store')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseKey)

const SHOT_TYPES = ['forehand', 'backhand', 'serve', 'volley']
const CAMERA_ANGLES = ['side', 'behind', 'front', 'court_level']

// Filename typos → canonical pro name. Add entries if more slip through.
const NAME_ALIASES = {
  'novak djokavic': 'Novak Djokovic',
}

function titleCase(s) {
  return s
    .split(' ')
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ')
}

// Parse "<pro>_<shot>_<angle>_<timestamp>.mp4" where <angle> may be multi-token
// ("court_level"). Returns null if the filename doesn't conform.
function parseFilename(filename) {
  const base = filename.replace(/\.mp4$/i, '')
  const parts = base.split('_')
  if (parts.length < 4) return null

  const shotIdx = parts.findIndex((p) => SHOT_TYPES.includes(p))
  if (shotIdx < 1) return null

  const proSlug = parts.slice(0, shotIdx).join(' ')
  const shotType = parts[shotIdx]
  // Timestamp is the trailing numeric token; everything between shot and
  // timestamp is the camera angle (re-joined with underscore).
  const last = parts[parts.length - 1]
  if (!/^\d+$/.test(last)) return null
  const cameraAngle = parts.slice(shotIdx + 1, -1).join('_')
  if (!CAMERA_ANGLES.includes(cameraAngle)) return null

  const aliased = NAME_ALIASES[proSlug.toLowerCase()]
  const proName = aliased ?? titleCase(proSlug)
  return {
    proName,
    proNameWasAliased: Boolean(aliased),
    rawProSlug: proSlug,
    shotType,
    cameraAngle,
    timestampMs: Number(last),
  }
}

async function main() {
  console.log(`\nLinking orphan pro-video blobs → pro_swings  (${apply ? 'APPLY' : 'dry run'})\n`)

  // 1. List every blob in the store. Historical uploads from /admin/tag-clips
  //    went under pro-videos/<file>, but the 9 files uploaded by hand to the
  //    new public store sit at the root. parseFilename() rejects anything that
  //    doesn't match <pro>_<shot>_<angle>_<ts>.mp4, so listing the whole store
  //    is safe.
  const { blobs } = await list({ token: blobToken })
  console.log(`Found ${blobs.length} blob(s) in store\n`)

  // 2. Load existing pro_swings so we can match by filename and re-point
  //    video_url at the new public blob URL. Most rows here were created
  //    by /admin/tag-clips against the old PRIVATE store, so their URLs
  //    now 404 even though the row and filename are correct.
  const { data: existingSwings, error: swErr } = await supabase
    .from('pro_swings')
    .select('id, video_url')
  if (swErr) throw swErr
  const swingByFilename = new Map()
  for (const s of existingSwings ?? []) {
    const fn = (s.video_url ?? '').split('?')[0].split('/').pop()
    if (fn) swingByFilename.set(fn, s)
  }

  // 3. Load all pros for name lookup.
  const { data: allPros, error: prosErr } = await supabase.from('pros').select('id, name')
  if (prosErr) throw prosErr
  const prosByLowerName = new Map((allPros ?? []).map((p) => [p.name.toLowerCase(), p]))

  let plannedInsert = 0
  let plannedUpdate = 0
  let alreadyCurrent = 0
  let skippedUnparsed = 0
  let skippedMissingPro = 0
  let inserted = 0
  let updated = 0
  let failed = 0

  for (const blob of blobs) {
    const filename = blob.pathname.split('/').pop() ?? ''
    const label = filename

    const parsed = parseFilename(filename)
    if (!parsed) {
      console.log(`  SKIP  ${label}  — filename doesn't match <pro>_<shot>_<angle>_<ts>.mp4`)
      skippedUnparsed++
      continue
    }

    const existing = swingByFilename.get(filename)

    // Case A: pro_swings row already exists — re-point video_url if stale.
    if (existing) {
      if (existing.video_url === blob.url) {
        console.log(`  OK    ${label}  — already pointing at new public blob`)
        alreadyCurrent++
        continue
      }
      if (!apply) {
        console.log(
          `  PLAN  ${label}  → update pro_swings.${existing.id}.video_url to new blob URL`,
        )
        plannedUpdate++
        continue
      }
      try {
        const { error: updErr } = await supabase
          .from('pro_swings')
          .update({ video_url: blob.url })
          .eq('id', existing.id)
        if (updErr) throw updErr
        console.log(`  UPD   ${label}  → pro_swings.${existing.id}.video_url updated`)
        updated++
      } catch (err) {
        console.error(`  FAIL  ${label}  — ${err.message ?? err}`)
        failed++
      }
      continue
    }

    // Case B: orphan blob — insert a new pro_swings row.
    const pro = prosByLowerName.get(parsed.proName.toLowerCase())
    if (!pro) {
      console.log(
        `  SKIP  ${label}  — no pros row for "${parsed.proName}". Add the pro first, then re-run.`,
      )
      skippedMissingPro++
      continue
    }

    const aliasNote = parsed.proNameWasAliased
      ? `  ⚠ filename slug "${parsed.rawProSlug}" → canonical "${parsed.proName}"`
      : ''

    if (!apply) {
      console.log(
        `  PLAN  ${label}  → INSERT pro=${pro.name}  shot=${parsed.shotType}  angle=${parsed.cameraAngle}${aliasNote}`,
      )
      plannedInsert++
      continue
    }

    try {
      const { data: swing, error: insErr } = await supabase
        .from('pro_swings')
        .insert({
          pro_id: pro.id,
          shot_type: parsed.shotType,
          video_url: blob.url,
          keypoints_json: { fps_sampled: 30, frame_count: 0, frames: [] },
          fps: 30,
          duration_ms: null,
          phase_labels: {},
          metadata: {
            camera_angle: parsed.cameraAngle,
            source: 'blob_backfill',
            label: `${parsed.shotType}_${parsed.cameraAngle}`,
            original_filename: filename,
          },
        })
        .select('id')
        .single()
      if (insErr || !swing) throw insErr ?? new Error('no row returned')
      console.log(
        `  INS   ${label}  → pro_swings.id=${swing.id}  pro=${pro.name}${aliasNote}`,
      )
      inserted++
    } catch (err) {
      console.error(`  FAIL  ${label}  — ${err.message ?? err}`)
      failed++
    }
  }

  console.log(
    `\nDone.  ` +
      (apply
        ? `inserted=${inserted}  updated=${updated}  `
        : `planned_insert=${plannedInsert}  planned_update=${plannedUpdate}  `) +
      `already_current=${alreadyCurrent}  unparsed=${skippedUnparsed}  ` +
      `missing_pro=${skippedMissingPro}  failed=${failed}`,
  )
  if (!apply) console.log('\nThis was a dry run. Re-run with --apply to commit.')
  if (skippedMissingPro > 0) {
    console.log(
      '\nNote: orphan blobs with no matching pros entry were skipped. Create those pros ' +
        '(via /admin/tag-clips or direct insert) and re-run.',
    )
  }
}

main().catch((err) => {
  console.error('\nBackfill failed:', err.message ?? err)
  process.exit(1)
})
