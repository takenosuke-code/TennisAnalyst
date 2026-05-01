/**
 * cleanup-blobs.ts — bulk-delete old Vercel Blob videos to free quota.
 *
 * The /analyze upload pipeline stores raw videos in Vercel Blob under the
 * `videos/` prefix. The Hobby plan caps at 1 GB total stored; every test
 * upload sticks around forever unless deleted. Once you cross the cap,
 * new uploads 400 with "Storage quota exceeded for Hobby plan" — which
 * is the exact error the user hit at /analyze.
 *
 * Run with:
 *   BLOB_READ_WRITE_TOKEN=vercel_blob_rw_... npx tsx scripts/cleanup-blobs.ts [--olderThanDays N] [--dry-run]
 *
 * Defaults to deleting EVERYTHING under `videos/` (no age filter, no
 * dry-run). Pass --dry-run to see what would be deleted without
 * deleting. Pass --olderThanDays N to keep blobs newer than N days.
 *
 * Get BLOB_READ_WRITE_TOKEN from Vercel dashboard:
 *   Project → Storage → your Blob store → .env.local tab → copy the token.
 */

import { list, del } from '@vercel/blob'

const token = process.env.BLOB_READ_WRITE_TOKEN
if (!token) {
  console.error(
    'Missing BLOB_READ_WRITE_TOKEN in env. Get it from Vercel → Storage → Blob → .env.local tab.',
  )
  process.exit(1)
}

const args = process.argv.slice(2)
const dryRun = args.includes('--dry-run')
const olderThanIdx = args.indexOf('--olderThanDays')
const olderThanDays =
  olderThanIdx >= 0 ? parseInt(args[olderThanIdx + 1] ?? '0', 10) : 0

if (Number.isNaN(olderThanDays) || olderThanDays < 0) {
  console.error('--olderThanDays must be a non-negative integer')
  process.exit(1)
}

const cutoffMs = olderThanDays > 0 ? Date.now() - olderThanDays * 24 * 60 * 60 * 1000 : null

async function main() {
  console.log(`[cleanup] mode: ${dryRun ? 'DRY RUN' : 'DELETE'}`)
  console.log(`[cleanup] age filter: ${cutoffMs ? `older than ${olderThanDays} days` : 'none (all blobs)'}`)
  console.log('')

  let cursor: string | undefined
  let totalListed = 0
  let totalDeleted = 0
  let totalBytesFreed = 0

  while (true) {
    const result = await list({ prefix: 'videos/', cursor, token })

    for (const blob of result.blobs) {
      totalListed += 1
      if (cutoffMs !== null) {
        const uploadedAt = new Date(blob.uploadedAt).getTime()
        if (uploadedAt > cutoffMs) {
          continue
        }
      }
      const sizeMB = (blob.size / 1024 / 1024).toFixed(1)
      console.log(`  ${dryRun ? '[would delete]' : '[deleting]'} ${blob.pathname} (${sizeMB} MB, ${blob.uploadedAt})`)
      if (!dryRun) {
        try {
          await del(blob.url, { token })
          totalDeleted += 1
          totalBytesFreed += blob.size
        } catch (err) {
          console.error(`    failed: ${err instanceof Error ? err.message : String(err)}`)
        }
      } else {
        totalDeleted += 1
        totalBytesFreed += blob.size
      }
    }

    cursor = result.cursor
    if (!cursor) break
  }

  console.log('')
  console.log(`[cleanup] listed ${totalListed} blobs under videos/`)
  console.log(
    `[cleanup] ${dryRun ? 'would delete' : 'deleted'} ${totalDeleted} blob(s), ${(totalBytesFreed / 1024 / 1024).toFixed(1)} MB freed`,
  )
}

main().catch((err) => {
  console.error('[cleanup] fatal:', err)
  process.exit(1)
})
