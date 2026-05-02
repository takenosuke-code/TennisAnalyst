/**
 * smoke-test-blob-write.ts — verify a Vercel Blob token can actually write.
 *
 * Run this with the same token you put in Railway's env. If it succeeds
 * here, the token is fine and Railway's deployment is the problem. If
 * it fails here with a 403, the token itself is bad and rotating again
 * (or copying from a different store) is the fix.
 *
 *   BLOB_READ_WRITE_TOKEN=vercel_blob_rw_... npx tsx scripts/smoke-test-blob-write.ts
 *
 * Optional: pass a custom path to mirror exactly what Railway tries.
 *   BLOB_READ_WRITE_TOKEN=vercel_blob_rw_... npx tsx scripts/smoke-test-blob-write.ts baseline-trims/foo.txt
 */

import { put, del } from '@vercel/blob'

const token = process.env.BLOB_READ_WRITE_TOKEN
if (!token) {
  console.error(
    'Missing BLOB_READ_WRITE_TOKEN. Get the value from Vercel → Storage → TennisIQ → .env.local tab.',
  )
  process.exit(1)
}

if (!token.startsWith('vercel_blob_rw_')) {
  console.error(
    `Token prefix is "${token.slice(0, 20)}..." — expected "vercel_blob_rw_". A read-only token (vercel_blob_r_) cannot write and will 403.`,
  )
  process.exit(1)
}

const pathname = process.argv[2] ?? `smoke-test/${Date.now()}.txt`
const body = `smoke test ${new Date().toISOString()}`

async function main() {
  console.log(`[smoke] Token prefix OK (vercel_blob_rw_...)`)
  console.log(`[smoke] Attempting put: ${pathname}`)
  console.log('')

  let blob
  try {
    blob = await put(pathname, body, {
      access: 'public',
      token,
      contentType: 'text/plain',
      addRandomSuffix: false,
    })
  } catch (err) {
    console.error('[smoke] PUT FAILED:')
    console.error(err instanceof Error ? err.message : String(err))
    console.error('')
    console.error('Diagnosis:')
    console.error('- 403: token belongs to a different store, OR was just rotated and you grabbed the old value, OR is read-only.')
    console.error('- 401: token format invalid.')
    console.error('- 402: storage quota hit.')
    process.exit(1)
  }

  console.log(`[smoke] PUT OK`)
  console.log(`[smoke] URL: ${blob.url}`)
  console.log(`[smoke] downloadUrl: ${blob.downloadUrl}`)
  console.log('')
  console.log('Token works locally for writes.')
  console.log('If Railway still 403s with this same token, Railway is using a stale/different value.')
  console.log('')

  try {
    await del(blob.url, { token })
    console.log(`[smoke] cleanup: deleted test blob`)
  } catch (err) {
    console.warn(`[smoke] cleanup failed (non-fatal): ${err instanceof Error ? err.message : err}`)
  }
}

main().catch((err) => {
  console.error('[smoke] fatal:', err)
  process.exit(1)
})
