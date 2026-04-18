#!/usr/bin/env node
// Downloads yt-dlp's self-contained (PyInstaller) binary at install time.
//
// youtube-dl-exec ships the Python zipapp variant of yt-dlp, which relies
// on a system python3. Vercel's Node runtime doesn't have python3, so the
// zipapp fails with "env: python3: No such file or directory". The
// self-contained releases (yt-dlp_linux, yt-dlp_macos, yt-dlp_linux_aarch64)
// bundle a Python interpreter inside and run without any system dep.
//
// We pick the right asset for the current platform, drop it at bin/yt-dlp,
// and make it executable. Next.js's outputFileTracingIncludes bundles the
// file into the admin-route function at deploy time.
//
// The yt-dlp version is pinned via YT_DLP_VERSION below so redeploys are
// reproducible.

import { createWriteStream, chmodSync, existsSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { arch, platform } from 'node:os'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'

const YT_DLP_VERSION = '2025.09.05'
const TARGET_DIR = 'bin'
const TARGET_FILE = join(TARGET_DIR, 'yt-dlp')

function selectAsset() {
  const os = platform()
  const cpu = arch()
  if (os === 'linux') {
    if (cpu === 'x64') return 'yt-dlp_linux'
    if (cpu === 'arm64') return 'yt-dlp_linux_aarch64'
  }
  if (os === 'darwin') return 'yt-dlp_macos'
  if (os === 'win32') return 'yt-dlp.exe'
  throw new Error(`Unsupported platform: ${os}/${cpu}`)
}

async function main() {
  if (existsSync(TARGET_FILE)) {
    console.log('[fetch-yt-dlp] Already present at', TARGET_FILE)
    return
  }

  const asset = selectAsset()
  const url = `https://github.com/yt-dlp/yt-dlp/releases/download/${YT_DLP_VERSION}/${asset}`
  console.log(`[fetch-yt-dlp] ${asset} @ ${YT_DLP_VERSION} → ${TARGET_FILE}`)

  mkdirSync(TARGET_DIR, { recursive: true })

  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok || !res.body) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`)
  }
  await pipeline(Readable.fromWeb(res.body), createWriteStream(TARGET_FILE))
  chmodSync(TARGET_FILE, 0o755)
  console.log('[fetch-yt-dlp] done.')
}

main().catch((err) => {
  console.error('[fetch-yt-dlp] FAILED:', err.message ?? err)
  process.exit(1)
})
