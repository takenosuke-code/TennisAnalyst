#!/usr/bin/env node
// Copies @ffmpeg/core's WASM + JS bundle into public/ffmpeg/ so the Clip
// Studio page can load ffmpeg.wasm from a same-origin URL instead of unpkg.
// This eliminates the third-party CDN dependency and the supply-chain risk
// it carries.
//
// Runs on every `npm install` via the postinstall hook in package.json.

import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const SRC = resolve('node_modules/@ffmpeg/core/dist/umd')
const DST = resolve('public/ffmpeg')

if (!existsSync(SRC)) {
  // Not installed yet (e.g. first install before the dep resolves). Vercel
  // runs postinstall after everything resolves, so this is a no-op safety net.
  console.log('[copy-ffmpeg-core] @ffmpeg/core not present yet; skipping.')
  process.exit(0)
}

mkdirSync(DST, { recursive: true })

for (const entry of readdirSync(SRC)) {
  if (!entry.startsWith('ffmpeg-core.')) continue
  copyFileSync(join(SRC, entry), join(DST, entry))
  console.log(`[copy-ffmpeg-core] ${entry} → public/ffmpeg/`)
}
