import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // @ffmpeg-installer does a runtime dynamic require() to find its platform
  // binary, which Turbopack can't bundle. Mark it as a server external so
  // Next.js leaves the require alone and loads it from node_modules at
  // runtime.
  serverExternalPackages: ['@ffmpeg-installer/ffmpeg'],
}

export default nextConfig
