import { notFound } from 'next/navigation'

// Admin routes depend on local binaries (yt-dlp, a bundled ffmpeg, and an
// arm64 python3) that don't exist on Vercel, so we hide them in the deployed
// build. Set NEXT_PUBLIC_ADMIN_ENABLED=true in .env for local development.
export default function AdminLayout({ children }: { children: React.ReactNode }) {
  if (process.env.NEXT_PUBLIC_ADMIN_ENABLED !== 'true') {
    notFound()
  }
  return <>{children}</>
}
