import type { Metadata } from 'next'
import './globals.css'
import Nav from '@/components/Nav'

export const metadata: Metadata = {
  title: 'TennisIQ — Swing Analysis',
  description:
    'Compare your tennis swing to the pros using AI-powered pose tracking and analysis.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="h-full">
      <head>
        {/* Manrope (OFL) — humanist sans, replaces system / Inter / Geist.
            Preconnect saves a round trip; the css2 import covers the
            weights we actually use (400 body, 500 emphasis, 700 strong,
            800 hero display). Loaded once at the root — every route
            inherits it through var(--font-sans) on body. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full flex flex-col bg-app-shell text-cream">
        <Nav />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  )
}
