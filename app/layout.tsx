import type { Metadata } from 'next'
import './globals.css'
import Nav from '@/components/Nav'

export const metadata: Metadata = {
  title: 'Swingframe — Swing Analysis',
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
        {/* Manrope (OFL) for body + UI; Fraunces (OFL) for the editorial
            display headline. Pair: humanist sans for everything
            functional, dramatic variable serif for the hero so the words
            stand out against the green court instead of looking like
            generic SaaS type. Both loaded together to keep one Google
            Fonts round-trip. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Fraunces:opsz,wght@9..144,700;9..144,900&display=swap"
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
