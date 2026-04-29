import type { Metadata } from 'next'
import './globals.css'
import Nav from '@/components/Nav'
import ThemeQueryParam from '@/components/ThemeQueryParam'

export const metadata: Metadata = {
  title: 'TennisIQ - Swing Analysis',
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
      <body className="min-h-full flex flex-col bg-[var(--background)] text-[var(--foreground)]">
        <ThemeQueryParam />
        <Nav />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  )
}
