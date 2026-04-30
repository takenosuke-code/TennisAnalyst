import Link from 'next/link'

// Tiny tennis-themed line glyphs. Single-stroke, hand-drawn feel —
// these replace the emoji feature icons (🎯🌊👁️📌🤖📐). Kept inline
// so we don't add an icon-library dependency and the strokes inherit
// `currentColor` for easy theme tinting.
function Glyph({ d, className = '' }: { d: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  )
}

const FEATURES: Array<{
  title: string
  desc: string
  stripe: 'clay' | 'hard-court' | 'lavender'
  glyph: string
}> = [
  {
    title: 'Joint tracking',
    desc: 'Pose AI marks 33 body landmarks — shoulders, elbows, wrists, knees — on every frame.',
    stripe: 'clay',
    // Stick figure
    glyph:
      'M16 6.5a2 2 0 1 1 0 4 2 2 0 0 1 0-4zM16 10.5v8m-5-5h10m-7 5l-3 7m5-7l3 7',
  },
  {
    title: 'Swing path trails',
    desc: 'Watch your racket-hand motion traced in real time as a smooth Bezier curve.',
    stripe: 'hard-court',
    // Curve
    glyph: 'M5 22 C 10 8, 22 8, 27 22',
  },
  {
    title: 'Toggle any joint',
    desc: 'Show or hide individual joint groups so you can focus on what matters for your shot.',
    stripe: 'lavender',
    // Eye
    glyph:
      'M4 16c3-6 9-9 12-9s9 3 12 9c-3 6-9 9-12 9s-9-3-12-9zM16 12.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z',
  },
  {
    title: 'Save a baseline',
    desc: 'Mark your best-day swing as the reference. Compare every future session against it.',
    stripe: 'clay',
    // Pin
    glyph: 'M16 4l5 5-3 2 2 6-4 4-4-4 2-6-3-2zM12 25l4-4',
  },
  {
    title: 'AI coaching',
    desc: 'Claude reads your joint angles vs your best day and tells you what held up and what drifted.',
    stripe: 'hard-court',
    // Tennis ball with stitching
    glyph: 'M16 4a12 12 0 1 1 0 24 12 12 0 0 1 0-24zM5 11c4 1 8 1 12 0M5 21c4-1 8-1 12 0',
  },
  {
    title: 'Side by side',
    desc: 'Synchronized video playback. Scrub both swings in lockstep at any speed.',
    stripe: 'lavender',
    // Two panels
    glyph: 'M5 7h10v18H5zM17 7h10v18H17z',
  },
]

const STRIPE_BG: Record<'clay' | 'hard-court' | 'lavender', string> = {
  clay: 'bg-clay',
  'hard-court': 'bg-hard-court',
  lavender: 'bg-lavender-2',
}

export default function HomePage() {
  return (
    <div>
      {/* Hero — lavender wash, asymmetric 2-col on desktop, stacked on mobile.
          Editorial headline (Manrope 800, big), pill CTA, no animate-pulse,
          no cyan-emerald gradient text. The visual on the right is a
          deliberately simple "court line" panel with a tennis ball glyph —
          we don't want stock photography or an emoji. */}
      <section className="bg-lavender-wash text-ink">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 pt-16 pb-20 lg:pt-24 lg:pb-28 grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7">
            <p className="text-[11px] sm:text-xs uppercase tracking-[0.18em] text-ink/70 mb-5">
              ai pose tracking · for tennis
            </p>
            <h1 className="font-display font-extrabold text-ink leading-[0.95] tracking-tight text-[44px] sm:text-[64px] lg:text-[88px] mb-6">
              beat your<br />last swing.
            </h1>
            <p className="text-ink/75 text-base sm:text-lg max-w-xl mb-8 leading-relaxed">
              Drop a video. We read your joint angles, save your best day, and tell
              you what drifted in plain English. No clipboards, no stopwatches,
              no jargon.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center px-7 py-3.5 rounded-full bg-clay hover:bg-[#c4633f] text-cream text-sm font-semibold tracking-wide transition-colors"
              >
                analyze my swing
              </Link>
              <Link
                href="/baseline"
                className="inline-flex items-center justify-center px-7 py-3.5 rounded-full bg-cream hover:bg-cream-soft text-ink text-sm font-semibold tracking-wide transition-colors"
              >
                my baselines
              </Link>
            </div>
          </div>

          {/* Hero visual — a cream "card" with a court diagram + ball,
              hard corners. Renders on lg+ as the asymmetric companion to
              the headline; hidden on mobile so the type carries the hero. */}
          <div className="hidden lg:block lg:col-span-5">
            <div className="relative">
              <div className="bg-cream aspect-[4/5] w-full max-w-md ml-auto p-6 flex flex-col">
                <div className="h-2 bg-clay -mx-6 -mt-6" />
                <div className="flex-1 flex items-center justify-center">
                  {/* Tennis court diagram, single-stroke. */}
                  <svg viewBox="0 0 200 280" className="w-full h-auto text-ink" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                    <rect x="20" y="20" width="160" height="240" />
                    <line x1="20" y1="140" x2="180" y2="140" />
                    <line x1="100" y1="20" x2="100" y2="140" />
                    <rect x="50" y="80" width="100" height="60" />
                    <rect x="50" y="140" width="100" height="60" />
                    <line x1="100" y1="200" x2="100" y2="260" />
                    {/* Ball */}
                    <circle cx="135" cy="105" r="6" fill="currentColor" stroke="none" />
                  </svg>
                </div>
                <div className="pt-4 border-t border-ink/10 flex items-baseline justify-between">
                  <span className="text-[11px] uppercase tracking-[0.18em] text-ink/60">today</span>
                  <span className="font-display font-extrabold text-2xl text-ink">+4.2°</span>
                </div>
                <p className="text-xs text-ink/55 mt-1">elbow extension vs. baseline</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature grid — pastel panels with a colored top stripe per card.
          Hard corners throughout. Stripe alternates clay/hard-court/lavender. */}
      <section className="bg-cream text-ink">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 py-20 lg:py-24">
          <div className="mb-12 max-w-2xl">
            <p className="text-[11px] uppercase tracking-[0.18em] text-ink/60 mb-3">what&apos;s inside</p>
            <h2 className="font-display font-extrabold text-3xl sm:text-4xl text-ink leading-tight">
              everything you need to level up.
            </h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f) => (
              <article key={f.title} className="bg-cream-soft flex flex-col">
                <div className={`h-2 ${STRIPE_BG[f.stripe]}`} />
                <div className="p-6 flex flex-col gap-4 flex-1">
                  <Glyph d={f.glyph} className="w-7 h-7 text-ink" />
                  <h3 className="font-display font-bold text-lg text-ink">{f.title}</h3>
                  <p className="text-sm text-ink/65 leading-relaxed">{f.desc}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* How it works — court-ink section to break the rhythm. Numbered steps
          render on a deep panel with cream type; squared corners throughout. */}
      <section className="bg-ink text-cream">
        <div className="max-w-5xl mx-auto px-5 sm:px-8 py-20 lg:py-24">
          <div className="mb-12 max-w-2xl">
            <p className="text-[11px] uppercase tracking-[0.18em] text-cream/60 mb-3">how it works</p>
            <h2 className="font-display font-extrabold text-3xl sm:text-4xl leading-tight">
              four steps, one cup of coffee.
            </h2>
          </div>
          <ol className="grid grid-cols-1 md:grid-cols-4 gap-5">
            {[
              { n: '01', label: 'upload', desc: 'Drop your swing video — MP4 or MOV.' },
              { n: '02', label: 'track', desc: 'AI extracts 33 joint landmarks from every frame.' },
              { n: '03', label: 'pin a baseline', desc: 'Mark your best swing as the reference.' },
              { n: '04', label: 'beat it', desc: 'Upload future swings and see what changed.' },
            ].map((step) => (
              <li key={step.n} className="bg-ink-soft p-5 flex flex-col gap-3">
                <span className="font-display font-extrabold text-clay-soft text-2xl tracking-tight">
                  {step.n}
                </span>
                <span className="font-display font-bold text-cream">{step.label}</span>
                <span className="text-sm text-cream/60 leading-relaxed">{step.desc}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Closing CTA — back to the lavender wash so the page bookends. */}
      <section className="bg-lavender-wash text-ink">
        <div className="max-w-3xl mx-auto px-5 sm:px-8 py-20 lg:py-24 text-center">
          <h2 className="font-display font-extrabold text-3xl sm:text-5xl leading-tight mb-4">
            ready to see the difference?
          </h2>
          <p className="text-ink/70 text-base sm:text-lg mb-8">
            No sign-up required. Upload your video and get instant feedback.
          </p>
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center px-8 py-4 rounded-full bg-clay hover:bg-[#c4633f] text-cream text-sm font-semibold tracking-wide transition-colors"
          >
            start analyzing free
          </Link>
        </div>
      </section>
    </div>
  )
}
