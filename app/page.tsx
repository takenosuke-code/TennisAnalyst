import Link from 'next/link'

// Inline SVG line icons (24x24, stroke-based). Avoids the lucide-react
// dependency and keeps the bundle lean. The mapping follows what each
// feature actually does: a target for joint tracking, a wave for swing
// trails, an eye for visibility, a pin for baseline, a brain-ish glyph
// for AI coaching, a comparison glyph for side-by-side.
const FEATURES = [
  {
    title: 'Joint Tracking',
    desc: 'Pose extraction marks 33 body landmarks — shoulders, elbows, wrists, knees — on every sampled frame.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <circle cx="12" cy="12" r="9" />
        <circle cx="12" cy="12" r="4" />
        <circle cx="12" cy="12" r="1" />
      </svg>
    ),
  },
  {
    title: 'Swing Path Trails',
    desc: 'See your racket-hand motion path traced as a smooth Bezier curve across the swing.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M3 12c3-6 6-6 9 0s6 6 9 0" />
      </svg>
    ),
  },
  {
    title: 'Toggle Any Joint',
    desc: 'Show or hide individual joint groups to focus on what matters for your shot.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    ),
  },
  {
    title: 'Save a Baseline',
    desc: 'Mark your best-day swing as a baseline. Compare every future session against it.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M19 9c0 7-7 12-7 12S5 16 5 9a7 7 0 0114 0z" />
        <circle cx="12" cy="9" r="2.5" />
      </svg>
    ),
  },
  {
    title: 'AI Coaching',
    desc: 'Claude reads your joint angles vs your best day and tells you what held up and what drifted.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />
        <circle cx="12" cy="12" r="4" />
      </svg>
    ),
  },
  {
    title: 'Side by Side',
    desc: 'Synchronized video playback. Scrub both swings in lockstep at any speed.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <rect x="3" y="5" width="8" height="14" rx="1.5" />
        <rect x="13" y="5" width="8" height="14" rx="1.5" />
      </svg>
    ),
  },
]

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="relative max-w-6xl mx-auto px-4 pt-20 pb-16 grid lg:grid-cols-2 gap-12 items-center">
          <div className="text-center lg:text-left">
            <h1 className="text-5xl md:text-7xl font-black text-white leading-tight mb-6">
              Beat your
              <br />
              last swing.
            </h1>

            <p className="text-white/60 text-xl max-w-2xl mx-auto lg:mx-0 mb-10 leading-relaxed">
              Pose tracking for every swing you record. See your technique get sharper,
              more consistent, more powerful — week over week.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link
                href="/analyze"
                className="px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-white font-bold rounded-2xl text-lg transition-colors"
              >
                Analyze My Swing
              </Link>
              <Link
                href="/baseline"
                className="px-8 py-4 bg-white/10 hover:bg-white/15 text-white font-bold rounded-2xl text-lg transition-colors border border-white/10"
              >
                My Baselines
              </Link>
            </div>
          </div>

          {/* Hero product visual. Placeholder until a real VideoCanvas
              screenshot is dropped in (see Phase 1.2 of the ship plan).
              Aspect 9:16 to match phone tennis clips, ~440px wide on
              desktop. */}
          <div
            className="mx-auto w-full max-w-[340px] aspect-[9/16] rounded-2xl border border-dashed border-white/15 bg-white/[0.03] flex items-center justify-center text-center p-6"
            aria-label="Product preview placeholder"
          >
            {/* TODO Phase 1.2: replace with screenshot of /baseline/<id> render
                (skeleton overlay + angle badges + clean background). */}
            <p className="text-white/40 text-sm leading-relaxed">
              Skeleton overlay + joint angles render here.<br />
              Drop a screenshot of <code className="text-white/60">/baseline/&lt;id&gt;</code> view to fill this slot.
            </p>
          </div>
        </div>
      </section>

      {/* Feature grid */}
      <section className="max-w-6xl mx-auto px-4 py-20">
        <h2 className="text-3xl font-bold text-white text-center mb-12">
          Everything you need to level up
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="rounded-2xl border border-white/5 bg-white/[0.03] p-6 hover:bg-white/[0.06] transition-colors"
            >
              <div className="text-emerald-400 mb-3">{f.icon}</div>
              <h3 className="text-white font-semibold text-lg mb-2">{f.title}</h3>
              <p className="text-white/50 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="border-t border-white/5 py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold text-white mb-12">How it works</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { n: '1', label: 'Upload', desc: 'Drop your swing video (MP4/MOV)' },
              { n: '2', label: 'Track', desc: 'Pose extraction marks 33 joint landmarks every frame' },
              { n: '3', label: 'Pin a baseline', desc: 'Mark your best swing as the reference' },
              { n: '4', label: 'Beat it', desc: 'Upload future swings and see what changed' },
            ].map((step) => (
              <div key={step.n} className="flex flex-col items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-black text-xl">
                  {step.n}
                </div>
                <h3 className="text-white font-semibold">{step.label}</h3>
                <p className="text-white/50 text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-white/5 py-20 text-center">
        <div className="max-w-xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to see the difference?
          </h2>
          <p className="text-white/50 mb-8">
            No sign-up required. Upload your video and get instant feedback.
          </p>
          <Link
            href="/analyze"
            className="inline-block px-8 py-4 bg-emerald-500 hover:bg-emerald-400 text-white font-bold rounded-2xl text-lg transition-colors"
          >
            Start Analyzing Free
          </Link>
        </div>
      </section>
    </div>
  )
}
