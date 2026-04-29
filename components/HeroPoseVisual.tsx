/*
 * Hero visual for the homepage. A stylized SVG of a tennis player at
 * contact with the same kind of skeleton + joint-angle overlay the
 * product produces on a real upload. Not a screenshot — but it
 * communicates "this is what you get back" without waiting for the
 * founder to drop a real PNG.
 *
 * When a real product screenshot lands, replace the inner SVG with an
 * <Image> tag pointing at /public/hero-pose.png and keep the wrapper
 * styling.
 */

export default function HeroPoseVisual() {
  return (
    <div className="mx-auto w-full max-w-[360px] aspect-[9/16] rounded-2xl overflow-hidden bg-[#0E1116] relative shadow-2xl shadow-emerald-500/10 border border-white/10">
      {/* Subtle court-bg gradient so the SVG doesn't float on pure black */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#1a2530] via-[#0E1116] to-[#0E1116]" />

      <svg
        viewBox="0 0 360 640"
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden
      >
        {/* Faint court-line motif */}
        <line x1="20" y1="540" x2="340" y2="540" stroke="#2b6cb0" strokeOpacity="0.25" strokeWidth="1.5" />
        <line x1="180" y1="500" x2="180" y2="630" stroke="#2b6cb0" strokeOpacity="0.15" strokeWidth="1" />

        {/* Player silhouette — abstract tennis pose at contact (right-handed forehand). */}
        <g fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.10)" strokeWidth="1">
          {/* head */}
          <circle cx="170" cy="120" r="22" />
          {/* torso */}
          <path d="M 140 145 Q 135 220 150 290 Q 175 310 200 290 Q 215 220 210 145 Q 175 130 140 145 Z" />
          {/* dominant arm extended (contact position) */}
          <path d="M 210 165 Q 270 175 295 195 Q 318 200 322 215 Q 320 222 308 222 Q 285 220 264 215 Q 235 205 210 195 Z" />
          {/* off arm tucked */}
          <path d="M 142 170 Q 110 200 100 245 Q 100 260 110 260 Q 118 250 130 230 Q 140 210 148 195 Z" />
          {/* legs in athletic stance */}
          <path d="M 145 290 Q 130 380 115 470 Q 115 490 135 490 Q 150 405 160 320 Z" />
          <path d="M 195 290 Q 215 380 235 470 Q 240 495 220 495 Q 195 410 180 320 Z" />
        </g>

        {/* Racket */}
        <g stroke="#F4F1EA" strokeOpacity="0.6" strokeWidth="2" fill="none">
          <line x1="320" y1="218" x2="338" y2="208" />
          <ellipse cx="343" cy="200" rx="14" ry="18" transform="rotate(20 343 200)" />
        </g>

        {/* Pose-overlay skeleton — the product's actual deliverable */}
        <g stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" fill="none">
          {/* shoulder bar */}
          <line x1="148" y1="158" x2="208" y2="158" />
          {/* hip bar */}
          <line x1="152" y1="280" x2="200" y2="280" />
          {/* spine */}
          <line x1="178" y1="158" x2="176" y2="280" />
          {/* dominant arm: shoulder -> elbow -> wrist (extended) */}
          <line x1="208" y1="158" x2="265" y2="195" />
          <line x1="265" y1="195" x2="318" y2="218" />
          {/* off arm */}
          <line x1="148" y1="158" x2="120" y2="220" />
          <line x1="120" y1="220" x2="108" y2="255" />
          {/* legs: hip -> knee -> ankle */}
          <line x1="152" y1="280" x2="130" y2="395" />
          <line x1="130" y1="395" x2="125" y2="488" />
          <line x1="200" y1="280" x2="222" y2="395" />
          <line x1="222" y1="395" x2="228" y2="488" />
        </g>

        {/* Joint dots */}
        <g fill="#F4F1EA">
          <circle cx="148" cy="158" r="4.5" />
          <circle cx="208" cy="158" r="4.5" />
          <circle cx="265" cy="195" r="4.5" />
          <circle cx="318" cy="218" r="4.5" />
          <circle cx="120" cy="220" r="4.5" />
          <circle cx="108" cy="255" r="4.5" />
          <circle cx="152" cy="280" r="4.5" />
          <circle cx="200" cy="280" r="4.5" />
          <circle cx="130" cy="395" r="4.5" />
          <circle cx="125" cy="488" r="4.5" />
          <circle cx="222" cy="395" r="4.5" />
          <circle cx="228" cy="488" r="4.5" />
        </g>

        {/* Angle pills — the product's signature data viz */}
        <g fontFamily="-apple-system, BlinkMacSystemFont, sans-serif" fontSize="12" fontWeight="600">
          <rect x="220" y="178" rx="6" ry="6" width="48" height="20" fill="rgba(0,0,0,0.7)" stroke="#10b981" strokeOpacity="0.4" />
          <text x="244" y="192" textAnchor="middle" fill="#34d399">162°</text>

          <rect x="78" y="232" rx="6" ry="6" width="44" height="20" fill="rgba(0,0,0,0.7)" stroke="#10b981" strokeOpacity="0.4" />
          <text x="100" y="246" textAnchor="middle" fill="#34d399">98°</text>

          <rect x="80" y="370" rx="6" ry="6" width="48" height="20" fill="rgba(0,0,0,0.7)" stroke="#10b981" strokeOpacity="0.4" />
          <text x="104" y="384" textAnchor="middle" fill="#34d399">142°</text>
        </g>

        {/* Racket-head trail — fading dots showing the swing path */}
        <g fill="#D7E22A">
          <circle cx="252" cy="280" r="3" opacity="0.25" />
          <circle cx="270" cy="262" r="3.2" opacity="0.4" />
          <circle cx="290" cy="245" r="3.4" opacity="0.55" />
          <circle cx="310" cy="228" r="3.6" opacity="0.7" />
          <circle cx="328" cy="215" r="3.8" opacity="0.85" />
        </g>
      </svg>

      {/* Faux chip in bottom corner — visual signal that "real diagnostics
          live here" without showing the real RTMPose · Modal GPU chip
          (which scared the reviewers). */}
      <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
        Pose extracted · 28 frames
      </div>
    </div>
  )
}
