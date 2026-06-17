import Link from "next/link";
import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { SudokuPreview } from "@/components/ui/SudokuPreview";

/* ── Feature cards ─────────────────────────────────────────────── */
const FEATURES = [
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <rect x="2" y="2" width="24" height="24" rx="4" stroke="#00D4FF" strokeWidth="1.5" strokeOpacity="0.4"/>
        <rect x="6" y="6" width="5" height="5" rx="1" fill="#00D4FF" fillOpacity="0.15" stroke="#00D4FF" strokeWidth="1"/>
        <rect x="17" y="6" width="5" height="5" rx="1" fill="#00D4FF" fillOpacity="0.15" stroke="#00D4FF" strokeWidth="1"/>
        <rect x="6" y="17" width="5" height="5" rx="1" fill="#00D4FF" fillOpacity="0.15" stroke="#00D4FF" strokeWidth="1"/>
        <rect x="17" y="17" width="5" height="5" rx="1" fill="#00D4FF" fillOpacity="0.6" stroke="#00D4FF" strokeWidth="1"/>
        <line x1="14" y1="2" x2="14" y2="26" stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.15"/>
        <line x1="2" y1="14" x2="26" y2="14" stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.15"/>
      </svg>
    ),
    title: "CV Grid Detection",
    body:
      "Three-strategy vision pipeline — contour detection, KMeans clustering, and slice fallback — handles warped, low-res, and hand-drawn grids reliably.",
    tag: "OpenCV",
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <circle cx="14" cy="14" r="5" stroke="#00D4FF" strokeWidth="1.5"/>
        <circle cx="14" cy="4"  r="2" fill="#00D4FF" fillOpacity="0.6"/>
        <circle cx="14" cy="24" r="2" fill="#00D4FF" fillOpacity="0.6"/>
        <circle cx="4"  cy="14" r="2" fill="#00D4FF" fillOpacity="0.6"/>
        <circle cx="24" cy="14" r="2" fill="#00D4FF" fillOpacity="0.6"/>
        <line x1="14" y1="6"  x2="14" y2="9"  stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.5"/>
        <line x1="14" y1="19" x2="14" y2="22" stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.5"/>
        <line x1="6"  y1="14" x2="9"  y2="14" stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.5"/>
        <line x1="19" y1="14" x2="22" y2="14" stroke="#00D4FF" strokeWidth="1" strokeOpacity="0.5"/>
        <circle cx="9"  cy="9"  r="1.5" fill="#00D4FF" fillOpacity="0.35"/>
        <circle cx="19" cy="9"  r="1.5" fill="#00D4FF" fillOpacity="0.35"/>
        <circle cx="9"  cy="19" r="1.5" fill="#00D4FF" fillOpacity="0.35"/>
        <circle cx="19" cy="19" r="1.5" fill="#00D4FF" fillOpacity="0.35"/>
      </svg>
    ),
    title: "CNN Digit Recognition",
    body:
      "DigitCNN trained on MNIST + Persian Hoda dataset. Focal loss for class imbalance. Supports English and Farsi numerals with 99%+ accuracy.",
    tag: "PyTorch",
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
        <path d="M4 8h20M4 14h12M4 20h16" stroke="#00D4FF" strokeWidth="1.5" strokeLinecap="round" strokeOpacity="0.4"/>
        <path d="M20 17l4 3-4 3" stroke="#00D4FF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="22" cy="9" r="3" stroke="#00D4FF" strokeWidth="1.5" fillOpacity="0"/>
        <circle cx="22" cy="9" r="1" fill="#00D4FF"/>
      </svg>
    ),
    title: "Backtracking Solver",
    body:
      "Constraint-propagation backtracking solves any valid sudoku in under 2ms. Returns both original and solved grids with per-cell confidence scores.",
    tag: "Algorithm",
  },
] as const;

/* ── Stats ─────────────────────────────────────────────────────── */
const STATS: {
  label: string;
  targetValue: number;
  decimals: number;
  suffix: string;
  prefix?: string;
}[] = [
  { label: "Accuracy",       targetValue: 99.2, decimals: 1, suffix: "%" },
  { label: "Inference",      targetValue: 2,    decimals: 0, suffix: "ms", prefix: "<" },
  { label: "Digit scripts",  targetValue: 2,    decimals: 0, suffix: "" },
];

/* ── Page ──────────────────────────────────────────────────────── */
export default function HomePage() {
  return (
    <div className="flex flex-col">

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section
        className="relative overflow-hidden"
        style={{ padding: "var(--space-section) 24px" }}
      >
        {/* Background glow blob */}
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: "10%",
            left: "55%",
            width: "500px",
            height: "500px",
            borderRadius: "50%",
            background: "radial-gradient(ellipse, rgba(0,212,255,0.06) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        <div
          className="mx-auto flex flex-col items-center gap-16 lg:flex-row lg:items-center lg:justify-between"
          style={{ maxWidth: "1280px" }}
        >
          {/* Left: text */}
          <div className="flex max-w-xl flex-col gap-6 text-center lg:text-left">
            <div
              className="animate-fade-up inline-flex items-center gap-2 self-center rounded-full px-3 py-1 text-caption lg:self-start"
              style={{
                background: "rgba(0,212,255,0.08)",
                border: "1px solid rgba(0,212,255,0.2)",
                color: "#00D4FF",
                animationDelay: "0ms",
                letterSpacing: "-0.12px",
              }}
            >
              <span
                style={{
                  width: "6px",
                  height: "6px",
                  borderRadius: "50%",
                  background: "#00D4FF",
                  boxShadow: "0 0 6px #00D4FF",
                  display: "inline-block",
                }}
              />
              CV + CNN + Backtracking
            </div>

            <h1
              className="animate-fade-up text-hero"
              style={{ animationDelay: "80ms" }}
            >
              Solve Any{" "}
              <span
                style={{
                  color: "#00D4FF",
                  textShadow: "0 0 24px rgba(0,212,255,0.35)",
                }}
              >
                Sudoku
              </span>
              <br />
              Instantly
            </h1>

            <p
              className="animate-fade-up text-lead"
              style={{
                color: "var(--fg-muted)",
                animationDelay: "160ms",
                maxWidth: "460px",
              }}
            >
              Upload a puzzle photo. Our vision pipeline extracts the grid,
              the neural network reads each digit, and the solver completes it
              in milliseconds.
            </p>

            <div
              className="animate-fade-up flex flex-col items-center gap-3 sm:flex-row lg:items-start"
              style={{ animationDelay: "240ms" }}
            >
              <Link href="/solve" className="btn-primary">
                Solve a puzzle
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </Link>
              <Link href="/train" className="btn-ghost-pill">
                Train model
              </Link>
            </div>
          </div>

          {/* Right: animated sudoku grid */}
          <div
            className="animate-fade-in"
            style={{ animationDelay: "320ms", flexShrink: 0 }}
          >
            <div
              style={{
                padding: "20px",
                background: "var(--surface)",
                borderRadius: "var(--r-lg)",
                border: "1px solid var(--border-col)",
                boxShadow: "0 0 40px rgba(0,212,255,0.06)",
              }}
            >
              <SudokuPreview />
              <p
                className="mt-3 text-center text-caption"
                style={{ color: "var(--fg-muted)", letterSpacing: "-0.12px" }}
              >
                Live solve animation ↑
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats row ────────────────────────────────────────────── */}
      <section
        style={{
          borderTop: "1px solid var(--border-col)",
          borderBottom: "1px solid var(--border-col)",
          padding: "40px 24px",
          background: "var(--surface)",
        }}
      >
        <div
          className="mx-auto grid grid-cols-3"
          style={{
            maxWidth: "1280px",
          }}
        >
          {STATS.map(({ label, targetValue, decimals, suffix, prefix }) => (
            <div
              key={label}
              className="flex flex-col items-center gap-1 px-6 text-center"
            >
              <span
                style={{
                  fontFamily: "var(--font-space-mono), monospace",
                  fontSize: "clamp(28px, 3.5vw, 40px)",
                  fontWeight: 700,
                  color: "#00D4FF",
                  letterSpacing: "-0.03em",
                  lineHeight: 1.1,
                  textShadow: "0 0 16px rgba(0,212,255,0.25)",
                }}
              >
                <AnimatedCounter
                  target={targetValue}
                  decimals={decimals}
                  suffix={suffix}
                  prefix={prefix ?? ""}
                  duration={1600}
                />
              </span>
              <span className="text-caption" style={{ color: "var(--fg-muted)" }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────── */}
      <section
        style={{ padding: "var(--space-section) 24px", maxWidth: "100%" }}
      >
        <div className="mx-auto" style={{ maxWidth: "1280px" }}>
          <div className="mb-12 text-center">
            <h2
              className="text-display"
              style={{ marginBottom: "12px" }}
            >
              How it works
            </h2>
            <p className="text-body mx-auto max-w-md" style={{ color: "var(--fg-muted)" }}>
              Three specialised models — one seamless pipeline.
            </p>
          </div>

          <div
            className="grid gap-5"
            style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}
          >
            {FEATURES.map(({ icon, title, body, tag }, i) => (
              <div
                key={title}
                className="animate-fade-up card-dark flex flex-col gap-5 p-7"
                style={{ animationDelay: `${i * 80}ms` }}
              >
                {/* Icon */}
                <div
                  style={{
                    width: "52px",
                    height: "52px",
                    borderRadius: "var(--r-md)",
                    background: "rgba(0,212,255,0.06)",
                    border: "1px solid rgba(0,212,255,0.15)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  {icon}
                </div>

                {/* Tag */}
                <span
                  className="self-start rounded-full px-2 py-0.5 text-caption"
                  style={{
                    fontFamily: "var(--font-space-mono), monospace",
                    fontSize: "10px",
                    color: "#00D4FF",
                    background: "rgba(0,212,255,0.08)",
                    border: "1px solid rgba(0,212,255,0.18)",
                    letterSpacing: "0.04em",
                  }}
                >
                  {tag}
                </span>

                <div className="flex flex-col gap-2">
                  <h3
                    style={{
                      fontFamily: "var(--font-space-mono), monospace",
                      fontSize: "17px",
                      fontWeight: 700,
                      color: "#ffffff",
                      letterSpacing: "-0.03em",
                    }}
                  >
                    {title}
                  </h3>
                  <p className="text-body" style={{ color: "var(--fg-muted)" }}>
                    {body}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA banner ───────────────────────────────────────────── */}
      <section
        style={{
          padding: "var(--space-section) 24px",
          borderTop: "1px solid var(--border-col)",
        }}
      >
        <div
          className="mx-auto flex flex-col items-center gap-6 text-center"
          style={{ maxWidth: "640px" }}
        >
          <h2 className="text-display">
            Ready to solve?
          </h2>
          <p className="text-body" style={{ color: "var(--fg-muted)" }}>
            Upload any sudoku image — printed, hand-drawn, or photographed.
            English and Farsi digits supported.
          </p>
          <Link href="/solve" className="btn-primary">
            Get started — it&apos;s free
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </Link>
        </div>
      </section>

    </div>
  );
}
