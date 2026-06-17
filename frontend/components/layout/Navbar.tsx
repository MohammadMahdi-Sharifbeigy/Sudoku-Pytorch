"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/solve",    label: "Solve" },
  { href: "/train",    label: "Train" },
  { href: "/optimize", label: "Optimize" },
] as const;

function GridLogo() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 22 22"
      fill="none"
      aria-hidden="true"
    >
      {/* 3×3 sudoku icon */}
      {[0, 1, 2].map((row) =>
        [0, 1, 2].map((col) => (
          <rect
            key={`${row}-${col}`}
            x={col * 8 + col * 1}
            y={row * 8 + row * 1}
            width="8"
            height="8"
            rx="1.5"
            fill={row === 1 && col === 1 ? "#00D4FF" : "none"}
            stroke="#00D4FF"
            strokeWidth={row === 1 && col === 1 ? "0" : "1"}
            opacity={row === 1 && col === 1 ? "1" : "0.45"}
          />
        ))
      )}
      {/* Bold 3×3 outer grid lines */}
      <rect
        x="0.5"
        y="0.5"
        width="21"
        height="21"
        rx="3"
        stroke="#00D4FF"
        strokeWidth="1"
        strokeOpacity="0.25"
        fill="none"
      />
    </svg>
  );
}

export function Navbar() {
  const pathname = usePathname();

  return (
    <header
      className="glass sticky top-0 z-50 border-b border-[#1E1E2E]"
      style={{ height: "44px" }}
    >
      {/* Faint cyan hairline at bottom */}
      <div
        className="absolute bottom-0 left-0 right-0 h-px"
        style={{
          background:
            "linear-gradient(90deg, transparent 0%, rgba(0,212,255,0.18) 50%, transparent 100%)",
        }}
      />

      <div className="mx-auto flex h-full max-w-[1440px] items-center justify-between px-6">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 transition-opacity hover:opacity-80"
          aria-label="SudokuAI home"
        >
          <GridLogo />
          <span
            className="text-nav font-bold tracking-tight"
            style={{
              fontFamily: "var(--font-space-mono), monospace",
              color: "#ffffff",
              letterSpacing: "-0.04em",
            }}
          >
            SudokuAI
          </span>
        </Link>

        {/* Nav links */}
        <nav aria-label="Main navigation">
          <ul className="flex items-center gap-8">
            {NAV_LINKS.map(({ href, label }) => {
              const active = pathname.startsWith(href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    className="relative text-nav transition-colors"
                    style={{
                      color: active ? "#00D4FF" : "rgba(200,200,212,0.75)",
                      textShadow: active
                        ? "0 0 8px rgba(0,212,255,0.4)"
                        : "none",
                    }}
                  >
                    {label}
                    {active && (
                      <span
                        className="absolute -bottom-[12px] left-0 right-0 h-px"
                        style={{
                          background:
                            "linear-gradient(90deg, transparent, #00D4FF, transparent)",
                        }}
                      />
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* CTA */}
        <Link
          href="/solve"
          className="btn-press text-nav hidden rounded-[var(--r-pill)] px-4 py-1.5 font-semibold sm:block"
          style={{
            background: "rgba(0,212,255,0.12)",
            color: "#00D4FF",
            border: "1px solid rgba(0,212,255,0.25)",
            transition: "background 0.15s ease, border-color 0.15s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(0,212,255,0.2)";
            e.currentTarget.style.borderColor = "rgba(0,212,255,0.5)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(0,212,255,0.12)";
            e.currentTarget.style.borderColor = "rgba(0,212,255,0.25)";
          }}
        >
          Solve now
        </Link>
      </div>
    </header>
  );
}
