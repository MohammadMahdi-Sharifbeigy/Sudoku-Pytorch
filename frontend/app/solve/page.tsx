"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { solveSudoku, type SolveResponse, type CellData } from "@/lib/api";
import { CellGridSkeleton, SudokuBoardSkeleton } from "@/components/ui/Skeletons";

/* ── Types ─────────────────────────────────────────────────────── */
type Tab = "solution" | "cells" | "raw";

/* ── Constants ──────────────────────────────────────────────────── */
const LOADING_STEPS = [
  "Detecting grid…",
  "Extracting cells…",
  "Running CNN…",
  "Solving puzzle…",
] as const;

const TABS: { id: Tab; label: string }[] = [
  { id: "solution", label: "Solution" },
  { id: "cells",    label: "Cell Analysis" },
  { id: "raw",      label: "Raw Grid" },
];

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB

/* ── Helpers ────────────────────────────────────────────────────── */
function confColor(c: number): string {
  if (c >= 0.8) return "#00E396";
  if (c >= 0.5) return "#FFB800";
  return "#FF4560";
}

function gridToText(grid: number[][]): string {
  return grid.map((row) => row.join(" ")).join("\n");
}

/* ── Sub-components ─────────────────────────────────────────────── */

/** Animated spinner ring */
function Spinner() {
  return (
    <svg
      width="36"
      height="36"
      viewBox="0 0 36 36"
      fill="none"
      aria-hidden="true"
      style={{ animation: "spin 1s linear infinite" }}
    >
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <circle
        cx="18" cy="18" r="15"
        stroke="rgba(0,212,255,0.15)"
        strokeWidth="2.5"
      />
      <path
        d="M18 3 A15 15 0 0 1 33 18"
        stroke="#00D4FF"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Loading overlay with step labels */
function LoadingState({ step }: { step: number }) {
  return (
    <div
      className="animate-fade-in mt-6 flex flex-col items-center gap-4 rounded-[var(--r-lg)] p-6"
      style={{
        background: "rgba(0,212,255,0.04)",
        border: "1px solid rgba(0,212,255,0.14)",
      }}
    >
      <Spinner />
      <div className="flex flex-col items-center gap-2">
        {LOADING_STEPS.map((label, i) => (
          <div
            key={label}
            className="flex items-center gap-2 transition-all duration-300"
            style={{
              opacity: i <= step ? 1 : 0.25,
              transform: i === step ? "translateX(0)" : "none",
            }}
          >
            {i < step ? (
              /* completed */
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <circle cx="7" cy="7" r="6" fill="rgba(0,227,150,0.15)" stroke="#00E396" strokeWidth="1"/>
                <path d="M4 7l2 2 4-4" stroke="#00E396" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            ) : i === step ? (
              /* active dot */
              <span
                style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: "#00D4FF",
                  boxShadow: "0 0 8px rgba(0,212,255,0.8)",
                  display: "inline-block",
                  animation: "glow-pulse 1.2s ease-in-out infinite",
                }}
              />
            ) : (
              /* pending */
              <span
                style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: "rgba(200,200,212,0.2)",
                  display: "inline-block",
                }}
              />
            )}
            <span
              className="text-caption"
              style={{
                color: i === step ? "#00D4FF" : i < step ? "#00E396" : "var(--fg-muted)",
                fontFamily: "var(--font-space-mono), monospace",
                fontSize: "12px",
                letterSpacing: "0.02em",
              }}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Inline error card */
function ErrorCard({ message }: { message: string }) {
  return (
    <div
      className="animate-fade-in mt-5 flex items-start gap-3 rounded-[var(--r-md)] px-4 py-3"
      style={{
        background: "rgba(255,69,96,0.07)",
        border: "1px solid rgba(255,69,96,0.22)",
      }}
      role="alert"
    >
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }}>
        <circle cx="9" cy="9" r="8" stroke="#FF4560" strokeWidth="1.3"/>
        <path d="M9 5v4M9 12.5h.01" stroke="#FF4560" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
      <span className="text-caption" style={{ color: "#FF4560" }}>{message}</span>
    </div>
  );
}

/** Apple-style pill segment tab bar */
function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  return (
    <div
      role="tablist"
      className="relative flex items-center gap-0.5 rounded-[var(--r-pill)] p-1"
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid var(--border-col)",
        width: "fit-content",
      }}
    >
      {TABS.map(({ id, label }) => {
        const isActive = active === id;
        return (
          <button
            key={id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(id)}
            className="btn-press relative z-10 min-h-12 rounded-[var(--r-pill)] px-4 py-1.5 text-caption transition-all duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#00D4FF]"
            style={{
              background: isActive ? "var(--cyan)" : "transparent",
              color: isActive ? "#0A0A0F" : "var(--fg-muted)",
              fontWeight: isActive ? 600 : 400,
              fontSize: "13px",
              letterSpacing: isActive ? "-0.2px" : "-0.12px",
              border: "none",
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/** Solution tab — side-by-side original + solved images */
function SolutionTab({ result, onCopy }: { result: SolveResponse; onCopy: () => void }) {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 sm:grid-cols-2">
        {[
          { label: "Original", b64: result.original_image_b64 },
          { label: "Solved",   b64: result.solved_image_b64 },
        ].map(({ label, b64 }) => (
          <div key={label} className="flex flex-col gap-2">
            <span
              className="text-caption"
              style={{
                color: "var(--fg-muted)",
                fontFamily: "var(--font-space-mono), monospace",
                fontSize: "11px",
                letterSpacing: "0.06em",
                textTransform: "uppercase",
              }}
            >
              {label}
            </span>
            <div
              style={{
                borderRadius: "var(--r-lg)",
                overflow: "hidden",
                border: "1px solid var(--border-col)",
                background: "#0e0e16",
                boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
                aspectRatio: "1/1",
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${b64}`}
                alt={`${label} sudoku grid`}
                style={{ width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated" }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Stats row */}
      <div
        className="flex items-center justify-between rounded-[var(--r-md)] px-4 py-3"
        style={{ background: "var(--surface)", border: "1px solid var(--border-col)" }}
      >
        <div className="flex items-center gap-6">
          {result.solve_time_ms != null && (
            <div className="flex flex-col">
              <span
                style={{
                  fontFamily: "var(--font-space-mono), monospace",
                  fontSize: "18px",
                  fontWeight: 700,
                  color: "#00D4FF",
                  lineHeight: 1,
                }}
              >
                {result.solve_time_ms.toFixed(1)}ms
              </span>
              <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>solve time</span>
            </div>
          )}
          <div className="flex flex-col">
            <span
              style={{
                fontFamily: "var(--font-space-mono), monospace",
                fontSize: "18px",
                fontWeight: 700,
                color: "#00E396",
                lineHeight: 1,
              }}
            >
              {result.cells.filter((c) => c.value > 0 && c.confidence >= 0.8).length}
            </span>
            <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>high-conf digits</span>
          </div>
          <div className="flex flex-col">
            <span
              style={{
                fontFamily: "var(--font-space-mono), monospace",
                fontSize: "18px",
                fontWeight: 700,
                color: "#FFB800",
                lineHeight: 1,
              }}
            >
              {result.cells.filter((c) => c.value > 0 && c.confidence < 0.5).length}
            </span>
            <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>low-conf cells</span>
          </div>
        </div>

        <button
          onClick={onCopy}
          className="btn-press flex min-h-12 items-center gap-2 rounded-[var(--r-sm)] px-3 py-2 text-caption transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#00D4FF]"
          aria-label="Copy solved grid"
          style={{
            background: "rgba(0,212,255,0.08)",
            border: "1px solid rgba(0,212,255,0.2)",
            color: "#00D4FF",
            fontSize: "13px",
            cursor: "pointer",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,212,255,0.16)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,212,255,0.08)"; }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <rect x="4" y="4" width="9" height="9" rx="1.5" stroke="#00D4FF" strokeWidth="1.2"/>
            <path d="M4 4V2.5A1.5 1.5 0 0 1 5.5 1H11.5A1.5 1.5 0 0 1 13 2.5V8.5A1.5 1.5 0 0 1 11.5 10H10" stroke="#00D4FF" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
          Copy Grid
        </button>
      </div>
    </div>
  );
}

/** Cell Analysis tab — 9×9 grid of cell cards */
function CellAnalysisTab({ cells }: { cells: CellData[] }) {
  const avgConf =
    cells.filter((c) => c.value > 0).reduce((sum, c) => sum + c.confidence, 0) /
    Math.max(1, cells.filter((c) => c.value > 0).length);

  const lowConfCount = cells.filter((c) => c.value > 0 && c.confidence < 0.5).length;

  return (
    <div className="flex flex-col gap-4">
      {/* Summary stats */}
      <div
        className="flex items-center gap-4 rounded-[var(--r-md)] px-4 py-3"
        style={{ background: "var(--surface)", border: "1px solid var(--border-col)" }}
      >
        <div className="flex items-center gap-1.5">
          <span style={{ fontSize: "13px", fontFamily: "var(--font-space-mono), monospace", fontWeight: 700, color: confColor(avgConf) }}>
            {(avgConf * 100).toFixed(1)}%
          </span>
          <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>avg confidence</span>
        </div>
        <div style={{ width: 1, height: 20, background: "var(--border-col)" }} />
        <div className="flex items-center gap-1.5">
          <span style={{ fontSize: "13px", fontFamily: "var(--font-space-mono), monospace", fontWeight: 700, color: "#00D4FF" }}>
            {cells.filter((c) => c.value > 0).length}
          </span>
          <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>detected digits</span>
        </div>
        {lowConfCount > 0 && (
          <>
            <div style={{ width: 1, height: 20, background: "var(--border-col)" }} />
            <div className="flex items-center gap-1.5">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                <path d="M6 1L11 10H1L6 1z" stroke="#FFB800" strokeWidth="1.1" strokeLinejoin="round"/>
                <path d="M6 4.5V7M6 8.5h.01" stroke="#FFB800" strokeWidth="1.1" strokeLinecap="round"/>
              </svg>
              <span style={{ fontSize: "13px", fontFamily: "var(--font-space-mono), monospace", fontWeight: 700, color: "#FFB800" }}>
                {lowConfCount}
              </span>
              <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>low-conf warnings</span>
            </div>
          </>
        )}
      </div>

      {/* 9×9 cell grid */}
      <div className="overflow-x-auto">
        <div
          className="min-w-[620px]"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(9, 1fr)",
            gap: "2px",
            background: "var(--border-col)",
            borderRadius: "var(--r-md)",
            overflow: "hidden",
            border: "1px solid var(--border-col)",
          }}
        >
          {cells.map((cell) => {
            const boxRow = Math.floor(cell.row / 3);
            const boxCol = Math.floor(cell.col / 3);
            const isBoxBorderRight  = cell.col === 2 || cell.col === 5;
            const isBoxBorderBottom = cell.row === 2 || cell.row === 5;
            const conf = cell.confidence;
            const color = cell.value > 0 ? confColor(conf) : "var(--fg-dim)";

            return (
              <div
                key={`${cell.row}-${cell.col}`}
                className="flex flex-col items-center gap-1 p-1"
                style={{
                  background: (boxRow + boxCol) % 2 === 0 ? "var(--surface)" : "var(--surface-2, #16161F)",
                  borderRight: isBoxBorderRight  ? "2px solid rgba(0,212,255,0.2)" : undefined,
                  borderBottom: isBoxBorderBottom ? "2px solid rgba(0,212,255,0.2)" : undefined,
                  minHeight: "64px",
                }}
                title={cell.value > 0 ? `Confidence: ${(conf * 100).toFixed(1)}%` : "Empty cell"}
              >
              {/* 28×28 cell image */}
              {cell.image_b64 ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:image/png;base64,${cell.image_b64}`}
                  alt={cell.value > 0 ? `Digit ${cell.value}` : "empty"}
                  width={28}
                  height={28}
                  style={{ imageRendering: "pixelated", borderRadius: "2px", opacity: cell.value > 0 ? 1 : 0.3 }}
                />
              ) : (
                <div style={{ width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--fg-dim)", display: "inline-block" }} />
                </div>
              )}

              {/* Digit */}
              <span
                style={{
                  fontFamily: "var(--font-space-mono), monospace",
                  fontSize: "13px",
                  fontWeight: 700,
                  color,
                  lineHeight: 1,
                  textShadow: cell.value > 0 ? `0 0 6px ${color}55` : "none",
                }}
              >
                {cell.value > 0 ? cell.value : "·"}
              </span>

              {/* Confidence bar */}
              {cell.value > 0 && (
                <div
                  style={{
                    width: "100%",
                    height: "3px",
                    background: "rgba(255,255,255,0.06)",
                    borderRadius: "var(--r-pill)",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${conf * 100}%`,
                      background: color,
                      borderRadius: "var(--r-pill)",
                      transition: "width 0.6s cubic-bezier(0.16,1,0.3,1)",
                    }}
                  />
                </div>
              )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5" style={{ paddingLeft: "2px" }}>
        {[
          { color: "#00E396", label: "≥80% — high" },
          { color: "#FFB800", label: "50–79% — medium" },
          { color: "#FF4560", label: "<50% — low" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: "inline-block" }} />
            <span style={{ fontSize: "11px", color: "var(--fg-muted)", letterSpacing: "-0.1px" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Raw Grid tab — 9×9 pre-solve predicted values */
function RawGridTab({ grid, cells }: { grid: number[][]; cells: CellData[] }) {
  const confMap = new Map(cells.map((c) => [`${c.row}-${c.col}`, c.confidence]));

  return (
    <div className="flex flex-col gap-3">
      <p className="text-caption" style={{ color: "var(--fg-muted)" }}>
        CNN predictions before backtracking solver. Amber cells have confidence&nbsp;&lt;&nbsp;50%.
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(9, 1fr)",
          gap: "2px",
          background: "var(--border-col)",
          borderRadius: "var(--r-md)",
          overflow: "hidden",
          border: "1px solid var(--border-col)",
        }}
      >
        {grid.flatMap((row, ri) =>
          row.map((val, ci) => {
            const conf = confMap.get(`${ri}-${ci}`) ?? 1;
            const isLow = val > 0 && conf < 0.5;
            const isBoxBorderRight  = ci === 2 || ci === 5;
            const isBoxBorderBottom = ri === 2 || ri === 5;

            return (
              <div
                key={`${ri}-${ci}`}
                style={{
                  height: "48px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: isLow
                    ? "rgba(255,184,0,0.08)"
                    : (Math.floor(ri / 3) + Math.floor(ci / 3)) % 2 === 0
                      ? "var(--surface)"
                      : "var(--surface-2, #16161F)",
                  borderRight:  isBoxBorderRight  ? "2px solid rgba(0,212,255,0.2)" : undefined,
                  borderBottom: isBoxBorderBottom ? "2px solid rgba(0,212,255,0.2)" : undefined,
                  transition: "background 0.2s ease",
                  animation: isLow ? "glow-pulse 2.5s ease-in-out infinite" : undefined,
                }}
                title={val > 0 ? `Confidence: ${(conf * 100).toFixed(1)}%` : "Empty"}
              >
                <span
                  style={{
                    fontFamily: "var(--font-space-mono), monospace",
                    fontSize: "16px",
                    fontWeight: 700,
                    color: val === 0 ? "var(--fg-dim)" : isLow ? "#FFB800" : "var(--fg)",
                    textShadow: isLow ? "0 0 8px rgba(255,184,0,0.4)" : "none",
                  }}
                >
                  {val === 0 ? "·" : val}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

/* ── Main Page ──────────────────────────────────────────────────── */
export default function SolvePage() {
  const [file, setFile]           = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult]       = useState<SolveResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [activeTab, setActiveTab] = useState<Tab>("solution");
  const [error, setError]         = useState<string | null>(null);
  const abortRef                  = useRef<AbortController | null>(null);
  const stepTimerRef              = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Revoke object URL on unmount */
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  /* ── Drop handler ────────────────────────────────────────────── */
  const onDrop = useCallback((accepted: File[], rejected: { errors: readonly { code: string }[] }[]) => {
    if (rejected.length > 0) {
      const code = rejected[0]?.errors[0]?.code;
      if (code === "file-too-large") {
        toast.error("File too large — max 10 MB");
      } else {
        toast.error("Only JPG / PNG images accepted");
      }
      return;
    }
    if (!accepted[0]) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(accepted[0]);
    setPreviewUrl(URL.createObjectURL(accepted[0]));
    setResult(null);
    setError(null);
  }, [previewUrl]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"] },
    maxSize: MAX_FILE_BYTES,
    multiple: false,
    disabled: loading,
  });

  /* ── Solve handler ───────────────────────────────────────────── */
  async function handleSolve() {
    if (!file) { toast.error("Upload an image first"); return; }

    setLoading(true);
    setLoadingStep(0);
    setError(null);
    setResult(null);

    /* Step animation — cycles every 1.6s, max 3 (wait for real response) */
    let step = 0;
    stepTimerRef.current = setInterval(() => {
      step = Math.min(step + 1, LOADING_STEPS.length - 1);
      setLoadingStep(step);
    }, 1600);

    abortRef.current = new AbortController();

    try {
      const data = await solveSudoku(file, abortRef.current.signal);
      const lowConfidenceCount = data.cells.filter((cell) => cell.value > 0 && cell.confidence < 0.5).length;
      setResult(data);
      setActiveTab("solution");
      toast.success("Puzzle solved!");
      if (lowConfidenceCount > 0) {
        toast.warning(`Low confidence on ${lowConfidenceCount} cells`);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      const raw = err instanceof Error ? err.message : "Solve failed";
      const msg = /grid|contour|cell|decode/i.test(raw) ? "No grid detected" : raw;
      setError(msg);
      toast.error(msg);
    } finally {
      clearInterval(stepTimerRef.current!);
      setLoading(false);
      setLoadingStep(0);
    }
  }

  /* ── Copy grid ───────────────────────────────────────────────── */
  function handleCopy() {
    if (!result) return;
    navigator.clipboard
      .writeText(gridToText(result.solved_grid))
      .then(() => toast.success("Grid copied to clipboard"))
      .catch(() => toast.error("Copy failed"));
  }

  /* ── Render ──────────────────────────────────────────────────── */
  return (
    <div
      className="mx-auto w-full px-6 py-10"
      style={{ maxWidth: "1280px" }}
    >
      {/* Page header */}
      <div className="mb-10">
        <div
          className="mb-3 inline-flex items-center gap-2 rounded-full px-3 py-1"
          style={{
            background: "rgba(0,212,255,0.08)",
            border: "1px solid rgba(0,212,255,0.18)",
          }}
        >
          <span
            style={{
              width: 6, height: 6, borderRadius: "50%",
              background: "#00D4FF",
              boxShadow: "0 0 6px #00D4FF",
              display: "inline-block",
            }}
          />
          <span
            style={{
              fontFamily: "var(--font-space-mono), monospace",
              fontSize: "11px",
              color: "#00D4FF",
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            CV + CNN + Backtracking
          </span>
        </div>
        <h1 className="text-display" style={{ marginBottom: "8px" }}>
          Solve a Puzzle
        </h1>
        <p className="text-body" style={{ color: "var(--fg-muted)", maxWidth: "480px" }}>
          Upload any sudoku image — printed, hand-drawn, or photographed. English and Farsi digits supported.
        </p>
      </div>

      {/* Two-column layout */}
      <div
        className={result ? "grid gap-8 lg:grid-cols-[minmax(340px,420px)_1fr]" : "grid max-w-[480px] gap-8"}
        style={{ alignItems: "start" }}
      >
        {/* ── LEFT PANEL ──────────────────────────────────────────── */}
        <div className="flex flex-col gap-4">

          {/* DropZone */}
          <div
            {...getRootProps()}
            tabIndex={0}
            role="button"
            aria-label="Upload sudoku image"
            className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#00D4FF]"
            style={{
              borderRadius: "var(--r-lg)",
              padding: "24px",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease",
              border: `1.5px dashed ${isDragActive ? "#00D4FF" : "rgba(0,212,255,0.22)"}`,
              background: isDragActive
                ? "rgba(0,212,255,0.06)"
                : "rgba(10,10,15,0.5)",
              backdropFilter: "saturate(180%) blur(12px)",
              WebkitBackdropFilter: "saturate(180%) blur(12px)",
              boxShadow: isDragActive
                ? "0 0 0 1px rgba(0,212,255,0.35), 0 0 32px rgba(0,212,255,0.08)"
                : "none",
              outline: "none",
              position: "relative",
              overflow: "hidden",
            }}
          >
            <input {...getInputProps()} />

            {/* Scan line on drag */}
            {isDragActive && (
              <div
                aria-hidden="true"
                style={{
                  position: "absolute",
                  left: 0, right: 0,
                  height: "2px",
                  background: "linear-gradient(90deg, transparent, #00D4FF, transparent)",
                  animation: "scan 1s linear infinite",
                  top: 0,
                }}
              />
            )}

            {previewUrl ? (
              /* Image preview */
              <div className="flex flex-col items-center gap-3">
                <div
                  style={{
                    borderRadius: "var(--r-md)",
                    overflow: "hidden",
                    border: "1px solid var(--border-col)",
                    boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
                    maxHeight: "240px",
                    width: "100%",
                  }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Uploaded sudoku preview"
                    style={{
                      width: "100%",
                      height: "240px",
                      objectFit: "contain",
                      background: "#0e0e16",
                      display: "block",
                    }}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                    <circle cx="7" cy="7" r="6" stroke="#00E396" strokeWidth="1.2"/>
                    <path d="M4 7l2 2 4-4" stroke="#00E396" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span
                    className="text-caption"
                    style={{
                      color: "var(--fg-muted)",
                      fontFamily: "var(--font-space-mono), monospace",
                      fontSize: "11px",
                    }}
                  >
                    {file?.name} · {((file?.size ?? 0) / 1024).toFixed(0)} KB
                  </span>
                </div>
                <span className="text-caption" style={{ color: "var(--fg-muted)", fontSize: "11px" }}>
                  Click or drag to replace
                </span>
              </div>
            ) : (
              /* Empty state */
              <div className="flex flex-col items-center gap-4 py-8 text-center">
                <div
                  style={{
                    width: 56, height: 56,
                    borderRadius: "var(--r-md)",
                    background: "rgba(0,212,255,0.06)",
                    border: `1px solid ${isDragActive ? "rgba(0,212,255,0.5)" : "rgba(0,212,255,0.15)"}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "border-color 0.2s ease",
                  }}
                >
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path
                      d="M12 16V8M8 12l4-4 4 4"
                      stroke={isDragActive ? "#00D4FF" : "rgba(0,212,255,0.7)"}
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      style={{ transition: "stroke 0.2s ease" }}
                    />
                    <rect
                      x="3" y="3" width="18" height="18" rx="3"
                      stroke={isDragActive ? "#00D4FF" : "rgba(0,212,255,0.25)"}
                      strokeWidth="1.2"
                      style={{ transition: "stroke 0.2s ease" }}
                    />
                  </svg>
                </div>
                <div className="flex flex-col gap-1">
                  <span
                    style={{
                      fontFamily: "var(--font-space-mono), monospace",
                      fontSize: "14px",
                      fontWeight: 700,
                      color: isDragActive ? "#00D4FF" : "var(--fg)",
                      transition: "color 0.2s ease",
                    }}
                  >
                    {isDragActive ? "Drop to upload" : "Drop image here"}
                  </span>
                  <span className="text-caption" style={{ color: "var(--fg-muted)" }}>
                    or click to browse · JPG, PNG · max 10 MB
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Solve button */}
          <button
            onClick={handleSolve}
            disabled={!file || loading}
            className="btn-press w-full focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#00D4FF]"
            aria-label="Solve uploaded sudoku"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              background: file && !loading ? "var(--cyan)" : "rgba(0,212,255,0.15)",
              color: file && !loading ? "#0A0A0F" : "rgba(0,212,255,0.4)",
              fontSize: "17px",
              fontWeight: 600,
              lineHeight: 1,
              letterSpacing: "-0.374px",
              borderRadius: "var(--r-pill)",
              padding: "16px 28px",
              minHeight: "48px",
              border: "none",
              cursor: file && !loading ? "pointer" : "not-allowed",
              transition: "background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease",
              boxShadow: file && !loading ? "0 0 24px rgba(0,212,255,0.2)" : "none",
            }}
            onMouseEnter={(e) => {
              if (file && !loading) {
                (e.currentTarget as HTMLButtonElement).style.background = "#1ADCFF";
                (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 0 32px rgba(0,212,255,0.35)";
              }
            }}
            onMouseLeave={(e) => {
              if (file && !loading) {
                (e.currentTarget as HTMLButtonElement).style.background = "var(--cyan)";
                (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 0 24px rgba(0,212,255,0.2)";
              }
            }}
          >
            {loading ? (
              <>
                <Spinner />
                Processing…
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Solve Sudoku
              </>
            )}
          </button>

          {/* Loading step labels */}
          {loading && <LoadingState step={loadingStep} />}

          {/* Error card */}
          {error && !loading && <ErrorCard message={error} />}

          {loading && (
            <div className="grid gap-4 rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#12121A]/60 p-4">
              <SudokuBoardSkeleton />
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ─────────────────────────────────────────── */}
        {result && (
          <div
            className="animate-fade-in flex flex-col gap-5"
            style={{ animationDuration: "0.4s" }}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <circle cx="8" cy="8" r="7" stroke="#00E396" strokeWidth="1.2"/>
                  <path d="M5 8l2 2 4-4" stroke="#00E396" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span
                  style={{
                    fontFamily: "var(--font-space-mono), monospace",
                    fontSize: "13px",
                    fontWeight: 700,
                    color: "#00E396",
                    letterSpacing: "-0.02em",
                  }}
                >
                  Puzzle solved
                </span>
              </div>
              <TabBar active={activeTab} onChange={setActiveTab} />
            </div>

            {/* Tab panels */}
            <div
              className="card-dark p-5"
              style={{ minHeight: "400px" }}
            >
              {activeTab === "solution" && (
                <SolutionTab result={result} onCopy={handleCopy} />
              )}
              {activeTab === "cells" && (
                <CellAnalysisTab cells={result.cells} />
              )}
              {activeTab === "raw" && (
                <RawGridTab grid={result.original_grid} cells={result.cells} />
              )}
            </div>
          </div>
        )}
        {loading && !result && (
          <div className="animate-fade-in flex flex-col gap-5">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="h-5 w-32 animate-pulse rounded-full bg-white/[0.08]" />
              <div className="h-12 w-64 animate-pulse rounded-[var(--r-pill)] bg-white/[0.06]" />
            </div>
            <div className="card-dark grid gap-5 overflow-x-auto p-5">
              <SudokuBoardSkeleton />
              <CellGridSkeleton />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
