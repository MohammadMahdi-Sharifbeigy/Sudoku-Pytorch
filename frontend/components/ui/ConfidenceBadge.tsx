"use client";

export interface ConfidenceBadgeProps {
  confidence: number;
}

function confidenceTone(confidence: number) {
  if (confidence >= 0.8) {
    return {
      label: "High confidence",
      color: "#00E396",
      background: "linear-gradient(135deg, rgba(0,227,150,0.18), rgba(0,227,150,0.06))",
      border: "rgba(0,227,150,0.28)",
    };
  }

  if (confidence >= 0.5) {
    return {
      label: "Medium confidence",
      color: "#FFB800",
      background: "linear-gradient(135deg, rgba(255,184,0,0.18), rgba(255,184,0,0.06))",
      border: "rgba(255,184,0,0.3)",
    };
  }

  return {
    label: "Low confidence",
    color: "#FF4560",
    background: "linear-gradient(135deg, rgba(255,69,96,0.18), rgba(255,69,96,0.06))",
    border: "rgba(255,69,96,0.3)",
  };
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const normalized = Math.max(0, Math.min(1, confidence));
  const tone = confidenceTone(normalized);

  return (
    <span
      className="inline-flex h-8 items-center gap-2 rounded-[var(--r-pill)] border px-3 text-caption backdrop-blur-xl"
      style={{
        color: tone.color,
        background: tone.background,
        borderColor: tone.border,
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08)",
      }}
      aria-label={`${tone.label}: ${(normalized * 100).toFixed(0)} percent`}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{
          background: tone.color,
          boxShadow: `0 0 12px ${tone.color}66`,
        }}
      />
      <span className="font-mono text-[12px] font-bold tracking-[-0.02em]">
        {(normalized * 100).toFixed(0)}%
      </span>
    </span>
  );
}
