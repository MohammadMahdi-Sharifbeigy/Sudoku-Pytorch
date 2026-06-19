"use client";

import { useId } from "react";
import type { ModelEntry } from "@/lib/api";

export interface ModelSelectProps {
  label: string;
  models: ModelEntry[];
  value: string | null;
  onChange: (id: string | null) => void;
  loading?: boolean;
  disabled?: boolean;
}

/** Format a model entry for its <option> label, e.g. "best_model.pt · 1.2 MB (default)". */
function optionLabel(model: ModelEntry): string {
  const base = `${model.filename} · ${model.size_mb} MB`;
  return model.is_default ? `${base} (default)` : base;
}

/**
 * Controlled, themed model picker. The first option ("Default (server)")
 * maps to `null`; every other option carries a model `id`.
 */
export function ModelSelect({
  label,
  models,
  value,
  onChange,
  loading = false,
  disabled = false,
}: ModelSelectProps) {
  const selectId = useId();
  const isDisabled = disabled || loading;

  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={selectId}
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
        {loading ? (
          <span style={{ color: "var(--cyan)", marginLeft: 6 }}>· loading…</span>
        ) : null}
      </label>

      <div style={{ position: "relative" }}>
        <select
          id={selectId}
          value={value ?? ""}
          disabled={isDisabled}
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
          className="w-full appearance-none transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#00D4FF]"
          style={{
            minHeight: "44px",
            padding: "0 40px 0 14px",
            borderRadius: "var(--r-md)",
            background: "var(--surface)",
            border: "1px solid var(--border-col)",
            color: "var(--fg)",
            fontFamily: "var(--font-space-mono), monospace",
            fontSize: "13px",
            letterSpacing: "-0.01em",
            cursor: isDisabled ? "not-allowed" : "pointer",
            opacity: isDisabled ? 0.55 : 1,
          }}
          onMouseEnter={(e) => {
            if (!isDisabled) e.currentTarget.style.borderColor = "rgba(0,212,255,0.45)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "var(--border-col)";
          }}
        >
          <option value="">Default (server)</option>
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {optionLabel(model)}
            </option>
          ))}
        </select>

        {/* Chevron — purely decorative, click passes through to the select */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          fill="none"
          aria-hidden="true"
          style={{
            position: "absolute",
            right: "14px",
            top: "50%",
            transform: "translateY(-50%)",
            pointerEvents: "none",
            opacity: isDisabled ? 0.4 : 0.8,
          }}
        >
          <path
            d="M3.5 5.25 7 8.75l3.5-3.5"
            stroke="var(--cyan)"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </div>
  );
}
