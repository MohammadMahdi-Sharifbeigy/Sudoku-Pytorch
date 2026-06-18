"use client";

import type { ReactNode } from "react";

export interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon?: ReactNode;
}

export function StatCard({ label, value, unit, icon }: StatCardProps) {
  return (
    <div
      className="group rounded-[var(--r-lg)] border p-5 backdrop-blur-2xl transition duration-300 hover:-translate-y-0.5"
      style={{
        background: "linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.026))",
        borderColor: "rgba(255,255,255,0.08)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.07), 0 18px 55px rgba(0,0,0,0.18)",
      }}
    >
      <div className="mb-4 flex items-center justify-between gap-4">
        <p className="text-caption font-semibold" style={{ color: "rgba(200,200,212,0.6)" }}>
          {label}
        </p>
        {icon ? (
          <div
            className="grid h-9 w-9 place-items-center rounded-full border text-[#00D4FF] transition group-hover:border-[#00D4FF]/35 group-hover:bg-[#00D4FF]/10"
            style={{
              borderColor: "rgba(255,255,255,0.08)",
              background: "rgba(255,255,255,0.035)",
            }}
          >
            {icon}
          </div>
        ) : null}
      </div>
      <div className="flex items-end gap-2">
        <span
          className="font-mono text-3xl font-bold text-white"
          style={{ letterSpacing: "-0.04em", lineHeight: 1 }}
        >
          {value}
        </span>
        {unit ? (
          <span className="pb-0.5 text-caption font-semibold text-[#5A5A7A]">{unit}</span>
        ) : null}
      </div>
      <div className="mt-4 h-px w-full bg-gradient-to-r from-[#00D4FF]/40 via-transparent to-transparent opacity-50 transition group-hover:opacity-100" />
    </div>
  );
}
