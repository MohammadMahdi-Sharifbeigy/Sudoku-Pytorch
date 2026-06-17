"use client";

import { useEffect, useState } from "react";

const PUZZLE: (number | null)[] = [
  5, 3, null, null, 7, null, null, null, null,
  6, null, null, 1, 9, 5, null, null, null,
  null, 9, 8, null, null, null, null, 6, null,
  8, null, null, null, 6, null, null, null, 3,
  4, null, null, 8, null, 3, null, null, 1,
  7, null, null, null, 2, null, null, null, 6,
  null, 6, null, null, null, null, 2, 8, null,
  null, null, null, 4, 1, 9, null, null, 5,
  null, null, null, null, 8, null, null, 7, 9,
];

const SOLUTION: number[] = [
  5, 3, 4, 6, 7, 8, 9, 1, 2,
  6, 7, 2, 1, 9, 5, 3, 4, 8,
  1, 9, 8, 3, 4, 2, 5, 6, 7,
  8, 5, 9, 7, 6, 1, 4, 2, 3,
  4, 2, 6, 8, 5, 3, 7, 9, 1,
  7, 1, 3, 9, 2, 4, 8, 5, 6,
  9, 6, 1, 5, 3, 7, 2, 8, 4,
  2, 8, 7, 4, 1, 9, 6, 3, 5,
  3, 4, 5, 2, 8, 6, 1, 7, 9,
];

function cellBorder(idx: number) {
  const row = Math.floor(idx / 9);
  const col = idx % 9;
  const borderRight  = col < 8 ? (col % 3 === 2 ? "border-r-[1.5px] border-r-[#2A2A3E]" : "border-r border-r-[#1E1E2E]") : "";
  const borderBottom = row < 8 ? (row % 3 === 2 ? "border-b-[1.5px] border-b-[#2A2A3E]" : "border-b border-b-[#1E1E2E]") : "";
  return `${borderRight} ${borderBottom}`;
}

export function SudokuPreview() {
  const [revealed, setRevealed] = useState<Set<number>>(new Set());

  useEffect(() => {
    const empty = PUZZLE.map((v, i) => (v === null ? i : -1)).filter((i) => i >= 0);
    let t = 0;
    const timers: ReturnType<typeof setTimeout>[] = [];

    for (const idx of empty) {
      const delay = t++ * 60 + Math.random() * 30;
      timers.push(
        setTimeout(() => {
          setRevealed((prev) => new Set([...prev, idx]));
        }, delay)
      );
    }

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div
      className="select-none"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(9, 1fr)",
        border: "1.5px solid #2A2A3E",
        borderRadius: "var(--r-md)",
        overflow: "hidden",
        width: "min(340px, 90vw)",
        aspectRatio: "1",
        background: "#0A0A0F",
      }}
      role="img"
      aria-label="Animated sudoku puzzle being solved"
    >
      {PUZZLE.map((given, idx) => {
        const isSolved = given === null && revealed.has(idx);
        return (
          <div
            key={idx}
            className={`flex items-center justify-center ${cellBorder(idx)}`}
            style={{ aspectRatio: "1" }}
          >
            {given !== null ? (
              <span
                style={{
                  fontFamily: "var(--font-space-mono), monospace",
                  fontSize: "clamp(10px, 1.8vw, 15px)",
                  fontWeight: 700,
                  color: "#C8C8D4",
                }}
              >
                {given}
              </span>
            ) : isSolved ? (
              <span
                className="animate-cell-reveal"
                style={{
                  fontFamily: "var(--font-space-mono), monospace",
                  fontSize: "clamp(10px, 1.8vw, 15px)",
                  fontWeight: 400,
                  color: "#00D4FF",
                  textShadow: "0 0 8px rgba(0,212,255,0.5)",
                }}
              >
                {SOLUTION[idx]}
              </span>
            ) : (
              <span
                style={{
                  width: "4px",
                  height: "4px",
                  borderRadius: "50%",
                  background: "#1E1E2E",
                  display: "block",
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
