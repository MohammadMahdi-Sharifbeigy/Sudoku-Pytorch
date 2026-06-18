"use client";

export interface SudokuBoardProps {
  grid: number[][];
  highlights?: Set<string>;
}

function cellBorder(row: number, col: number): string {
  const right = col < 8 ? (col % 3 === 2 ? "2px solid #2A2A3E" : "1px solid #1E1E2E") : "0";
  const bottom = row < 8 ? (row % 3 === 2 ? "2px solid #2A2A3E" : "1px solid #1E1E2E") : "0";
  return `${right}|${bottom}`;
}

export function SudokuBoard({ grid, highlights }: SudokuBoardProps) {
  const cells = Array.from({ length: 81 }, (_, index) => {
    const row = Math.floor(index / 9);
    const col = index % 9;
    const value = grid[row]?.[col] ?? 0;
    const isHighlighted = highlights?.has(`${row},${col}`) ?? false;
    const isSolverFilled = value === 0;
    const [borderRight, borderBottom] = cellBorder(row, col).split("|");

    return (
      <div
        key={`${row}-${col}`}
        className="relative grid aspect-square place-items-center overflow-hidden"
        style={{
          borderRight,
          borderBottom,
          background: isHighlighted
            ? "radial-gradient(circle at center, rgba(255,184,0,0.18), rgba(255,184,0,0.055))"
            : isSolverFilled
              ? "rgba(0,212,255,0.035)"
              : "rgba(255,255,255,0.018)",
          boxShadow: isHighlighted ? "inset 0 0 18px rgba(255,184,0,0.14)" : "none",
        }}
      >
        {isHighlighted ? (
          <span className="absolute inset-1 rounded-[6px] border border-[#FFB800]/20 animate-glow-pulse" />
        ) : null}
        {value > 0 ? (
          <span
            className="font-mono font-bold"
            style={{
              color: isSolverFilled ? "#00D4FF" : "#F4F7FB",
              fontSize: "clamp(14px, 3.4vw, 28px)",
              lineHeight: 1,
              textRendering: "geometricPrecision",
              textShadow: isSolverFilled ? "0 0 12px rgba(0,212,255,0.38)" : "none",
            }}
          >
            {value}
          </span>
        ) : (
          <span className="h-1.5 w-1.5 rounded-full bg-[#1E1E2E]" />
        )}
      </div>
    );
  });

  return (
    <div
      className="grid select-none overflow-hidden rounded-[var(--r-lg)] border bg-[#0A0A0F]"
      style={{
        gridTemplateColumns: "repeat(9, minmax(0, 1fr))",
        borderColor: "#2A2A3E",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.05), 0 20px 70px rgba(0,0,0,0.22)",
      }}
      role="grid"
      aria-label="Sudoku board"
    >
      {cells}
    </div>
  );
}
