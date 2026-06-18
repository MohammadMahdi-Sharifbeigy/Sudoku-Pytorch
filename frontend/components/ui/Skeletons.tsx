"use client";

export function SudokuBoardSkeleton() {
  return (
    <div
      className="grid aspect-square w-full animate-pulse overflow-hidden rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#0A0A0F]"
      style={{ gridTemplateColumns: "repeat(9, minmax(0, 1fr))" }}
      aria-label="Loading sudoku board"
    >
      {Array.from({ length: 81 }, (_, index) => (
        <div
          key={index}
          className="border-r border-b border-[#1E1E2E] bg-white/[0.035]"
        />
      ))}
    </div>
  );
}

export function CellGridSkeleton() {
  return (
    <div
      className="grid min-w-[620px] animate-pulse gap-0.5 rounded-[var(--r-md)] border border-[#1E1E2E] bg-[#1E1E2E] p-0.5"
      style={{ gridTemplateColumns: "repeat(9, minmax(0, 1fr))" }}
      aria-label="Loading cell grid"
    >
      {Array.from({ length: 81 }, (_, index) => (
        <div key={index} className="h-16 rounded-[4px] bg-white/[0.045]" />
      ))}
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div
      className="min-h-[320px] animate-pulse rounded-[var(--r-lg)] border border-[#1E1E2E] bg-white/[0.025] p-5"
      aria-label="Loading chart"
    >
      <div className="mb-6 h-5 w-36 rounded-full bg-white/[0.08]" />
      <div className="relative h-[230px] overflow-hidden rounded-[var(--r-md)] bg-black/20">
        <div className="absolute left-4 right-4 top-1/2 h-px bg-[#1E1E2E]" />
        <div className="absolute inset-x-6 bottom-10 h-24 rounded-[50%] border-t-2 border-[#00D4FF]/25" />
        <div className="absolute inset-x-10 bottom-16 h-20 rounded-[50%] border-t-2 border-[#FFB800]/20" />
      </div>
    </div>
  );
}
