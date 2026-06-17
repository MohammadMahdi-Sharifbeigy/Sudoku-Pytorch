import { Navbar } from "./Navbar";

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Navbar />
      <main className="flex-1">{children}</main>
      <footer
        className="border-t text-caption"
        style={{
          borderColor: "#1E1E2E",
          color: "#3A3A5A",
          padding: "32px 24px",
        }}
      >
        <div className="mx-auto flex max-w-[1440px] flex-col items-center gap-2 sm:flex-row sm:justify-between">
          <span
            style={{
              fontFamily: "var(--font-space-mono), monospace",
              fontSize: "12px",
              color: "#2A2A3E",
            }}
          >
            SudokuAI
          </span>
          <span style={{ fontSize: "12px", letterSpacing: "-0.12px" }}>
            PyTorch · FastAPI · Next.js · OpenCV
          </span>
        </div>
      </footer>
    </>
  );
}
