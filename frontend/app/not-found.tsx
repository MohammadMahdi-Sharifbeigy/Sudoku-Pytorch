import Link from "next/link";

export default function NotFound() {
  return (
    <main className="grid min-h-[70dvh] place-items-center px-6 py-16">
      <div className="max-w-xl rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#12121A]/80 p-8 text-center backdrop-blur-2xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#00D4FF]">
          404
        </p>
        <h1 className="mt-3 text-display text-white">This square has no number.</h1>
        <p className="mt-4 text-body text-[#C8C8D4]/70">
          The page you opened is not on the board. Return home and choose a valid move.
        </p>
        <Link href="/" className="btn-primary mt-6 min-h-12">
          Back home
        </Link>
      </div>
    </main>
  );
}
