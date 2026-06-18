"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="grid min-h-[70dvh] place-items-center px-6 py-16">
      <div className="max-w-xl rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#12121A]/80 p-8 text-center backdrop-blur-2xl">
        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#FF4560]">
          Page error
        </p>
        <h1 className="mt-3 text-display text-white">This grid needs another pass.</h1>
        <p className="mt-4 text-body text-[#C8C8D4]/70">
          Something failed while rendering this page. Try again or return home.
        </p>
        <div className="mt-6 flex justify-center gap-3">
          <button type="button" onClick={reset} className="btn-primary min-h-12">
            Try again
          </button>
          <Link href="/" className="btn-ghost-pill min-h-12">
            Home
          </Link>
        </div>
      </div>
    </main>
  );
}
