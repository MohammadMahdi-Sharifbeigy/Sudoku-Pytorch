"use client";

import Link from "next/link";
import { Component, type ErrorInfo, type ReactNode } from "react";

interface GlobalErrorBoundaryProps {
  children: ReactNode;
}

interface GlobalErrorBoundaryState {
  hasError: boolean;
}

export class GlobalErrorBoundary extends Component<
  GlobalErrorBoundaryProps,
  GlobalErrorBoundaryState
> {
  state: GlobalErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): GlobalErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Unhandled frontend error", error, errorInfo);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <main className="grid min-h-[70dvh] place-items-center px-6 py-16">
        <div className="max-w-xl rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#12121A]/80 p-8 text-center backdrop-blur-2xl">
          <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#FF4560]">
            Runtime error
          </p>
          <h1 className="mt-3 text-display text-white">The board hit a conflict.</h1>
          <p className="mt-4 text-body text-[#C8C8D4]/70">
            Refresh the page to restore the app state. If it repeats, check the browser console for details.
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <button
              type="button"
              onClick={() => {
                this.setState({ hasError: false });
                window.location.reload();
              }}
              className="btn-primary min-h-12"
            >
              Refresh
            </button>
            <Link href="/" className="btn-ghost-pill min-h-12">
              Home
            </Link>
          </div>
        </div>
      </main>
    );
  }
}
