import type { Metadata } from "next";
import { Space_Mono, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Shell } from "@/components/layout/Shell";
import { GlobalErrorBoundary } from "@/components/layout/GlobalErrorBoundary";

const spaceMono = Space_Mono({
  weight: ["400", "700"],
  variable: "--font-space-mono",
  subsets: ["latin"],
  display: "swap",
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "SudokuAI — Solve Any Sudoku Instantly",
  description:
    "Upload a photo of any sudoku puzzle. Our CV + CNN pipeline detects the grid, recognises digits, and solves it in under 2ms.",
  keywords: ["sudoku", "solver", "AI", "computer vision", "pytorch", "deep learning"],
  openGraph: {
    title: "SudokuAI",
    description: "Solve any sudoku with computer vision + deep learning",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${spaceMono.variable} ${inter.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-dvh flex flex-col antialiased">
        <Providers>
          <GlobalErrorBoundary>
            <Shell>{children}</Shell>
          </GlobalErrorBoundary>
        </Providers>
      </body>
    </html>
  );
}
