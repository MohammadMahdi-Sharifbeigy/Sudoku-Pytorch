"use client";

import { useState } from "react";
import { Check, Clipboard } from "lucide-react";

export interface CodeBlockProps {
  code: string;
  language: string;
}

const KEYWORDS = /\b(import|from|const|let|with|return|if|else|as|None|true|false|providers|model|session)\b/g;
const STRINGS = /("[^"]*"|'[^']*')/g;

function tokenize(line: string): Array<{ text: string; tone: "plain" | "string" | "keyword" }> {
  const tokens: Array<{ text: string; tone: "plain" | "string" | "keyword" }> = [];
  const matches = Array.from(line.matchAll(new RegExp(`${STRINGS.source}|${KEYWORDS.source}`, "g")));
  let cursor = 0;

  for (const match of matches) {
    const text = match[0];
    const index = match.index ?? 0;
    if (index > cursor) tokens.push({ text: line.slice(cursor, index), tone: "plain" });
    tokens.push({ text, tone: STRINGS.test(text) ? "string" : "keyword" });
    STRINGS.lastIndex = 0;
    cursor = index + text.length;
  }

  if (cursor < line.length) tokens.push({ text: line.slice(cursor), tone: "plain" });
  return tokens;
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const lines = code.split("\n");

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div
      className="group relative overflow-hidden rounded-[var(--r-lg)] border bg-[#08080D]"
      style={{
        borderColor: "rgba(255,255,255,0.08)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.06), 0 22px 70px rgba(0,0,0,0.2)",
      }}
    >
      <div className="flex h-11 items-center justify-between border-b border-[#1E1E2E] px-4">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-[#FF4560]/80" />
          <span className="h-3 w-3 rounded-full bg-[#FFB800]/80" />
          <span className="h-3 w-3 rounded-full bg-[#00E396]/80" />
        </div>
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#5A5A7A]">
          {language}
        </span>
      </div>
      <button
        type="button"
        onClick={copyCode}
        className="btn-press absolute right-3 top-14 z-10 inline-flex h-9 translate-y-1 items-center gap-2 rounded-[var(--r-pill)] border border-[#00D4FF]/25 bg-[#00D4FF]/10 px-3 text-caption text-[#00D4FF] opacity-0 backdrop-blur-xl transition duration-200 hover:border-[#00D4FF]/60 hover:bg-[#00D4FF]/15 group-hover:translate-y-0 group-hover:opacity-100 focus:translate-y-0 focus:opacity-100"
        aria-label="Copy code"
      >
        {copied ? <Check className="size-3.5" /> : <Clipboard className="size-3.5" />}
        {copied ? "Copied" : "Copy"}
      </button>
      <pre
        className="overflow-x-auto p-5 pr-24 text-sm leading-6"
        style={{
          color: "#DDE7EF",
          fontFamily: "var(--font-space-mono), monospace",
        }}
      >
        <code>
          {lines.map((line, lineIndex) => (
            <span key={`${line}-${lineIndex}`}>
              {tokenize(line).map((token, tokenIndex) => (
                <span
                  key={`${token.text}-${tokenIndex}`}
                  style={{
                    color:
                      token.tone === "string"
                        ? "#FFB800"
                        : token.tone === "keyword"
                          ? "#00D4FF"
                          : undefined,
                  }}
                >
                  {token.text}
                </span>
              ))}
              {lineIndex < lines.length - 1 ? "\n" : null}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
