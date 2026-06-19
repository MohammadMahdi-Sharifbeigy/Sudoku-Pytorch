"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  AlertTriangle,
  ArrowDownToLine,
  CheckCircle2,
  ChevronDown,
  Cpu,
  FileArchive,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Server,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { ModelSelect } from "@/components/ui/ModelSelect";
import { StatCard } from "@/components/ui/StatCard";
import {
  getModelInfo,
  listCnnModels,
  modelDownloadUrl,
  runOptimization,
  type BenchmarkEntry,
  type ModelEntry,
  type ModelInfo,
} from "@/lib/api";

type LoadState = "idle" | "loading" | "ready" | "error";
type BenchmarkState = "idle" | "running" | "complete" | "error";
type AccordionId = "torchscript" | "onnx";

const FORMAT_LABELS: Record<string, string> = {
  pt: "PyTorch",
  ts: "TorchScript",
  onnx: "ONNX",
};

const FORMAT_NOTES: Record<string, string> = {
  pt: "Reference checkpoint for Python training and recovery.",
  ts: "Portable PyTorch graph for Python or C++ inference.",
  onnx: "Runtime-neutral graph for edge and service deployment.",
};

const TORCHSCRIPT_SNIPPET = `import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("models/best_model.ts", map_location=device)
model.eval()

with torch.inference_mode():
    logits = model(batch.to(device))
    prediction = logits.argmax(dim=1)`;

const ONNX_SNIPPET = `import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "models/best_model.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: batch.astype(np.float32)})
prediction = outputs[0].argmax(axis=1)`;

function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US").format(value);
}

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toFixed(digits);
}

function formatDate(value: string | null): string {
  if (!value) return "Not trained";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function GlassPanel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-[var(--r-lg)] border border-[#1E1E2E] bg-[#12121A]/75 backdrop-blur-2xl ${className}`}
      style={{
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.055)",
      }}
    >
      {children}
    </section>
  );
}

function StatusPill({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone: "cyan" | "green" | "amber" | "red";
}) {
  const color =
    tone === "green"
      ? "#00E396"
      : tone === "amber"
        ? "#FFB800"
        : tone === "red"
          ? "#FF4560"
          : "#00D4FF";

  return (
    <span
      className="inline-flex h-8 items-center gap-2 rounded-[var(--r-pill)] border px-3 text-caption"
      style={{
        color,
        borderColor: `${color}38`,
        background: `${color}14`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {children}
    </span>
  );
}

function LoadingSpinner() {
  return (
    <span className="relative grid h-5 w-5 place-items-center">
      <span
        className="absolute h-5 w-5 rounded-full border border-[#0A0A0F]/20 border-t-[#0A0A0F]"
        style={{ animation: "opt-spin 0.9s linear infinite" }}
      />
      <span className="h-1.5 w-1.5 rounded-full bg-[#0A0A0F]" />
    </span>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value?: number; payload?: BenchmarkEntry }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;

  const entry = payload[0]?.payload;

  return (
    <div
      className="rounded-[var(--r-md)] border px-4 py-3 text-caption backdrop-blur-xl"
      style={{
        background: "rgba(10,10,15,0.84)",
        borderColor: "rgba(255,255,255,0.12)",
        boxShadow: "0 18px 50px rgba(0,0,0,0.34)",
      }}
    >
      <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#5A5A7A]">
        {FORMAT_LABELS[String(label)] ?? label}
      </div>
      <div className="flex items-center justify-between gap-6 text-[#C8C8D4]">
        <span>Latency</span>
        <span className="font-mono text-white">{formatMetric(entry?.latency_ms, 3)} ms</span>
      </div>
      <div className="mt-1 flex items-center justify-between gap-6 text-[#C8C8D4]">
        <span>Size</span>
        <span className="font-mono text-white">{formatMetric(entry?.size_mb, 3)} MB</span>
      </div>
    </div>
  );
}

function LatencyChart({ data }: { data: BenchmarkEntry[] }) {
  const chartData = data.filter((entry) => entry.latency_ms != null);

  return (
    <div
      className="min-h-[340px] rounded-[var(--r-lg)] border p-5"
      style={{
        background: "rgba(255,255,255,0.035)",
        borderColor: "rgba(255,255,255,0.07)",
      }}
    >
      <div className="mb-5">
        <h2 className="font-mono text-xl font-bold text-white">Latency comparison</h2>
        <p className="text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
          Lower is better. Each bar uses the backend benchmark result.
        </p>
      </div>

      <div className="h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.045)" />
            <XAxis
              dataKey="format"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(200,200,212,0.52)", fontSize: 12 }}
              tickFormatter={(value) => FORMAT_LABELS[String(value)] ?? String(value).toUpperCase()}
              tickMargin={12}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(200,200,212,0.42)", fontSize: 12 }}
              tickFormatter={(value) => `${Number(value).toFixed(1)}ms`}
              width={62}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.035)" }} />
            <Bar
              dataKey="latency_ms"
              fill="#00D4FF"
              radius={[10, 10, 4, 4]}
              isAnimationActive
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CodeAccordion({
  id,
  title,
  description,
  code,
  open,
  onToggle,
}: {
  id: AccordionId;
  title: string;
  description: string;
  code: string;
  open: boolean;
  onToggle: (id: AccordionId) => void;
}) {
  return (
    <div
      className="overflow-hidden rounded-[var(--r-lg)] border transition-colors"
      style={{
        borderColor: open ? "rgba(0,212,255,0.28)" : "rgba(255,255,255,0.07)",
        background: open ? "rgba(0,212,255,0.045)" : "rgba(255,255,255,0.025)",
      }}
    >
      <button
        type="button"
        className="btn-press flex w-full items-center justify-between gap-4 px-5 py-4 text-left"
        onClick={() => onToggle(id)}
        aria-expanded={open}
      >
        <span>
          <span className="block font-mono text-lg font-bold text-white">{title}</span>
          <span className="mt-1 block text-caption" style={{ color: "rgba(200,200,212,0.58)" }}>
            {description}
          </span>
        </span>
        <ChevronDown
          className="size-5 shrink-0 text-[#00D4FF] transition-transform duration-300"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>

      <div
        className="grid transition-[grid-template-rows] duration-300 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="px-4 pb-4 sm:px-5 sm:pb-5">
            <CodeBlock code={code} language={title} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function OptimizePage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelState, setModelState] = useState<LoadState>("idle");
  const [benchmarkState, setBenchmarkState] = useState<BenchmarkState>("idle");
  const [results, setResults] = useState<BenchmarkEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cnnModels, setCnnModels] = useState<ModelEntry[]>([]);
  const [cnnModel, setCnnModel] = useState<string | null>(null);
  const [openAccordions, setOpenAccordions] = useState<Record<AccordionId, boolean>>({
    torchscript: true,
    onnx: false,
  });

  const loadModelInfo = useCallback(async (signal?: AbortSignal) => {
    setModelState("loading");
    setError(null);

    try {
      const info = await getModelInfo(signal);
      setModelInfo(info);
      setModelState("ready");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof Error ? err.message : "Failed to load model info";
      setError(message);
      setModelState("error");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadInitialModelInfo() {
      setModelState("loading");
      setError(null);

      try {
        const info = await getModelInfo(controller.signal);
        setModelInfo(info);
        setModelState("ready");
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message = err instanceof Error ? err.message : "Failed to load model info";
        setError(message);
        setModelState("error");
      }
    }

    void loadInitialModelInfo();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    listCnnModels(controller.signal)
      .then((models) => { if (active) setCnnModels(models); })
      .catch(() => { if (active) setCnnModels([]); });

    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  const fastest = useMemo(() => {
    const measurable = results.filter((entry) => entry.latency_ms != null);
    if (!measurable.length) return null;

    return measurable.reduce((best, entry) =>
      Number(entry.latency_ms) < Number(best.latency_ms) ? entry : best,
    );
  }, [results]);

  const handleBenchmark = async () => {
    setBenchmarkState("running");
    setError(null);
    setResults([]);

    try {
      const data = await runOptimization(cnnModel ?? undefined);
      setResults(data.results);
      setBenchmarkState("complete");
      toast.success("Model exported!");
      void loadModelInfo();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Optimization failed";
      setError(message);
      setBenchmarkState("error");
      toast.error(`Model export failed: ${message}`);
    }
  };

  const toggleAccordion = (id: AccordionId) => {
    setOpenAccordions((current) => ({ ...current, [id]: !current[id] }));
  };

  return (
    <div className="min-h-dvh overflow-hidden">
      <style>{`
        @keyframes opt-spin {
          to { transform: rotate(360deg); }
        }
        @keyframes opt-shimmer {
          0% { transform: translateX(-120%); opacity: .24; }
          100% { transform: translateX(260%); opacity: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          .opt-motion { animation: none !important; transition: none !important; transform: none !important; }
        }
      `}</style>

      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <GlassPanel className="p-5 sm:p-7 lg:p-8">
            <div className="mb-8 inline-flex items-center gap-2 rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] px-3 py-1.5 text-caption text-[#00D4FF]">
              <Zap className="size-3.5" />
              Export and benchmark
            </div>

            <h1 className="text-display max-w-4xl text-white">Optimization panel</h1>
            <p className="mt-4 max-w-3xl text-lead" style={{ color: "rgba(200,200,212,0.72)" }}>
              Convert the trained digit CNN into deployable artifacts and compare runtime latency before shipping.
            </p>

            <div className="mt-8 grid gap-4 md:grid-cols-2">
              <div className="rounded-[var(--r-lg)] border border-[#1E1E2E] bg-black/20 p-5">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full border border-[#00D4FF]/20 bg-[#00D4FF]/10 text-[#00D4FF]">
                  <Layers3 className="size-5" />
                </div>
                <h2 className="font-mono text-xl font-bold text-white">TorchScript</h2>
                <p className="mt-2 text-caption" style={{ color: "rgba(200,200,212,0.62)" }}>
                  Best when the deployment stack is still PyTorch-native. It keeps model behavior close to the training environment and works cleanly with Python or C++ runtime paths.
                </p>
              </div>

              <div className="rounded-[var(--r-lg)] border border-[#1E1E2E] bg-black/20 p-5">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full border border-[#FFB800]/20 bg-[#FFB800]/10 text-[#FFB800]">
                  <Server className="size-5" />
                </div>
                <h2 className="font-mono text-xl font-bold text-white">ONNX</h2>
                <p className="mt-2 text-caption" style={{ color: "rgba(200,200,212,0.62)" }}>
                  Best when inference needs a neutral runtime, smaller service footprint, or portability across CPU providers and edge deployments.
                </p>
              </div>
            </div>
          </GlassPanel>

          <GlassPanel className="flex flex-col justify-between p-5 sm:p-6">
            <div>
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#5A5A7A]">
                    Model status
                  </p>
                  <h2 className="mt-1 font-mono text-2xl font-bold text-white">
                    {modelState === "loading" ? "Checking model" : modelInfo?.exists ? "Ready to optimize" : "No checkpoint found"}
                  </h2>
                </div>
                {modelInfo?.exists ? (
                  <StatusPill tone="green">Available</StatusPill>
                ) : modelState === "loading" ? (
                  <StatusPill tone="cyan">Loading</StatusPill>
                ) : (
                  <StatusPill tone="amber">Train first</StatusPill>
                )}
              </div>

              {modelInfo?.exists === false ? (
                <div
                  className="mb-5 rounded-[var(--r-md)] border px-4 py-3 text-caption"
                  style={{
                    borderColor: "rgba(255,184,0,0.28)",
                    background: "rgba(255,184,0,0.08)",
                    color: "#FFE2A3",
                  }}
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="mt-0.5 size-4 shrink-0 text-[#FFB800]" />
                    <span>
                      No trained model was found. Run a training session on the{" "}
                      <Link href="/train" className="font-semibold text-[#FFB800] underline-offset-4 hover:underline">
                        training dashboard
                      </Link>{" "}
                      before exporting optimized artifacts.
                    </span>
                  </div>
                </div>
              ) : null}

              <div className="grid gap-3 sm:grid-cols-2">
                <StatCard
                  label="File size"
                  value={formatMetric(modelInfo?.size_mb, 3)}
                  unit="MB"
                  icon={<FileArchive className="size-4" />}
                />
                <StatCard
                  label="Parameters"
                  value={formatNumber(modelInfo?.total_params)}
                  icon={<Cpu className="size-4" />}
                />
                <StatCard
                  label="Trainable"
                  value={formatNumber(modelInfo?.trainable_params)}
                  icon={<Gauge className="size-4" />}
                />
                <StatCard
                  label="Last trained"
                  value={formatDate(modelInfo?.created_at ?? null)}
                  icon={<CheckCircle2 className="size-4" />}
                />
              </div>
            </div>

            <div className="mt-6 grid gap-3">
              <ModelSelect
                label="Model to optimize"
                models={cnnModels}
                value={cnnModel}
                onChange={setCnnModel}
                disabled={benchmarkState === "running"}
              />
              <button
                type="button"
                onClick={handleBenchmark}
                disabled={!modelInfo?.exists || benchmarkState === "running"}
                className="opt-motion btn-press relative flex h-14 w-full items-center justify-center gap-3 overflow-hidden rounded-[var(--r-pill)] bg-[#00D4FF] px-6 text-[17px] font-bold text-[#0A0A0F] transition hover:-translate-y-0.5 hover:bg-[#1ADCFF] disabled:cursor-not-allowed disabled:opacity-55"
                style={{
                  boxShadow:
                    benchmarkState === "running"
                      ? "0 18px 54px rgba(0,212,255,0.16)"
                      : "0 14px 40px rgba(0,212,255,0.13)",
                }}
              >
                {benchmarkState === "running" ? (
                  <>
                    <span
                      className="opt-motion absolute inset-y-0 left-0 w-1/3 bg-white/30"
                      style={{ animation: "opt-shimmer 1.4s ease-in-out infinite" }}
                    />
                    <LoadingSpinner />
                    Optimizing model
                  </>
                ) : (
                  <>
                    <Play className="size-4 fill-[#0A0A0F]" />
                    Run Optimization & Benchmark
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => void loadModelInfo()}
                className="btn-press inline-flex h-10 items-center justify-center gap-2 rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] px-4 text-caption text-[#C8C8D4] transition hover:border-[#2A2A3E] hover:text-white"
              >
                <RefreshCw className="size-3.5" />
                Refresh model status
              </button>
            </div>

            {error ? (
              <div
                className="mt-4 rounded-[var(--r-md)] border px-4 py-3 text-caption"
                style={{
                  borderColor: "rgba(255,69,96,0.24)",
                  background: "rgba(255,69,96,0.07)",
                  color: "#FFD0D7",
                }}
                role="alert"
              >
                {error}
              </div>
            ) : null}
          </GlassPanel>
        </div>

        {benchmarkState === "complete" && results.length ? (
          <GlassPanel className="opt-motion animate-fade-up p-5 sm:p-6">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="mb-3 inline-flex items-center gap-2 rounded-[var(--r-pill)] border border-[#00E396]/20 bg-[#00E396]/10 px-3 py-1.5 text-caption text-[#00E396]">
                  <CheckCircle2 className="size-3.5" />
                  Benchmark complete
                </div>
                <h2 className="text-display text-white">Export results</h2>
                <p className="mt-2 text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
                  Fastest format:{" "}
                  <span className="font-mono text-[#FFB800]">
                    {fastest ? FORMAT_LABELS[fastest.format] ?? fastest.format.toUpperCase() : "--"}
                  </span>
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                {(["pt", "ts", "onnx"] as const).map((format) => (
                  <a
                    key={format}
                    href={modelDownloadUrl(format)}
                    className="btn-press inline-flex h-10 items-center gap-2 rounded-[var(--r-pill)] border border-[#00D4FF]/30 bg-[#00D4FF]/10 px-4 text-caption font-semibold text-[#00D4FF] transition hover:border-[#00D4FF] hover:bg-[#00D4FF]/15"
                  >
                    <ArrowDownToLine className="size-3.5" />
                    {format.toUpperCase()}
                  </a>
                ))}
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
              <div className="overflow-hidden rounded-[var(--r-lg)] border border-[#1E1E2E]">
                <table className="w-full min-w-[720px] border-separate border-spacing-0">
                  <thead>
                    <tr style={{ background: "rgba(255,255,255,0.04)" }}>
                      {["Format", "Size (MB)", "Latency (ms)", "Status"].map((heading) => (
                        <th
                          key={heading}
                          className="px-5 py-4 text-left text-caption font-semibold text-[#5A5A7A]"
                        >
                          {heading}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((entry) => {
                      const isFastest = fastest?.format === entry.format;

                      return (
                        <tr
                          key={entry.format}
                          className="transition-colors hover:bg-white/[0.035]"
                          style={{
                            background: isFastest ? "rgba(255,184,0,0.07)" : "transparent",
                            boxShadow: isFastest
                              ? "inset 3px 0 0 rgba(255,184,0,0.72)"
                              : "none",
                          }}
                        >
                          <td className="border-t border-[#1E1E2E] px-5 py-4">
                            <div className="flex flex-col gap-1">
                              <span className="font-mono font-bold text-white">
                                {FORMAT_LABELS[entry.format] ?? entry.format.toUpperCase()}
                              </span>
                              <span className="text-caption" style={{ color: "rgba(200,200,212,0.48)" }}>
                                {FORMAT_NOTES[entry.format] ?? entry.path}
                              </span>
                            </div>
                          </td>
                          <td className="border-t border-[#1E1E2E] px-5 py-4 font-mono text-[#C8C8D4]">
                            {formatMetric(entry.size_mb, 3)}
                          </td>
                          <td className="border-t border-[#1E1E2E] px-5 py-4 font-mono text-[#C8C8D4]">
                            {formatMetric(entry.latency_ms, 3)}
                          </td>
                          <td className="border-t border-[#1E1E2E] px-5 py-4">
                            {isFastest ? (
                              <StatusPill tone="amber">Fastest</StatusPill>
                            ) : entry.latency_ms == null ? (
                              <StatusPill tone="red">Unavailable</StatusPill>
                            ) : (
                              <StatusPill tone="cyan">Ready</StatusPill>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <LatencyChart data={results} />
            </div>
          </GlassPanel>
        ) : null}

        <GlassPanel className="p-5 sm:p-6">
          <div className="mb-5">
            <h2 className="font-mono text-2xl font-bold text-white">Deployment instructions</h2>
            <p className="mt-2 max-w-3xl text-caption" style={{ color: "rgba(200,200,212,0.58)" }}>
              Copy the runtime path that matches your deployment target. Replace the sample batch with your normalized 28 by 28 digit tensor.
            </p>
          </div>

          <div className="grid gap-4">
            <CodeAccordion
              id="torchscript"
              title="TorchScript runtime"
              description="Use this when the model stays inside a PyTorch serving stack."
              code={TORCHSCRIPT_SNIPPET}
              open={openAccordions.torchscript}
              onToggle={toggleAccordion}
            />
            <CodeAccordion
              id="onnx"
              title="ONNX Runtime"
              description="Use this for a neutral runtime with portable CPU inference."
              code={ONNX_SNIPPET}
              open={openAccordions.onnx}
              onToggle={toggleAccordion}
            />
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}
