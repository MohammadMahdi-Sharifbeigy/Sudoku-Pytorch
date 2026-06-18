"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  CheckCircle2,
  Play,
  RotateCcw,
  Sparkles,
  WifiOff,
} from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REPORT_URL = `${API_BASE}/api/reports/training_report.txt`;
const MAX_RETRIES = 3;

type DatasetMode = "mnist_fonts" | "mnist_hoda" | "all";
type TrainingStatus = "idle" | "connecting" | "training" | "complete" | "error";

interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  datasetMode: DatasetMode;
}

interface EpochPoint {
  epoch: number;
  epochs: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

interface TestResults {
  test_loss: number;
  test_acc: number;
  y_true: number[];
  y_pred: number[];
}

interface BestModel {
  epoch: number;
  val_loss: number;
}

type StreamEvent =
  | ({ type: "epoch" } & EpochPoint)
  | ({ type: "best_model" } & BestModel)
  | ({ type: "test_results" } & TestResults)
  | { type: "complete"; model_path: string }
  | { type: "error"; message: string };

const DATASETS: { value: DatasetMode; label: string }[] = [
  { value: "mnist_fonts", label: "MNIST + Fonts" },
  { value: "mnist_hoda", label: "MNIST + Hoda" },
  { value: "all", label: "All Datasets" },
];

const BATCH_SIZES = [32, 64, 128, 256] as const;

const DEFAULT_CONFIG: TrainingConfig = {
  epochs: 20,
  learningRate: 0.001,
  batchSize: 128,
  datasetMode: "mnist_fonts",
};

function formatMetric(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function formatPercent(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)}%` : "--";
}

function buildConfusionMatrix(yTrue: number[], yPred: number[]): number[][] {
  const matrix = Array.from({ length: 10 }, () => Array(10).fill(0) as number[]);

  yTrue.forEach((actual, index) => {
    const predicted = yPred[index];
    if (actual >= 0 && actual <= 9 && predicted >= 0 && predicted <= 9) {
      matrix[actual][predicted] += 1;
    }
  });

  return matrix;
}

function buildClassStats(yTrue: number[], yPred: number[]) {
  return Array.from({ length: 10 }, (_, classIndex) => {
    const total = yTrue.filter((value) => value === classIndex).length;
    const correct = yTrue.filter(
      (value, index) => value === classIndex && yPred[index] === classIndex,
    ).length;

    return {
      classIndex,
      correct,
      total,
      accuracy: total > 0 ? (correct / total) * 100 : 0,
    };
  });
}

function chartTick(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
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

function FieldLabel({
  label,
  detail,
}: {
  label: string;
  detail?: string;
}) {
  return (
    <label className="flex flex-col gap-2">
      <span className="flex items-center justify-between gap-3 text-caption">
        <span
          className="font-semibold text-[#F4F7FB]"
          style={{ fontSize: "13px", letterSpacing: "-0.18px" }}
        >
          {label}
        </span>
        {detail ? (
          <span
            className="font-mono text-[11px]"
            style={{ color: "rgba(200,200,212,0.44)" }}
          >
            {detail}
          </span>
        ) : null}
      </span>
    </label>
  );
}

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div
      className="rounded-[var(--r-lg)] border p-5 transition duration-300 hover:-translate-y-0.5"
      style={{
        background:
          "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025))",
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <div className="mb-4 h-1 w-10 rounded-full" style={{ background: accent }} />
      <p className="text-caption" style={{ color: "rgba(200,200,212,0.58)" }}>
        {label}
      </p>
      <p
        className="mt-2 font-mono text-3xl font-bold"
        style={{ color: "#FFFFFF", letterSpacing: "-0.04em" }}
      >
        {value}
      </p>
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ color?: string; name?: string; value?: number }>;
  label?: number | string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div
      className="rounded-[var(--r-md)] border px-4 py-3 text-caption backdrop-blur-xl"
      style={{
        background: "rgba(10,10,15,0.82)",
        borderColor: "rgba(255,255,255,0.12)",
        boxShadow: "0 18px 50px rgba(0,0,0,0.34)",
      }}
    >
      <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#5A5A7A]">
        Epoch {label}
      </div>
      <div className="grid gap-1.5">
        {payload.map((item) => (
          <div key={item.name} className="flex items-center justify-between gap-6">
            <span className="flex items-center gap-2 text-[#C8C8D4]">
              <span
                className="h-2 w-2 rounded-full"
                style={{ background: item.color ?? "#00D4FF" }}
              />
              {item.name}
            </span>
            <span className="font-mono text-white">{chartTick(Number(item.value ?? 0))}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingChart({
  title,
  data,
  series,
  suffix,
}: {
  title: string;
  data: EpochPoint[];
  suffix?: string;
  series: { key: keyof EpochPoint; name: string; color: string }[];
}) {
  return (
    <div
      className="min-h-[320px] rounded-[var(--r-lg)] border p-5"
      style={{
        background: "rgba(255,255,255,0.035)",
        borderColor: "rgba(255,255,255,0.07)",
      }}
    >
      <div className="mb-5 flex items-center justify-between gap-4">
        <div>
          <h2 className="font-mono text-xl font-bold text-white">{title}</h2>
          <p className="text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
            Smooth live telemetry from the training stream
          </p>
        </div>
        <div className="hidden items-center gap-4 sm:flex">
          {series.map((item) => (
            <span
              key={item.key}
              className="flex items-center gap-2 text-caption"
              style={{ color: "rgba(200,200,212,0.7)" }}
            >
              <span className="h-2 w-2 rounded-full" style={{ background: item.color }} />
              {item.name}
            </span>
          ))}
        </div>
      </div>

      <div className="h-[230px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
            <XAxis
              dataKey="epoch"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(200,200,212,0.42)", fontSize: 12 }}
              tickMargin={12}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "rgba(200,200,212,0.42)", fontSize: 12 }}
              tickFormatter={(value) => `${chartTick(Number(value))}${suffix ?? ""}`}
              width={58}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "rgba(255,255,255,0.08)" }} />
            {series.map((item) => (
              <Line
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={item.name}
                stroke={item.color}
                strokeWidth={2.4}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: item.color }}
                isAnimationActive
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const maxOffDiagonal = Math.max(
    1,
    ...matrix.flatMap((row, rowIndex) =>
      row.map((value, colIndex) => (rowIndex === colIndex ? 0 : value)),
    ),
  );

  return (
    <div className="overflow-x-auto rounded-[var(--r-lg)] border border-[#1E1E2E]">
      <table className="w-full min-w-[720px] border-separate border-spacing-0 text-center">
        <thead>
          <tr style={{ background: "rgba(255,255,255,0.04)" }}>
            <th className="px-3 py-3 text-left text-caption text-[#5A5A7A]">Actual</th>
            {Array.from({ length: 10 }, (_, index) => (
              <th key={index} className="px-3 py-3 text-caption text-[#5A5A7A]">
                {index}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th
                className="border-t border-[#1E1E2E] px-3 py-3 text-left font-mono text-sm text-white"
                scope="row"
              >
                {rowIndex}
              </th>
              {row.map((value, colIndex) => {
                const diagonal = rowIndex === colIndex;
                const intensity = diagonal
                  ? Math.min(0.24, 0.08 + value / Math.max(1, row[rowIndex]) * 0.16)
                  : Math.min(0.34, (value / maxOffDiagonal) * 0.34);
                const background = diagonal
                  ? `rgba(0, 227, 150, ${intensity})`
                  : `rgba(255, 69, 96, ${value === 0 ? 0.018 : 0.06 + intensity})`;

                return (
                  <td
                    key={`${rowIndex}-${colIndex}`}
                    className="border-t border-[#1E1E2E] px-3 py-3 font-mono text-sm transition-colors hover:bg-white/10"
                    style={{
                      background,
                      color: diagonal ? "#B9FFE8" : value > 0 ? "#FFD0D7" : "#4B4B68",
                    }}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TrainPage() {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [history, setHistory] = useState<EpochPoint[]>([]);
  const [bestModel, setBestModel] = useState<BestModel | null>(null);
  const [testResults, setTestResults] = useState<TestResults | null>(null);
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const completeRef = useRef(false);
  const trainingUrlRef = useRef<string | null>(null);
  const retryCountRef = useRef(0);
  const reconnectRef = useRef<(url: string) => void>(() => undefined);

  const isActive = status === "connecting" || status === "training";
  const progress = history.length
    ? Math.min(100, (history[history.length - 1].epoch / history[history.length - 1].epochs) * 100)
    : 0;

  const confusionMatrix = useMemo(
    () => (testResults ? buildConfusionMatrix(testResults.y_true, testResults.y_pred) : null),
    [testResults],
  );

  const classStats = useMemo(
    () => (testResults ? buildClassStats(testResults.y_true, testResults.y_pred) : []),
    [testResults],
  );

  const closeStream = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;

    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case "epoch":
        setStatus("training");
        setHistory((current) => [...current, event]);
        break;
      case "best_model":
        setBestModel({ epoch: event.epoch, val_loss: event.val_loss });
        break;
      case "test_results":
        setTestResults({
          test_acc: event.test_acc,
          test_loss: event.test_loss,
          y_true: event.y_true,
          y_pred: event.y_pred,
        });
        break;
      case "complete":
        completeRef.current = true;
        setModelPath(event.model_path);
        setStatus("complete");
        closeStream();
        break;
      case "error":
        completeRef.current = true;
        setError(event.message);
        setStatus("error");
        closeStream();
        break;
    }
  }, [closeStream]);

  const openStream = useCallback((url: string) => {
    // TODO: Move this EventSource lifecycle into hooks/useTrainingStream.ts in Section 9.
    const source = new EventSource(url);
    eventSourceRef.current = source;

    source.onopen = () => {
      setError(null);
      setStatus((current) => (current === "connecting" ? "training" : current));
    };

    source.onmessage = (message) => {
      try {
        handleStreamEvent(JSON.parse(message.data) as StreamEvent);
      } catch {
        setError("Training stream returned an unreadable event.");
        setStatus("error");
        completeRef.current = true;
        closeStream();
      }
    };

    source.onerror = () => {
      source.close();

      if (completeRef.current) return;

      if (retryCountRef.current >= MAX_RETRIES) {
        setError("Training stream connection dropped. Reconnect limit reached.");
        setStatus("error");
        eventSourceRef.current = null;
        return;
      }

      retryCountRef.current += 1;
      setRetryCount(retryCountRef.current);
      setStatus("connecting");

      retryTimerRef.current = setTimeout(() => {
        if (trainingUrlRef.current && !completeRef.current) {
          reconnectRef.current(trainingUrlRef.current);
        }
      }, 900 * retryCountRef.current);
    };
  }, [closeStream, handleStreamEvent]);

  useEffect(() => {
    reconnectRef.current = openStream;
  }, [openStream]);

  const startTraining = useCallback(() => {
    closeStream();
    completeRef.current = false;
    retryCountRef.current = 0;
    setRetryCount(0);
    setStatus("connecting");
    setError(null);
    setHistory([]);
    setBestModel(null);
    setTestResults(null);
    setModelPath(null);

    const url = new URL(`${API_BASE}/api/train/stream`);
    url.searchParams.set("epochs", String(config.epochs));
    url.searchParams.set("learning_rate", String(config.learningRate));
    url.searchParams.set("batch_size", String(config.batchSize));
    url.searchParams.set("dataset_mode", config.datasetMode);

    trainingUrlRef.current = url.toString();
    openStream(url.toString());
  }, [closeStream, config, openStream]);

  useEffect(() => closeStream, [closeStream]);

  return (
    <div className="min-h-dvh overflow-hidden">
      <style>{`
        @keyframes train-button-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,255,0.26), 0 18px 50px rgba(0,212,255,0.12); }
          50% { box-shadow: 0 0 0 8px rgba(0,212,255,0), 0 22px 70px rgba(0,212,255,0.24); }
        }
        @keyframes train-progress {
          from { transform: translateX(-30%); opacity: 0.35; }
          to { transform: translateX(130%); opacity: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          .train-motion { animation: none !important; transition: none !important; transform: none !important; }
        }
      `}</style>

      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
        <div className="grid gap-6 lg:grid-cols-[minmax(360px,440px)_1fr]">
          <GlassPanel className="p-5 sm:p-6">
            <div className="mb-7 flex items-start justify-between gap-4">
              <div>
                <div className="mb-3 inline-flex items-center gap-2 rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] px-3 py-1.5 text-caption text-[#00D4FF]">
                  <Sparkles className="size-3.5" />
                  Live training
                </div>
                <h1 className="text-display text-white">Training dashboard</h1>
                <p className="mt-3 max-w-[34rem] text-body" style={{ color: "rgba(200,200,212,0.68)" }}>
                  Configure the CNN run and watch loss, accuracy, and class behavior update from the SSE stream.
                </p>
              </div>
              <div
                className="hidden rounded-full border p-3 sm:block"
                style={{
                  borderColor: "rgba(0,212,255,0.18)",
                  background: "rgba(0,212,255,0.07)",
                }}
              >
                <Activity className="size-5 text-[#00D4FF]" />
              </div>
            </div>

            <div className="grid gap-5">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                <div>
                  <FieldLabel label="Epochs" detail="1-100" />
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={config.epochs}
                    disabled={isActive}
                    onChange={(event) =>
                      setConfig((current) => ({
                        ...current,
                        epochs: Number(event.target.value),
                      }))
                    }
                    className="mt-2 h-12 w-full rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.045] px-5 text-[17px] text-white outline-none transition focus:border-[#00D4FF] focus:ring-4 focus:ring-[#00D4FF]/10 disabled:opacity-50"
                  />
                </div>

                <div>
                  <FieldLabel label="Learning rate" detail="step 0.0001" />
                  <input
                    type="number"
                    min={0.0001}
                    step={0.0001}
                    value={config.learningRate}
                    disabled={isActive}
                    onChange={(event) =>
                      setConfig((current) => ({
                        ...current,
                        learningRate: Number(event.target.value),
                      }))
                    }
                    className="mt-2 h-12 w-full rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.045] px-5 text-[17px] text-white outline-none transition focus:border-[#00D4FF] focus:ring-4 focus:ring-[#00D4FF]/10 disabled:opacity-50"
                  />
                </div>
              </div>

              <div>
                <FieldLabel label="Batch size" />
                <select
                  value={config.batchSize}
                  disabled={isActive}
                  onChange={(event) =>
                    setConfig((current) => ({
                      ...current,
                      batchSize: Number(event.target.value),
                    }))
                  }
                  className="mt-2 h-12 w-full rounded-[var(--r-pill)] border border-[#1E1E2E] bg-[#181823] px-5 text-[17px] text-white outline-none transition focus:border-[#00D4FF] focus:ring-4 focus:ring-[#00D4FF]/10 disabled:opacity-50"
                >
                  {BATCH_SIZES.map((size) => (
                    <option key={size} value={size}>
                      {size}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <FieldLabel label="Dataset" />
                <div
                  className="mt-2 grid rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] p-1 sm:grid-cols-3"
                  role="radiogroup"
                  aria-label="Dataset"
                >
                  {DATASETS.map((dataset) => {
                    const selected = config.datasetMode === dataset.value;

                    return (
                      <button
                        key={dataset.value}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        disabled={isActive}
                        onClick={() =>
                          setConfig((current) => ({
                            ...current,
                            datasetMode: dataset.value,
                          }))
                        }
                        className="btn-press h-10 rounded-[var(--r-pill)] px-4 text-caption transition disabled:opacity-50"
                        style={{
                          background: selected ? "#00D4FF" : "transparent",
                          color: selected ? "#0A0A0F" : "rgba(200,200,212,0.72)",
                          fontWeight: selected ? 700 : 400,
                        }}
                      >
                        {dataset.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <button
                type="button"
                onClick={startTraining}
                disabled={isActive}
                className="train-motion btn-press mt-1 flex h-14 w-full items-center justify-center gap-3 rounded-[var(--r-pill)] bg-[#00D4FF] px-6 text-[17px] font-bold text-[#0A0A0F] transition hover:-translate-y-0.5 hover:bg-[#1ADCFF] disabled:cursor-not-allowed disabled:opacity-90"
                style={{
                  animation: isActive ? "train-button-pulse 1.8s ease-in-out infinite" : undefined,
                }}
              >
                {isActive ? (
                  <>
                    <span className="relative flex h-3 w-3">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#0A0A0F] opacity-45" />
                      <span className="relative inline-flex h-3 w-3 rounded-full bg-[#0A0A0F]" />
                    </span>
                    Training active
                  </>
                ) : (
                  <>
                    <Play className="size-4 fill-[#0A0A0F]" />
                    Start Training
                  </>
                )}
              </button>

              <div className="flex flex-wrap items-center justify-between gap-3 rounded-[var(--r-md)] border border-[#1E1E2E] bg-black/20 px-4 py-3">
                <span className="text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
                  Status
                </span>
                <span className="inline-flex items-center gap-2 text-caption text-white">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{
                      background:
                        status === "error"
                          ? "#FF4560"
                          : status === "complete"
                            ? "#00E396"
                            : isActive
                              ? "#00D4FF"
                              : "#5A5A7A",
                    }}
                  />
                  {status === "idle"
                    ? "Ready"
                    : status === "connecting"
                      ? `Connecting${retryCount ? `, retry ${retryCount}/${MAX_RETRIES}` : ""}`
                      : status === "training"
                        ? "Streaming epochs"
                        : status === "complete"
                          ? "Complete"
                          : "Needs attention"}
                </span>
              </div>

              {error ? (
                <div
                  className="flex items-start gap-3 rounded-[var(--r-md)] border px-4 py-3 text-caption"
                  style={{
                    borderColor: "rgba(255,69,96,0.24)",
                    background: "rgba(255,69,96,0.07)",
                    color: "#FFD0D7",
                  }}
                  role="alert"
                >
                  <WifiOff className="mt-0.5 size-4 shrink-0 text-[#FF4560]" />
                  {error}
                </div>
              ) : null}
            </div>
          </GlassPanel>

          <GlassPanel className="flex min-h-[420px] flex-col justify-between p-5 sm:p-6">
            <div>
              <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#5A5A7A]">
                    Training progress
                  </p>
                  <h2 className="mt-1 font-mono text-2xl font-bold text-white">
                    {history.length ? `Epoch ${history[history.length - 1].epoch}` : "Awaiting stream"}
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    closeStream();
                    setStatus("idle");
                    setError(null);
                    setRetryCount(0);
                  }}
                  className="btn-press inline-flex h-10 items-center gap-2 rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] px-4 text-caption text-[#C8C8D4] transition hover:border-[#2A2A3E] hover:text-white"
                >
                  <RotateCcw className="size-3.5" />
                  Reset stream
                </button>
              </div>

              <div className="mb-7">
                <div className="mb-2 flex justify-between text-caption">
                  <span style={{ color: "rgba(200,200,212,0.56)" }}>Completion</span>
                  <span className="font-mono text-white">{progress.toFixed(0)}%</span>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-white/[0.055]">
                  <div
                    className="train-motion h-full rounded-full bg-[#00D4FF] transition-all duration-700 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                  {isActive ? (
                    <span
                      className="train-motion absolute inset-y-0 left-0 w-1/3 rounded-full bg-white/35"
                      style={{ animation: "train-progress 1.6s ease-in-out infinite" }}
                    />
                  ) : null}
                </div>
              </div>

              {history.length ? (
                <div className="train-motion grid gap-5 animate-fade-up xl:grid-cols-2">
                  <TrainingChart
                    title="Loss curve"
                    data={history}
                    series={[
                      { key: "train_loss", name: "Train Loss", color: "#00D4FF" },
                      { key: "val_loss", name: "Val Loss", color: "#FFB800" },
                    ]}
                  />
                  <TrainingChart
                    title="Accuracy curve"
                    data={history}
                    suffix="%"
                    series={[
                      { key: "train_acc", name: "Train Acc", color: "#00D4FF" },
                      { key: "val_acc", name: "Val Acc", color: "#FFB800" },
                    ]}
                  />
                </div>
              ) : (
                <div
                  className="flex min-h-[280px] flex-col items-center justify-center rounded-[var(--r-lg)] border border-dashed border-[#1E1E2E] bg-white/[0.02] p-8 text-center"
                >
                  <Activity className="mb-4 size-8 text-[#00D4FF]" />
                  <h3 className="font-mono text-xl font-bold text-white">Charts appear when training starts</h3>
                  <p className="mt-2 max-w-md text-caption" style={{ color: "rgba(200,200,212,0.58)" }}>
                    The stream will plot monotone loss and accuracy curves with each epoch event.
                  </p>
                </div>
              )}
            </div>
          </GlassPanel>
        </div>

        {status === "complete" && testResults && confusionMatrix ? (
          <GlassPanel className="train-motion animate-fade-up p-5 sm:p-6">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="mb-3 inline-flex items-center gap-2 rounded-[var(--r-pill)] border border-[#00E396]/20 bg-[#00E396]/10 px-3 py-1.5 text-caption text-[#00E396]">
                  <CheckCircle2 className="size-3.5" />
                  Training complete
                </div>
                <h2 className="text-display text-white">Final results</h2>
                {modelPath ? (
                  <p className="mt-2 text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
                    Saved model: <span className="font-mono text-[#C8C8D4]">{modelPath}</span>
                  </p>
                ) : null}
              </div>

              <a
                href={REPORT_URL}
                className="btn-press inline-flex h-11 items-center gap-2 rounded-[var(--r-pill)] border border-[#00D4FF]/35 bg-[#00D4FF]/10 px-5 text-caption font-semibold text-[#00D4FF] transition hover:border-[#00D4FF] hover:bg-[#00D4FF]/15"
              >
                <ArrowDownToLine className="size-4" />
                Download report
              </a>
            </div>

            <div className="mb-6 grid gap-4 md:grid-cols-3">
              <MetricCard label="Final Test Accuracy" value={formatPercent(testResults.test_acc)} accent="#00E396" />
              <MetricCard label="Test Loss" value={formatMetric(testResults.test_loss)} accent="#00D4FF" />
              <MetricCard
                label="Best Val Loss"
                value={bestModel ? formatMetric(bestModel.val_loss) : "--"}
                accent="#FFB800"
              />
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
              <div>
                <div className="mb-3 flex items-end justify-between gap-4">
                  <div>
                    <h3 className="font-mono text-xl font-bold text-white">Confusion matrix</h3>
                    <p className="text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
                      Diagonal cells indicate correct classifications.
                    </p>
                  </div>
                </div>
                <ConfusionMatrix matrix={confusionMatrix} />
              </div>

              <div>
                <h3 className="mb-1 font-mono text-xl font-bold text-white">Per-class accuracy</h3>
                <p className="mb-3 text-caption" style={{ color: "rgba(200,200,212,0.56)" }}>
                  Correct predictions by digit class.
                </p>
                <div className="overflow-hidden rounded-[var(--r-lg)] border border-[#1E1E2E]">
                  <table className="w-full border-separate border-spacing-0">
                    <thead>
                      <tr style={{ background: "rgba(255,255,255,0.04)" }}>
                        {["Class", "Correct", "Total", "Accuracy"].map((heading) => (
                          <th
                            key={heading}
                            className="px-4 py-3 text-left text-caption font-semibold text-[#5A5A7A]"
                          >
                            {heading}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {classStats.map((row) => (
                        <tr key={row.classIndex} className="transition-colors hover:bg-white/[0.035]">
                          <td className="border-t border-[#1E1E2E] px-4 py-3 font-mono text-white">
                            {row.classIndex}
                          </td>
                          <td className="border-t border-[#1E1E2E] px-4 py-3 text-[#C8C8D4]">
                            {row.correct}
                          </td>
                          <td className="border-t border-[#1E1E2E] px-4 py-3 text-[#C8C8D4]">
                            {row.total}
                          </td>
                          <td className="border-t border-[#1E1E2E] px-4 py-3 font-mono text-[#00E396]">
                            {formatPercent(row.accuracy)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </GlassPanel>
        ) : null}
      </div>
    </div>
  );
}
