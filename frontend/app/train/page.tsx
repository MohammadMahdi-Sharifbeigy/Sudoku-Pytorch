"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  CheckCircle2,
  Play,
  RotateCcw,
  Sparkles,
  Target,
  TrendingDown,
  Trophy,
  WifiOff,
} from "lucide-react";
import { LossAccuracyChart } from "@/components/charts/LossAccuracyChart";
import { StatCard } from "@/components/ui/StatCard";
import {
  type DatasetMode,
  type TrainingConfig,
  useTrainingStream,
} from "@/hooks/useTrainingStream";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REPORT_URL = `${API_BASE}/api/reports/training_report.txt`;
const MAX_RETRIES = 3;

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

function formatMetric(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? "--" : value.toFixed(digits);
}

function formatPercent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "--" : `${value.toFixed(2)}%`;
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
      style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.055)" }}
    >
      {children}
    </section>
  );
}

function FieldLabel({ label, detail }: { label: string; detail?: string }) {
  return (
    <label className="flex flex-col gap-2">
      <span className="flex items-center justify-between gap-3 text-caption">
        <span className="font-semibold text-[#F4F7FB]" style={{ fontSize: "13px" }}>
          {label}
        </span>
        {detail ? <span className="font-mono text-[11px] text-[#5A5A7A]">{detail}</span> : null}
      </span>
    </label>
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
          <tr className="bg-white/[0.04]">
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
                  ? Math.min(0.24, 0.08 + (value / Math.max(1, row[rowIndex])) * 0.16)
                  : Math.min(0.34, (value / maxOffDiagonal) * 0.34);

                return (
                  <td
                    key={`${rowIndex}-${colIndex}`}
                    className="border-t border-[#1E1E2E] px-3 py-3 font-mono text-sm transition-colors hover:bg-white/10"
                    style={{
                      background: diagonal
                        ? `rgba(0, 227, 150, ${intensity})`
                        : `rgba(255, 69, 96, ${value === 0 ? 0.018 : 0.06 + intensity})`,
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
  const {
    isTraining,
    epochData,
    metrics,
    error,
    startTraining,
    resetTraining,
    testResults,
    status,
  } = useTrainingStream();

  const progress = epochData.length
    ? Math.min(100, (epochData[epochData.length - 1].epoch / epochData[epochData.length - 1].epochs) * 100)
    : 0;

  const confusionMatrix = useMemo(
    () => (testResults ? buildConfusionMatrix(testResults.y_true, testResults.y_pred) : null),
    [testResults],
  );

  const classStats = useMemo(
    () => (testResults ? buildClassStats(testResults.y_true, testResults.y_pred) : []),
    [testResults],
  );

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
                <p className="mt-3 max-w-[34rem] text-body text-[#C8C8D4]/70">
                  Configure the CNN run and watch loss, accuracy, and class behavior update from the SSE stream.
                </p>
              </div>
              <div className="hidden rounded-full border border-[#00D4FF]/20 bg-[#00D4FF]/10 p-3 sm:block">
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
                    disabled={isTraining}
                    onChange={(event) => setConfig((current) => ({ ...current, epochs: Number(event.target.value) }))}
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
                    disabled={isTraining}
                    onChange={(event) => setConfig((current) => ({ ...current, learningRate: Number(event.target.value) }))}
                    className="mt-2 h-12 w-full rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.045] px-5 text-[17px] text-white outline-none transition focus:border-[#00D4FF] focus:ring-4 focus:ring-[#00D4FF]/10 disabled:opacity-50"
                  />
                </div>
              </div>

              <div>
                <FieldLabel label="Batch size" />
                <select
                  value={config.batchSize}
                  disabled={isTraining}
                  onChange={(event) => setConfig((current) => ({ ...current, batchSize: Number(event.target.value) }))}
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
                <div className="mt-2 grid rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] p-1 sm:grid-cols-3" role="radiogroup" aria-label="Dataset">
                  {DATASETS.map((dataset) => {
                    const selected = config.datasetMode === dataset.value;
                    return (
                      <button
                        key={dataset.value}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        disabled={isTraining}
                        onClick={() => setConfig((current) => ({ ...current, datasetMode: dataset.value }))}
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
                onClick={() => startTraining(config)}
                disabled={isTraining}
                className="train-motion btn-press mt-1 flex h-14 w-full items-center justify-center gap-3 rounded-[var(--r-pill)] bg-[#00D4FF] px-6 text-[17px] font-bold text-[#0A0A0F] transition hover:-translate-y-0.5 hover:bg-[#1ADCFF] disabled:cursor-not-allowed disabled:opacity-90"
                style={{ animation: isTraining ? "train-button-pulse 1.8s ease-in-out infinite" : undefined }}
              >
                {isTraining ? (
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
                <span className="text-caption text-[#C8C8D4]/60">Status</span>
                <span className="inline-flex items-center gap-2 text-caption text-white">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{
                      background:
                        status === "error"
                          ? "#FF4560"
                          : status === "complete"
                            ? "#00E396"
                            : isTraining
                              ? "#00D4FF"
                              : "#5A5A7A",
                    }}
                  />
                  {status === "idle"
                    ? "Ready"
                    : status === "connecting"
                      ? `Connecting${metrics.retryCount ? `, retry ${metrics.retryCount}/${MAX_RETRIES}` : ""}`
                      : status === "training"
                        ? "Streaming epochs"
                        : status === "complete"
                          ? "Complete"
                          : "Needs attention"}
                </span>
              </div>

              {error ? (
                <div className="flex items-start gap-3 rounded-[var(--r-md)] border border-[#FF4560]/25 bg-[#FF4560]/10 px-4 py-3 text-caption text-[#FFD0D7]" role="alert">
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
                    {epochData.length ? `Epoch ${epochData[epochData.length - 1].epoch}` : "Awaiting stream"}
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={resetTraining}
                  className="btn-press inline-flex h-10 items-center gap-2 rounded-[var(--r-pill)] border border-[#1E1E2E] bg-white/[0.035] px-4 text-caption text-[#C8C8D4] transition hover:border-[#2A2A3E] hover:text-white"
                >
                  <RotateCcw className="size-3.5" />
                  Reset stream
                </button>
              </div>

              <div className="mb-7">
                <div className="mb-2 flex justify-between text-caption">
                  <span className="text-[#C8C8D4]/60">Completion</span>
                  <span className="font-mono text-white">{progress.toFixed(0)}%</span>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-white/[0.055]">
                  <div className="train-motion h-full rounded-full bg-[#00D4FF] transition-all duration-700 ease-out" style={{ width: `${progress}%` }} />
                  {isTraining ? <span className="train-motion absolute inset-y-0 left-0 w-1/3 rounded-full bg-white/35" style={{ animation: "train-progress 1.6s ease-in-out infinite" }} /> : null}
                </div>
              </div>

              {epochData.length ? (
                <div className="train-motion animate-fade-up">
                  <LossAccuracyChart
                    trainLoss={epochData.map((point) => point.train_loss)}
                    valLoss={epochData.map((point) => point.val_loss)}
                    trainAcc={epochData.map((point) => point.train_acc)}
                    valAcc={epochData.map((point) => point.val_acc)}
                  />
                </div>
              ) : (
                <div className="flex min-h-[280px] flex-col items-center justify-center rounded-[var(--r-lg)] border border-dashed border-[#1E1E2E] bg-white/[0.02] p-8 text-center">
                  <Activity className="mb-4 size-8 text-[#00D4FF]" />
                  <h3 className="font-mono text-xl font-bold text-white">Charts appear when training starts</h3>
                  <p className="mt-2 max-w-md text-caption text-[#C8C8D4]/60">
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
                {metrics.modelPath ? (
                  <p className="mt-2 text-caption text-[#C8C8D4]/60">
                    Saved model: <span className="font-mono text-[#C8C8D4]">{metrics.modelPath}</span>
                  </p>
                ) : null}
              </div>

              <a href={REPORT_URL} className="btn-press inline-flex h-11 items-center gap-2 rounded-[var(--r-pill)] border border-[#00D4FF]/35 bg-[#00D4FF]/10 px-5 text-caption font-semibold text-[#00D4FF] transition hover:border-[#00D4FF] hover:bg-[#00D4FF]/15">
                <ArrowDownToLine className="size-4" />
                Download report
              </a>
            </div>

            <div className="mb-6 grid gap-4 md:grid-cols-3">
              <StatCard label="Final Test Accuracy" value={formatPercent(metrics.testAcc)} icon={<Target className="size-4" />} />
              <StatCard label="Test Loss" value={formatMetric(metrics.testLoss)} icon={<TrendingDown className="size-4" />} />
              <StatCard label="Best Val Loss" value={formatMetric(metrics.bestValLoss)} icon={<Trophy className="size-4" />} />
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
              <div>
                <div className="mb-3">
                  <h3 className="font-mono text-xl font-bold text-white">Confusion matrix</h3>
                  <p className="text-caption text-[#C8C8D4]/60">Diagonal cells indicate correct classifications.</p>
                </div>
                <ConfusionMatrix matrix={confusionMatrix} />
              </div>

              <div>
                <h3 className="mb-1 font-mono text-xl font-bold text-white">Per-class accuracy</h3>
                <p className="mb-3 text-caption text-[#C8C8D4]/60">Correct predictions by digit class.</p>
                <div className="overflow-hidden rounded-[var(--r-lg)] border border-[#1E1E2E]">
                  <table className="w-full border-separate border-spacing-0">
                    <thead>
                      <tr className="bg-white/[0.04]">
                        {["Class", "Correct", "Total", "Accuracy"].map((heading) => (
                          <th key={heading} className="px-4 py-3 text-left text-caption font-semibold text-[#5A5A7A]">
                            {heading}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {classStats.map((row) => (
                        <tr key={row.classIndex} className="transition-colors hover:bg-white/[0.035]">
                          <td className="border-t border-[#1E1E2E] px-4 py-3 font-mono text-white">{row.classIndex}</td>
                          <td className="border-t border-[#1E1E2E] px-4 py-3 text-[#C8C8D4]">{row.correct}</td>
                          <td className="border-t border-[#1E1E2E] px-4 py-3 text-[#C8C8D4]">{row.total}</td>
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
