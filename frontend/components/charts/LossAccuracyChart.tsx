"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface LossAccuracyChartProps {
  trainLoss: number[];
  valLoss: number[];
  trainAcc: number[];
  valAcc: number[];
}

interface ChartPoint {
  epoch: number;
  trainLoss: number | null;
  valLoss: number | null;
  trainAcc: number | null;
  valAcc: number | null;
}

function chartTick(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
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
        background: "rgba(10,10,15,0.84)",
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
              <span className="h-2 w-2 rounded-full" style={{ background: item.color ?? "#00D4FF" }} />
              {item.name}
            </span>
            <span className="font-mono text-white">{chartTick(Number(item.value ?? 0))}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricChart({
  title,
  data,
  lines,
  suffix,
}: {
  title: string;
  data: ChartPoint[];
  suffix?: string;
  lines: { key: keyof ChartPoint; name: string; color: string }[];
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
          {lines.map((line) => (
            <span key={line.key} className="flex items-center gap-2 text-caption text-[#C8C8D4]">
              <span className="h-2 w-2 rounded-full" style={{ background: line.color }} />
              {line.name}
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
            {lines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                name={line.name}
                stroke={line.color}
                strokeWidth={2.4}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: line.color }}
                isAnimationActive
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function LossAccuracyChart({
  trainLoss,
  valLoss,
  trainAcc,
  valAcc,
}: LossAccuracyChartProps) {
  const length = Math.max(trainLoss.length, valLoss.length, trainAcc.length, valAcc.length);
  const data = Array.from({ length }, (_, index) => ({
    epoch: index + 1,
    trainLoss: trainLoss[index] ?? null,
    valLoss: valLoss[index] ?? null,
    trainAcc: trainAcc[index] ?? null,
    valAcc: valAcc[index] ?? null,
  }));

  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <MetricChart
        title="Loss curve"
        data={data}
        lines={[
          { key: "trainLoss", name: "Train Loss", color: "#00D4FF" },
          { key: "valLoss", name: "Val Loss", color: "#FFB800" },
        ]}
      />
      <MetricChart
        title="Accuracy curve"
        data={data}
        suffix="%"
        lines={[
          { key: "trainAcc", name: "Train Acc", color: "#00D4FF" },
          { key: "valAcc", name: "Val Acc", color: "#FFB800" },
        ]}
      />
    </div>
  );
}
