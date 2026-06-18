const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type DatasetMode = "mnist_fonts" | "mnist_hoda" | "all";

export interface CellData {
  row: number;
  col: number;
  value: number;
  confidence: number;
  is_given: boolean;
  image_b64: string;
}

export interface SolveResponse {
  success: boolean;
  error?: string;
  detail?: string;
  original_grid: number[][];
  solved_grid: number[][];
  cells: CellData[];
  original_image_b64: string;
  solved_image_b64: string;
  threshold_image_b64: string;
  cell_grid_image_b64: string;
  solve_time_ms?: number;
}

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  datasetMode: DatasetMode;
}

export interface ModelInfo {
  exists: boolean;
  size_mb: number | null;
  created_at: string | null;
  total_params: number | null;
  trainable_params: number | null;
}

export interface BenchmarkEntry {
  format: "pt" | "ts" | "onnx" | string;
  size_mb: number;
  latency_ms: number | null;
  path: string;
}

export interface BenchmarkResult {
  success: boolean;
  results: BenchmarkEntry[];
  error?: string | null;
}

interface BackendCellPrediction {
  row: number;
  col: number;
  has_digit: boolean;
  label: number;
  confidence: number;
  cell_image_b64: string;
}

interface BackendSolveResponse {
  success: boolean;
  error?: string;
  detail?: string;
  original_grid: number[][];
  solved_board: number[][] | null;
  per_cell: BackendCellPrediction[];
  solution_image_b64: string | null;
  board_image_b64: string;
  threshold_image_b64?: string;
  cell_grid_image_b64?: string;
}

interface ApiErrorPayload {
  error?: string;
  detail?: string;
  message?: string;
}

function normalizeSolveResponse(data: BackendSolveResponse): SolveResponse {
  return {
    success: data.success,
    error: data.error,
    detail: data.detail,
    original_grid: data.original_grid,
    solved_grid: data.solved_board ?? data.original_grid,
    cells: data.per_cell.map((cell) => ({
      row: cell.row,
      col: cell.col,
      value: cell.has_digit ? cell.label : 0,
      confidence: cell.confidence,
      is_given: cell.has_digit,
      image_b64: cell.cell_image_b64,
    })),
    original_image_b64: data.board_image_b64,
    solved_image_b64: data.solution_image_b64 ?? data.board_image_b64,
    threshold_image_b64: data.threshold_image_b64 ?? data.board_image_b64,
    cell_grid_image_b64: data.cell_grid_image_b64 ?? data.board_image_b64,
  };
}

export async function parseApiError(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as ApiErrorPayload;
    return data.error ?? data.detail ?? data.message ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);

  if (!res.ok) {
    throw new Error(await parseApiError(res));
  }

  return (await res.json()) as T;
}

export function trainingStreamUrl(config: TrainingConfig): string {
  const url = new URL(`${API_BASE}/api/train/stream`);
  url.searchParams.set("epochs", String(config.epochs));
  url.searchParams.set("learning_rate", String(config.learningRate));
  url.searchParams.set("batch_size", String(config.batchSize));
  url.searchParams.set("dataset_mode", config.datasetMode);
  return url.toString();
}

export async function solveSudoku(file: File, signal?: AbortSignal): Promise<SolveResponse> {
  const body = new FormData();
  body.append("image", file);

  const data = await requestJson<BackendSolveResponse>(`${API_BASE}/api/solve`, {
    method: "POST",
    body,
    signal,
  });

  return normalizeSolveResponse(data);
}

export async function solveImage(file: File, signal?: AbortSignal): Promise<SolveResponse> {
  return solveSudoku(file, signal);
}

export function startTraining(config: TrainingConfig): string {
  return trainingStreamUrl(config);
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  return requestJson<ModelInfo>(`${API_BASE}/api/model/info`, { signal });
}

export async function runOptimization(signal?: AbortSignal): Promise<BenchmarkResult> {
  return requestJson<BenchmarkResult>(`${API_BASE}/api/optimize`, {
    method: "POST",
    signal,
  });
}

export function modelDownloadUrl(format: "pt" | "ts" | "onnx"): string {
  return `${API_BASE}/api/models/download?format=${format}`;
}
