const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface CellData {
  row: number;
  col: number;
  value: number;          // 0 = empty, 1-9 = digit
  confidence: number;     // 0.0 – 1.0
  is_given: boolean;
  image_b64: string;      // base64 PNG, 28×28
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
  solve_time_ms?: number;
}

export async function solveImage(
  file: File,
  signal?: AbortSignal,
): Promise<SolveResponse> {
  const body = new FormData();
  body.append("file", file);

  const res = await fetch(`${API_BASE}/api/solve`, {
    method: "POST",
    body,
    signal,
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.error ?? data?.detail ?? `HTTP ${res.status}`);
  }

  return data as SolveResponse;
}
