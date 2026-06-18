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
  };
}

export async function solveImage(
  file: File,
  signal?: AbortSignal,
): Promise<SolveResponse> {
  const body = new FormData();
  body.append("image", file);

  const res = await fetch(`${API_BASE}/api/solve`, {
    method: "POST",
    body,
    signal,
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.error ?? data?.detail ?? `HTTP ${res.status}`);
  }

  return normalizeSolveResponse(data as BackendSolveResponse);
}
