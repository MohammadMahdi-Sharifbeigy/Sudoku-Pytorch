# Sudoku Solver — Full Rebuild Roadmap
## FastAPI Backend + Next.js 14 Frontend
### Annotated with Claude Code Skill Usage

---

## How to Use This Document

Each section is a standalone Claude Code session.
Every section lists which skills to read **before** writing any code and which to apply **after** for review and quality assurance.

To load a skill at the start of a session:
```
Read [skill path from CLAUDE.md] before making any changes.
```

The CLAUDE.md file at the project root contains all skill paths, the full API contract, design tokens, and quality gates. Read it at the start of every session.

---

## Architecture

```
sudoku-solver/
├── backend/          # FastAPI + PyTorch
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── inference.py    # /api/solve
│   │   │   ├── training.py     # /api/train (SSE streaming)
│   │   │   └── optimization.py # /api/optimize
│   │   ├── core/
│   │   │   ├── model.py        # DigitCNN, FocalLoss
│   │   │   ├── solver.py       # SudokuSolver
│   │   │   ├── vision.py       # CV pipeline
│   │   │   ├── train.py        # train_epoch, validate
│   │   │   └── data_utils.py   # DataLoaders
│   │   └── schemas.py          # Pydantic models
├── frontend/         # Next.js 14 + Tailwind + shadcn/ui
│   ├── app/
│   │   ├── page.tsx
│   │   ├── solve/page.tsx
│   │   ├── train/page.tsx
│   │   └── optimize/page.tsx
│   └── components/
│       ├── SudokuGrid.tsx
│       ├── CellGrid.tsx
│       ├── TrainingChart.tsx
│       └── ConfusionMatrix.tsx
└── models/           # .pt, .ts, .onnx weights
```

---

## Build Order

| # | Section | Est. Time |
|---|---|---|
| 1 | Scaffolding + Backend Foundation | 30 min |
| 2 | Inference API (/api/solve) | 20 min |
| 3 | Training API (SSE streaming) | 25 min |
| 4 | Optimization API | 15 min |
| 5 | Frontend Design System + Layout | 20 min |
| 6 | Solve Page | 30 min |
| 7 | Training Dashboard | 25 min |
| 8 | Optimization Page | 20 min |
| 9 | Reusable Components | 20 min |
| 10 | Polish + Error Handling + Tests | 25 min |
| 11 | Docker + Deployment | 15 min |

**Total estimated time: ~4.5 hours across 11 sessions**

---

## Section 1 — Project Scaffolding & Backend Foundation

### Skills — Read Before Starting
- `claude-code-setup` — project initialization conventions for Claude Code
- `pyright-lsp` — Python type checking configuration (set up pyright config now, not later)
- `feature-dev` — feature branch and scaffolding discipline

### Prompt

```
Read CLAUDE.md, then read skills/public/claude-code-setup/SKILL.md,
skills/public/pyright-lsp/SKILL.md, and skills/public/feature-dev/SKILL.md
before making any changes.

I'm rebuilding my Streamlit Sudoku Solver (currently at
https://github.com/MohammadMahdi-Sharifbeigy/Sudoku-Pytorch/tree/fullapp)
into a proper FastAPI backend + Next.js 14 frontend.

Read all files in the current directory to understand the existing codebase,
especially model.py, vision.py, solver.py, train.py, data_utils.py.

Then scaffold the full project structure:

1. Create `backend/` with:
   - FastAPI app (backend/app/main.py) with CORS for localhost:3000
   - Copy and adapt model.py, solver.py, vision.py, train.py,
     data_utils.py, report_utils.py, optimize_model.py into backend/app/core/
   - Create backend/app/schemas.py with Pydantic models for all request/
     response types (SolveRequest, SolveResponse, TrainingConfig,
     BenchmarkResult, CellPrediction, etc.)
   - Create backend/app/routers/inference.py with POST /api/solve that:
       * Accepts a multipart image upload
       * Runs the full vision pipeline (get_valid_cells_from_image)
       * Runs per-cell CNN prediction (get_per_cell_predictions)
       * Runs the backtracking solver
       * Returns structured JSON with: solved_board (9x9), original grid,
         per_cell list, confidence stats, and base64-encoded solution image
   - Create backend/requirements.txt
   - Create pyrightconfig.json at backend/ root

2. Create `frontend/` with Next.js 14 (app router, TypeScript, Tailwind CSS)
   - npx create-next-app@latest frontend --typescript --tailwind --app
   - Install: shadcn/ui, lucide-react, recharts, react-dropzone, axios

3. Create root docker-compose.yml with backend (port 8000) and frontend
   (port 3000) services.

Use async FastAPI throughout. The vision pipeline is CPU-heavy so run it
in a thread pool with asyncio.run_in_executor.
```

### Skills — Apply After Completion
- `pyright-lsp` — run `pyright backend/app/` and resolve all type errors
- `typescript-lsp` — run `npx tsc --noEmit` in frontend/ and resolve errors
- `caveman-commit` — commit with message: `feat(scaffold): initial project structure and backend foundation`

---

## Section 2 — Inference API (Solve Endpoint)

### Skills — Read Before Starting
- `pyright-lsp` — enforce strict typing on all new functions
- `security-guidance` — file upload endpoints are an attack surface; apply input validation and size limits

### Prompt

```
Read CLAUDE.md, then read skills/public/pyright-lsp/SKILL.md and
skills/public/security-guidance/SKILL.md before making any changes.

Read backend/app/routers/inference.py and backend/app/core/vision.py.

Implement the complete POST /api/solve endpoint:

1. Accept multipart/form-data with field "image" (jpg/png/jpeg, max 10MB).
2. Validate file type and size — return HTTP 400 for invalid input.
3. Decode the image with cv2.imdecode from bytes.
4. Run resize_and_maintain_aspect_ratio(img, new_width=1000).
5. Run get_valid_cells_from_image(img) to get (cells, M, board_image).
   DO NOT simplify this function. Preserve all three strategies:
   contour path → KMeans centroid recovery → deterministic slice fallback.
6. Run get_per_cell_predictions(model, cells, device) for label+confidence.
7. Build the 9x9 grid array using get_predicted_sudoku_grid_torch.
8. Run SudokuSolver(copy.deepcopy(grid_array)).solve().
9. If solved: run generate_solution_image and encode result as base64 PNG.
10. Return JSON per the schema in CLAUDE.md.

Load model once at startup using FastAPI lifespan, store in app.state.model.
Run the entire pipeline in asyncio.run_in_executor to avoid blocking.
Handle errors with HTTP 400/422/500 and structured JSON error bodies.
```

### Skills — Apply After Completion
- `pyright-lsp` — type-check the router module
- `code-review` (caveman-review) — review for correctness of async/executor pattern
- `security-guidance` — confirm upload validation is complete
- `caveman-commit` — `feat(inference): implement POST /api/solve with full vision pipeline`

---

## Section 3 — Training API with Server-Sent Events

### Skills — Read Before Starting
- `pyright-lsp` — SSE and threading code is easy to type incorrectly
- `ralph-loop` — handles long-running loop patterns; apply its discipline to the training loop callback

### Prompt

```
Read CLAUDE.md, then read skills/public/pyright-lsp/SKILL.md and
skills/public/ralph-loop/SKILL.md before making any changes.

Read backend/app/core/train.py, data_utils.py, model.py.

Implement GET /api/train/stream as a Server-Sent Events endpoint using
FastAPI's StreamingResponse + asyncio.Queue.

The endpoint should:
1. Accept query params: epochs, learning_rate, batch_size, dataset_mode
   (one of: "mnist_fonts" | "mnist_hoda" | "all")
2. Use a global asyncio.Lock to reject concurrent training with HTTP 409.
3. Start training in a background thread (run_in_executor).
4. Stream JSON events through SSE per the schema in CLAUDE.md.
5. Save the best model to models/best_model.pt.

Also implement GET /api/train/status returning { "is_training": bool }.

Use a global asyncio.Event + Lock to prevent concurrent training runs.
The training callback must put events into an asyncio.Queue that the SSE
endpoint drains. Do not use asyncio.run() inside the training thread.
```

### Skills — Apply After Completion
- `pyright-lsp` — type-check the training router
- `ralph-loop` — confirm the loop callback/queue pattern is correct
- `session-report` — output handoff note for Section 4
- `caveman-commit` — `feat(training): SSE streaming endpoint with asyncio queue`

---

## Section 4 — Optimization & Benchmark API

### Skills — Read Before Starting
- `pyright-lsp` — type-check optimization module
- `session-report` — read previous session handoff before starting

### Prompt

```
Read CLAUDE.md, then read skills/public/pyright-lsp/SKILL.md before
making any changes.

Read backend/app/core/optimize_model.py.

Implement POST /api/optimize in backend/app/routers/optimization.py:

1. Load models/best_model.pt — return HTTP 404 with clear message if absent.
2. Run run_optimization_and_benchmark from optimize_model.py in executor.
3. Return JSON per the benchmark schema in CLAUDE.md.

Also add GET /api/model/info returning:
- Whether best_model.pt exists
- File size in MB, creation date (ISO 8601)
- Total and trainable parameter counts

Also add backend/app/routers/models.py with:
- GET /api/models/download?format=pt|ts|onnx → FileResponse
- GET /api/reports/{filename} → FileResponse for text reports
```

### Skills — Apply After Completion
- `pyright-lsp` — type-check both new routers
- `security-guidance` — verify download endpoints cannot traverse outside models/
- `caveman-commit` — `feat(optimization): benchmark API and model download endpoints`

---

## Section 5 — Frontend Design System & Layout

### Skills — Read Before Starting
- `frontend-design` — REQUIRED; establishes design token conventions and component patterns
- `ui-ux-pro-max` — REQUIRED; full design system generation for the dark techy theme
- `theme-factory` — CSS custom properties and design token architecture
- `brand-guidelines` — ensures visual consistency across all pages

### Prompt

```
Read CLAUDE.md, then read skills/public/frontend-design/SKILL.md,
skills/user/ui-ux-pro-max/SKILL.md, skills/public/theme-factory/SKILL.md,
and skills/public/brand-guidelines/SKILL.md before making any changes.

Read the frontend/ directory structure.

Design direction: DARK TECHY — neural terminal meets geometric puzzle.
Apply the design system defined in CLAUDE.md (tokens, fonts, colors).

1. Create frontend/app/globals.css:
   - CSS custom properties for the full color system from CLAUDE.md
   - A subtle 28px dot grid pattern as body background (pure CSS)
   - Custom scrollbar styling matching the dark theme
   - Selection highlight color (cyan at 30% opacity)
   - @import for Space Mono and Inter from Google Fonts

2. Create frontend/components/layout/Navbar.tsx:
   - Logo: stylized 9-cell SVG icon (cyan) + "SudokuAI" in Space Mono
   - Nav links: Solve / Train / Optimize
   - Active link: cyan underline + subtle glow
   - Sticky, backdrop-blur-md, border-bottom with 1px glow line

3. Create frontend/components/layout/Shell.tsx:
   - Wraps all pages with Navbar + main + subtle footer
   - Max-width 1280px, horizontally centered

4. Create frontend/app/layout.tsx:
   - Root layout with Shell, global error boundary, shadcn/ui Toaster
   - Correct <html lang="en"> and metadata

5. Create frontend/app/page.tsx (landing):
   - Hero: "Solve Any Sudoku Instantly" with animated digit counter
   - Three feature cards: CV Grid Detection / CNN Digit Recognition /
     Backtracking Solver
   - Stats row: 99.2% accuracy / <2ms inference / Supports EN + FA digits
   - CTA button → /solve

Use only Tailwind utility classes. No inline styles.
Animations via Tailwind + keyframes in globals.css.
```

### Skills — Apply After Completion
- `typescript-lsp` — type-check all new components
- `webapp-testing` — visual smoke test of landing page at localhost:3000
- `caveman-review` — review design consistency against CLAUDE.md tokens
- `caveman-commit` — `feat(design): design system, layout components, landing page`

---

## Section 6 — Solve Page (Main Feature)

### Skills — Read Before Starting
- `frontend-design` — REQUIRED; component composition patterns
- `ui-ux-pro-max` — REQUIRED; file upload UX, result display patterns
- `typescript-lsp` — type the API response shapes before building UI

### Prompt

```
Read CLAUDE.md, then read skills/public/frontend-design/SKILL.md and
skills/user/ui-ux-pro-max/SKILL.md before making any changes.

Read frontend/app/page.tsx and the /api/solve response schema in CLAUDE.md.

Build frontend/app/solve/page.tsx — the main inference page.

LEFT PANEL — Upload & Controls:
1. DropZone component (react-dropzone):
   - Dashed cyan border on hover
   - Preview thumbnail of uploaded image
   - Accepts jpg/png/jpeg only, max 10MB
2. "Solve Sudoku" button (full width, cyan)
3. Loading state: animated spinner with step labels:
   "Detecting grid..." → "Extracting cells..." →
   "Running CNN..." → "Solving puzzle..."

RIGHT PANEL — Results (shown after successful solve):
Tabs: "Solution" | "Cell Analysis" | "Raw Grid"

"Solution" tab:
  - Side-by-side original + solved base64 images
  - "Copy Grid" button (copies 9x9 as text)

"Cell Analysis" tab:
  - 9x9 grid of cell cards:
    * 28x28 base64 cell image
    * Predicted digit (large, cyan) or center dot for empty
    * Confidence bar using colors from CLAUDE.md confidence scale
  - Box separators every 3 rows and columns
  - Summary stats: digit count, avg confidence, low-confidence warnings

"Raw Grid" tab:
  - Styled 9x9 of raw predicted values (pre-solve)
  - Highlight low-confidence cells in amber

Error states: "No grid detected" / "Unsolvable puzzle" / "Upload failed"
All fetch calls to NEXT_PUBLIC_API_URL/api/solve.
Use React useState/useCallback only — no external state library.
```

### Skills — Apply After Completion
- `typescript-lsp` — verify all API response types are correct
- `webapp-testing` — upload a test image and verify the full flow
- `caveman-review` — review UX flow and error handling
- `caveman-commit` — `feat(solve): main inference page with upload, results, cell analysis`

---

## Section 7 — Training Dashboard Page

### Skills — Read Before Starting
- `frontend-design` — REQUIRED; chart and dashboard layout patterns
- `typescript-lsp` — type the SSE event payloads before wiring EventSource

### Prompt

```
Read CLAUDE.md, then read skills/public/frontend-design/SKILL.md before
making any changes.

Build frontend/app/train/page.tsx — the live training dashboard.

TOP — Configuration Panel:
1. Epochs input (1–100, default 20)
2. Learning Rate input (default 0.001, step 0.0001)
3. Batch Size select: 32 / 64 / 128 / 256 (default 128)
4. Dataset radio: "MNIST + Fonts" | "MNIST + Hoda" | "All Datasets"
5. "Start Training" button — disabled during training, pulsing indicator

MIDDLE — Live Charts (appear once training starts, using recharts):
- Loss curve: Train Loss (cyan) + Val Loss (amber) LineChart
- Accuracy curve: Train Acc (cyan) + Val Acc (amber) LineChart
- Progress bar: current epoch / total epochs above charts

BOTTOM — Results (appear on training complete):
1. Metric cards: Final Test Accuracy, Test Loss, Best Val Loss
2. 10x10 Confusion Matrix (HTML table, color-scaled):
   - Diagonal: green
   - Off-diagonal: red-scaled by value
3. Per-class accuracy table: Class / Correct / Total / Accuracy
4. Download report button (GET /api/reports/training_report.txt)

Connect to GET /api/train/stream via EventSource.
Handle SSE events per the schema in CLAUDE.md.
Handle connection drops with automatic reconnect (max 3 retries).
Use the useTrainingStream hook (to be built in Section 9).
For now, inline the EventSource logic with a TODO comment marking the hook.
```

### Skills — Apply After Completion
- `typescript-lsp` — type all SSE event payloads and chart data structures
- `webapp-testing` — run a short training (1 epoch) and verify live chart updates
- `caveman-commit` — `feat(train): live training dashboard with SSE charts and confusion matrix`

---

## Section 8 — Optimization Page

### Skills — Read Before Starting
- `frontend-design` — REQUIRED; benchmark table and accordion patterns
- `typescript-lsp` — type the benchmark response

### Prompt

```
Read CLAUDE.md, then read skills/public/frontend-design/SKILL.md before
making any changes.

Build frontend/app/optimize/page.tsx — the benchmark panel.

1. Header with description of TorchScript vs ONNX tradeoffs.

2. Model status card (GET /api/model/info on mount):
   - Shows: model exists, file size, parameter count, last trained date
   - If model not found: warning banner with link to /train

3. "Run Optimization & Benchmark" button:
   - POST /api/optimize on click (loading spinner, ~10s)

4. Results table (after benchmark completes):
   Columns: Format / Size (MB) / Latency (ms) / Status
   - Fastest format highlighted with amber border
   - Bar chart (recharts) comparing latencies

5. Deployment instructions accordion:
   - TorchScript: Python/C++ loading snippet with copy button
   - ONNX: onnxruntime snippet with copy button
   Both snippets use the CodeBlock component (built in Section 9;
   for now inline a simple <pre> with copy button).

6. Download buttons for each format
   (GET /api/models/download?format=ts|onnx|pt)
```

### Skills — Apply After Completion
- `typescript-lsp` — type-check the page
- `caveman-commit` — `feat(optimize): benchmark panel with download and deployment snippets`

---

## Section 9 — Reusable UI Components

### Skills — Read Before Starting
- `frontend-design` — REQUIRED; component API design and composition
- `ui-ux-pro-max` — REQUIRED; component-level design system application
- `typescript-lsp` — all components must have complete prop types

### Prompt

```
Read CLAUDE.md, then read skills/public/frontend-design/SKILL.md and
skills/user/ui-ux-pro-max/SKILL.md before making any changes.

Read frontend/components/ to see what already exists.

Build these shared components:

1. frontend/components/ui/SudokuBoard.tsx
   Props: grid (number[][], 9x9), highlights?: Set<string> (e.g. "0,1")
   - Thicker borders every 3 cells (3x3 box boundaries)
   - Cyan for solver-filled cells (0 in original grid)
   - Amber for low-confidence cells
   - Space Mono font, centered digits

2. frontend/components/ui/ConfidenceBadge.tsx
   Props: confidence (0–1)
   - Color scale per CLAUDE.md (green/amber/red)
   - Dot + percentage display

3. frontend/components/ui/StatCard.tsx
   Props: label, value, unit?, icon?
   - Subtle border glow on hover
   - Space Mono for value, Inter for label

4. frontend/components/ui/CodeBlock.tsx
   Props: code (string), language (string)
   - Dark pre/code block with copy button
   - Space Mono font

5. frontend/components/charts/LossAccuracyChart.tsx
   Props: trainLoss[], valLoss[], trainAcc[], valAcc[]
   - Two recharts LineChart in responsive containers
   - Correct axes, tooltips, legend
   - Colors from CLAUDE.md design tokens

6. frontend/hooks/useTrainingStream.ts
   Custom hook:
   - Manages EventSource to /api/train/stream
   - Returns { isTraining, epochData, metrics, error, startTraining }
   - Handles cleanup on unmount
   - Reconnect logic (max 3 retries, exponential backoff)

After building components, replace inline implementations in
train/page.tsx and optimize/page.tsx with the new components.
```

### Skills — Apply After Completion
- `typescript-lsp` — type-check all components and the custom hook
- `webapp-testing` — verify components render correctly in all pages
- `caveman-review` — review component API consistency
- `caveman-commit` — `feat(components): reusable UI components and useTrainingStream hook`

---

## Section 10 — Error Handling, Loading States, and Tests

### Skills — Read Before Starting
- `playwright-skill` — REQUIRED; write end-to-end tests for the full user flows
- `webapp-testing` — REQUIRED; loading skeleton and error boundary patterns
- `security-guidance` — final security review before deployment prep
- `code-review` (caveman-review) — full codebase review pass

### Prompt

```
Read CLAUDE.md, then read skills/user/playwright-skill/SKILL.md,
skills/public/webapp-testing/SKILL.md, skills/public/security-guidance/SKILL.md,
and skills/user/caveman-review/SKILL.md before making any changes.

Read all frontend pages and components built so far.

Add production-quality polish:

1. Global error boundary in frontend/app/layout.tsx.

2. API client frontend/lib/api.ts:
   - Typed functions: solveSudoku(file), startTraining(config),
     getModelInfo(), runOptimization()
   - Base URL from NEXT_PUBLIC_API_URL env var
   - Error parsing from FastAPI error responses
   - AbortController support

3. shadcn/ui Toaster notifications:
   - Success: "Puzzle solved!" / "Training complete!" / "Model exported!"
   - Error: "No grid detected" / "Training failed: {message}"
   - Warning: "Low confidence on {n} cells"

4. Loading skeletons:
   - SudokuBoard skeleton (pulsing grey grid)
   - Cell grid skeleton (81 pulsing boxes)
   - Chart skeleton (pulsing line shape)

5. Responsive fixes:
   - Mobile: solve page stacks vertically, cell grid scrolls horizontally
   - All touch targets >= 48px

6. Keyboard accessibility:
   - Tab order on solve page: dropzone → submit → results tabs
   - focus-visible styles (cyan outline) on all interactive elements
   - aria-labels on icon-only buttons

7. 404 page (frontend/app/not-found.tsx) with Sudoku pun and home link.

8. Playwright tests (frontend/tests/):
   - solve.spec.ts: upload image → verify solved grid appears
   - train.spec.ts: start training → verify SSE chart updates
   - optimize.spec.ts: run benchmark → verify results table
   - navigation.spec.ts: all nav links resolve without 404

9. Create .env.local.example with NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Skills — Apply After Completion
- `playwright-skill` — run full test suite: `npx playwright test`
- `security-guidance` — final review of all API calls and file handling
- `caveman-review` — full codebase quality pass
- `session-report` — generate final pre-deployment report
- `caveman-commit` — `feat(polish): error boundaries, skeletons, a11y, playwright tests`

---

## Section 11 — Docker & Deployment

### Skills — Read Before Starting
- `claude-code-setup` — Docker and deployment configuration conventions
- `security-guidance` — Docker security: non-root user, no secrets in image

### Prompt

```
Read CLAUDE.md, then read skills/public/claude-code-setup/SKILL.md and
skills/public/security-guidance/SKILL.md before making any changes.

Create production-ready deployment configuration.

1. backend/Dockerfile:
   - FROM python:3.11-slim
   - Non-root user (adduser --disabled-password appuser)
   - Install OpenCV system deps: libglib2.0-0 libsm6 libxrender1 libxext6
   - Copy requirements, install, copy app
   - CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000

2. frontend/Dockerfile:
   - Multi-stage build (node:20-alpine builder + runner)
   - Build arg: NEXT_PUBLIC_API_URL
   - Non-root user in runner stage

3. docker-compose.yml (complete):
   - backend: build ./backend, port 8000, volume ./models:/app/models,
     volume ./data:/app/data
   - frontend: build ./frontend, port 3000, depends_on backend
   - Shared bridge network

4. backend/app/routers/models.py — file download endpoints:
   - GET /api/models/download?format=pt|ts|onnx → FileResponse
     (validate format param; reject path traversal attempts)
   - GET /api/reports/{filename} → FileResponse
     (whitelist filenames; do not allow arbitrary path access)

5. backend/.env.example:
   MODEL_PATH=models/best_model.pt
   DATA_PATH=data
   DEVICE=cpu
   CORS_ORIGINS=http://localhost:3000

6. README.md at repo root:
   - Quick start: docker-compose up --build
   - Manual setup: backend pip install / frontend npm install
   - Environment variables table
   - API endpoints summary (copy from CLAUDE.md)
   - Screenshots section placeholder
```

### Skills — Apply After Completion
- `security-guidance` — verify no secrets in Dockerfiles, path traversal protection on download endpoints
- `caveman-review` — review docker-compose for correctness
- `session-report` — final project completion report
- `caveman-commit` — `feat(docker): production Dockerfiles and compose with security hardening`

---

## Skill Quick Reference

| Skill | When to Use |
|---|---|
| `frontend-design` | Before every frontend section (5–10) |
| `ui-ux-pro-max` | Sections 5, 6, 9 — design system and component UX |
| `playwright-skill` | Section 10 — writing and running E2E tests |
| `webapp-testing` | Sections 5–10 — visual and functional verification |
| `pyright-lsp` | Sections 1–4 — Python type checking |
| `typescript-lsp` | Sections 5–10 — TypeScript type checking |
| `security-guidance` | Sections 2, 4, 10, 11 — API and Docker security |
| `ralph-loop` | Section 3 — long-running training loop pattern |
| `claude-code-setup` | Sections 1, 11 — project init and deployment |
| `feature-dev` | Section 1 — scaffolding discipline |
| `caveman-review` | After every section — code review pass |
| `caveman-commit` | After every section — structured commit messages |
| `session-report` | Sections 3, 10, 11 — handoff and completion notes |
| `theme-factory` | Section 5 — CSS custom properties architecture |
| `brand-guidelines` | Section 5 — visual consistency |
| `superpowers-marketplace` | Optionally at Section 9 — additional component patterns |

---

## Tips for Claude Code Sessions

### Referencing the Vision Pipeline
```
The vision pipeline MUST preserve all three strategies from vision.py:
1. Contour path (locate_cells_within_grid) — primary
2. KMeans centroid recovery (build_grid_from_partial_cells) — partial grids
3. Deterministic slice (slice_grid_into_cells) — fallback
Do not simplify. These exist because real Sudoku images are messy.
```

### For SSE Training
```
The training loop runs in a thread via run_in_executor.
Communication to the SSE endpoint must go through asyncio.Queue
drained by a coroutine on the main event loop.
Do not use asyncio.run() inside the training thread.
```

### When Starting Any Session
```
Read CLAUDE.md first. Then read the skill files listed for this section.
Then read the relevant source files. Then — and only then — begin writing code.
```
