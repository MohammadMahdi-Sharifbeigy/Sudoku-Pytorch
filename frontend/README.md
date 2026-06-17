# Frontend — Next.js 16 + TypeScript

Dark-techy UI for the Sudoku AI solver. Built with Next.js 16 App Router, TypeScript, Tailwind CSS v4, shadcn/ui, and recharts.

---

## Requirements

- Node.js 20+
- npm (or pnpm / bun)
- Backend running at `http://localhost:8000` (see `../backend/README.md`)

---

## Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.local.example .env.local
```

---

## Environment Variables

`.env.local`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

---

## Running

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Production Build

```bash
npm run build
npm start
```

### Type Check

```bash
npx tsc --noEmit
```

### Lint

```bash
npm run lint
```

### Docker

```bash
# From repo root
docker-compose up frontend
```

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page — project overview, feature highlights |
| `/solve` | Upload a Sudoku image, view solved grid + cell analysis |
| `/train` | Live training dashboard with real-time loss/accuracy charts |
| `/optimize` | Model benchmark panel — export TorchScript/ONNX, compare inference speeds |

---

## Solve Page (`/solve`)

- Drag-and-drop or click to upload a JPEG/PNG (max 10 MB)
- Animated loading steps: Detecting grid → Extracting cells → Running CNN → Solving puzzle
- Three result tabs:
  - **Solution** — overlay image with solved digits
  - **Cell Analysis** — 9×9 grid with per-cell confidence bars and extracted images
  - **Raw Grid** — original detected digits; low-confidence cells highlighted in amber

---

## Train Page (`/train`)

- Config panel: epochs (1–100), learning rate, batch size, dataset selector
- Connects to `GET /api/train/stream` via EventSource (SSE)
- Auto-reconnect on drop (max 3 retries, exponential backoff)
- Live recharts:
  - Loss chart (cyan = train, amber = val)
  - Accuracy chart (cyan = train, amber = val)
- Results section: metric cards, 10×10 confusion matrix, per-class accuracy table
- Download training report button

---

## Optimize Page (`/optimize`)

- Triggers `POST /api/optimize` to export TorchScript + ONNX
- Displays benchmark comparison (PyTorch vs TorchScript vs ONNX inference speed)
- Download buttons for each model format

---

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx         # Root layout, fonts, Toaster
│   ├── globals.css        # Design tokens, dot-grid pattern, keyframes
│   ├── page.tsx           # Landing page
│   ├── solve/page.tsx     # Inference UI
│   ├── train/page.tsx     # Training dashboard
│   ├── optimize/page.tsx  # Benchmark panel
│   └── not-found.tsx      # 404 page
├── components/
│   ├── layout/
│   │   ├── Navbar.tsx
│   │   └── Shell.tsx
│   ├── ui/
│   │   ├── SudokuBoard.tsx
│   │   ├── ConfidenceBadge.tsx
│   │   ├── StatCard.tsx
│   │   └── CodeBlock.tsx
│   └── charts/
│       ├── LossAccuracyChart.tsx
│       └── ConfusionMatrix.tsx
├── hooks/
│   └── useTrainingStream.ts
├── lib/
│   └── api.ts             # Typed API client with AbortController support
├── .env.local.example
└── package.json
```

---

## Design System

Theme: **Dark Techy** — neural terminal meets geometric puzzle.

| Token | Value |
|-------|-------|
| Background | `#0A0A0F` |
| Surface | `#12121A` |
| Border | `#1E1E2E` |
| Cyan (primary) | `#00D4FF` |
| Amber (secondary) | `#FFB800` |
| Error | `#FF4560` |
| Success | `#00E396` |
| Heading font | Space Mono |
| Body font | Inter |

Confidence color scale:
- ≥ 80% → `#00E396` (green)
- 50–79% → `#FFB800` (amber)
- < 50% → `#FF4560` (red)

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `next` | 16.2.9 | App framework |
| `react` | 19.2.4 | UI runtime |
| `recharts` | 3.8.1 | Training charts |
| `react-dropzone` | 15.x | File upload |
| `sonner` | 2.x | Toast notifications |
| `tailwindcss` | 4.x | Styling |
| `shadcn` | 4.x | Component primitives |
| `lucide-react` | 1.x | Icons |
