# Sudoku Solver — Full-Stack AI Application

An end-to-end computer vision and deep learning pipeline that detects, extracts, recognizes, and solves Sudoku puzzles from real-world images. Built as a production-grade full-stack application with a FastAPI backend and Next.js frontend.

**Stack:** Python 3.11 · FastAPI · PyTorch · OpenCV · Next.js 16 · TypeScript · Tailwind CSS · Docker

---

## Architecture

```
sudoku-solver/
├── backend/          # FastAPI + PyTorch + OpenCV
│   ├── app/
│   │   ├── main.py           # FastAPI app, lifespan, CORS
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── routers/          # API route handlers
│   │   └── core/             # ML pipeline (model, vision, solver, trainer)
│   └── requirements.txt
├── frontend/         # Next.js 16 App Router + TypeScript
│   ├── app/          # Pages (solve, train, optimize)
│   ├── components/   # Reusable UI components
│   └── lib/          # Typed API client
├── models/           # Model weights (.pt, .ts, .onnx)
├── data/             # Training datasets
└── docker-compose.yml
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
# Clone
git clone https://github.com/MohammadMahdi-Sharifbeigy/Sudoku-Pytorch.git
cd Sudoku-Pytorch

# Run both services
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

### Option B — Local Development

Run backend and frontend separately. See their individual READMEs:

- [`backend/README.md`](./backend/README.md) — Python setup, env vars, API reference
- [`frontend/README.md`](./frontend/README.md) — Node setup, env vars, pages

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/solve` | Upload image → solved board + cell data |
| GET | `/api/train/stream` | SSE stream of live training events |
| GET | `/api/train/status` | `{ is_training: bool }` |
| POST | `/api/optimize` | Export TorchScript + ONNX, run benchmark |
| GET | `/api/model/info` | Model file metadata + parameter counts |
| GET | `/api/models/download` | Download model file by `?format=pt\|ts\|onnx` |
| GET | `/api/reports/{filename}` | Download training or inference report |

---

## Model Weights

Pre-trained CNN weights are at `models/best_model.pt`. The model is loaded once at startup via FastAPI lifespan.

Optimized exports (generated via `/api/optimize`):
- `models/best_model.ts` — TorchScript
- `models/best_model.onnx` — ONNX

---

## Pipeline Overview

1. **Grid Extraction** — adaptive thresholding, contour detection, perspective warp (OpenCV)
2. **Cell Segmentation** — three-strategy pipeline: contour → KMeans → slice fallback
3. **Digit Recognition** — custom CNN trained on MNIST + Hoda (English + Persian digits)
4. **Puzzle Solving** — recursive backtracking algorithm
5. **Result Overlay** — solved digits re-projected onto the original image

---

## License

MIT — see [LICENSE](./LICENSE)
