# Backend — FastAPI + PyTorch + OpenCV

FastAPI service exposing the Sudoku vision pipeline, CNN inference, live training stream, and model optimization endpoints.

---

## Requirements

- Python 3.11+
- pip or `uv`
- (Optional) CUDA-capable GPU for faster training

---

## Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best_model.pt` | Fallback PyTorch model path when no latest trained checkpoint exists |
| `MODELS_DIR` | `models` | Directory for all model files |
| `DATA_PATH` | `data` | Path to training data directory |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `YOLO_MODEL_PATH` | _(unset)_ | Force a specific YOLOv8-pose checkpoint for grid detection |

> Paths are relative to the **repository root**, not the `backend/` directory, because the backend mounts `../models` and `../data` at runtime.

---

## Running

### Development

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1` — the training loop uses a global asyncio lock that assumes a single process.

### Docker

```bash
# From repo root
docker-compose up backend
```

---

## API Endpoints

### POST `/api/solve`

Upload a Sudoku image and get back the solved grid with per-cell confidence data.

**Request:** `multipart/form-data` with field `file` (JPEG or PNG, max 10 MB)

**Response:**
```json
{
  "success": true,
  "original_grid": [[0, 0, 3, ...], ...],
  "solved_grid": [[8, 1, 3, ...], ...],
  "cells": [
    {
      "row": 0, "col": 0, "value": 8,
      "confidence": 0.97, "is_given": false,
      "image_b64": "<base64 PNG>"
    }
  ],
  "original_image_b64": "<base64 PNG>",
  "solved_image_b64": "<base64 PNG>",
  "solve_time_ms": 42.1
}
```

---

### GET `/api/train/stream`

Server-Sent Events (SSE) stream for live training. Pass config as query parameters.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | `20` | Number of training epochs (1–100) |
| `lr` | float | `0.001` | Learning rate |
| `batch_size` | int | `128` | Batch size (32, 64, 128, 256) |
| `dataset` | string | `all` | Dataset selection: `mnist_fonts`, `mnist_hoda`, `all` |

**Example:**
```
GET /api/train/stream?epochs=20&lr=0.001&batch_size=128&dataset=all
```

**SSE event types:**

```json
{ "type": "epoch",        "epoch": 5, "epochs": 20, "train_loss": 0.12, "val_loss": 0.09, "train_acc": 94.5, "val_acc": 96.2 }
{ "type": "best_model",   "val_loss": 0.08, "epoch": 8 }
{ "type": "test_results", "test_loss": 0.09, "test_acc": 96.8, "y_true": [...], "y_pred": [...] }
{ "type": "complete",     "model_path": "models/sudoku_mnist-fonts_lr0p001_bs128_ep20.pt" }
{ "type": "error",        "message": "..." }
```

Training checkpoints are named from the run configuration:

```txt
sudoku_{dataset}_lr{learning_rate}_bs{batch_size}_ep{epochs}.pt
```

The backend also writes `models/latest_model.txt` so model info, optimization, inference reloads, and downloads can resolve the latest trained checkpoint without relying on a generic `best_model.pt` filename.

Returns **HTTP 409** if training is already in progress.

---

### GET `/api/train/status`

```json
{ "is_training": false }
```

---

### POST `/api/optimize`

Exports the model to TorchScript and ONNX, runs a CPU inference benchmark.

**Response:**
```json
{
  "success": true,
  "torchscript_path": "models/sudoku_mnist-fonts_lr0p001_bs128_ep20.ts",
  "onnx_path": "models/sudoku_mnist-fonts_lr0p001_bs128_ep20.onnx",
  "benchmark": {
    "pytorch_ms": 2.1,
    "torchscript_ms": 1.8,
    "onnx_ms": 1.2
  }
}
```

---

### GET `/api/model/info`

Returns metadata for all available model files.

---

### GET `/api/models/download`

Download a model file.

**Query:** `?format=pt` | `?format=ts` | `?format=onnx`

---

### GET `/api/reports/{filename}`

Download a saved report file (e.g., `training_report.txt`).

---

## Grid detection: YOLOv8-pose

In addition to the classical OpenCV contour pipeline, the grid corners can be
located with a YOLOv8-pose model. Convert the 4-corner polygon labels and train
from `backend/`:

```bash
.venv/Scripts/python.exe scripts/convert_labels_to_pose.py
.venv/Scripts/python.exe scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 8 --device auto
```

Best weights land in `models/yolo/sudoku_pose_*.pt` with the pointer
`models/yolo/latest_yolo.txt`. Pose becomes the default detector once trained
weights exist; set `YOLO_MODEL_PATH` to override, and the backend auto-falls back
to classical on failure. See the [root README](../README.md#grid-detection-yolov8-pose)
for the full workflow, the in-app **Grid Pose** training tab, and per-request
model selection.

---

## Project Structure

```
backend/
├── app/
│   ├── main.py            # FastAPI app, lifespan, CORS
│   ├── schemas.py         # All Pydantic models
│   └── routers/
│       ├── inference.py   # POST /api/solve
│       ├── training.py    # GET /api/train/stream, /api/train/status
│       ├── optimization.py# POST /api/optimize
│       └── models.py      # GET /api/model/info, /api/models/download
│   └── core/
│       ├── model.py       # DigitCNN, FocalLoss
│       ├── solver.py      # SudokuSolver (backtracking)
│       ├── vision.py      # CV pipeline (3-strategy cell extraction)
│       ├── train.py       # train_epoch, validate
│       └── data_utils.py  # DataLoaders
├── .env.example
├── pyrightconfig.json
└── requirements.txt
```

---

## Type Checking

```bash
cd backend
pyright app/
```
