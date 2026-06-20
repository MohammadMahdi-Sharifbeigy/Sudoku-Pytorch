# Sudoku Solver

Full-stack Sudoku solver that detects a puzzle from an image, recognizes digits with a PyTorch CNN, solves the board, and serves training plus model optimization workflows through a FastAPI backend and Next.js frontend.

**Stack:** Python 3.11, FastAPI, PyTorch, OpenCV, Next.js 16, TypeScript, Tailwind CSS, Docker

## Quick Start

```bash
docker-compose up --build
```

Services:

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

The compose setup mounts:

| Host path | Container path | Purpose |
|---|---|---|
| `./models` | `/app/models` | Trained `.pt`, `.ts`, `.onnx`, reports |
| `./data` | `/app/data` | Training datasets |

## Manual Setup

Backend:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- -p 3000
```

For local frontend development, set:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Environment Variables

| Variable | Used by | Default | Description |
|---|---|---|---|
| `MODEL_PATH` | backend | `models/best_model.pt` | Fallback model path when no latest trained checkpoint exists |
| `MODELS_DIR` | backend | `models` | Directory for model artifacts and reports |
| `DATA_PATH` | backend | `data` | Dataset directory mounted into the backend container |
| `DEVICE` | backend | `cpu` | Inference/training device hint |
| `CORS_ORIGINS` | backend | `http://localhost:3000` | Comma-separated allowed frontend origins |
| `NEXT_PUBLIC_API_URL` | frontend | `http://localhost:8000` | Public browser URL for backend API requests |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/solve` | Upload image, returns solved board and per-cell data |
| `GET` | `/api/train/stream` | SSE stream of training events |
| `GET` | `/api/train/status` | Returns `{ is_training: bool }` |
| `POST` | `/api/optimize` | Exports TorchScript and ONNX, returns benchmark results |
| `GET` | `/api/model/info` | Model metadata and parameter counts |
| `GET` | `/api/models/download` | Download latest model artifact by `?format=pt|ts|onnx` |
| `GET` | `/api/reports/{filename}` | Download whitelisted text reports |

## Model Artifacts

Training saves run-specific checkpoint names:

```txt
models/sudoku_{dataset}_lr{learning_rate}_bs{batch_size}_ep{epochs}.pt
```

The backend writes `models/latest_model.txt` after training so model info, optimization, and downloads resolve the latest checkpoint without relying on a generic filename.

Optimization exports use the same base name:

```txt
models/sudoku_mnist-fonts_lr0p001_bs128_ep20.pt
models/sudoku_mnist-fonts_lr0p001_bs128_ep20.ts
models/sudoku_mnist-fonts_lr0p001_bs128_ep20.onnx
```

## Grid detection: YOLOv8-pose

The grid can be located two ways: the **classical** OpenCV contour pipeline, or a
**YOLOv8-pose** model that regresses the four grid corners as keypoints. The pose
detector is more robust on photographed, skewed, or low-contrast boards.

### Dataset

The annotated grid dataset (`Sudoku-Detector.yolov8/`) ships 4-corner **polygon**
labels. Convert them to YOLO pose keypoint labels before training:

```bash
cd backend && .venv/Scripts/python.exe scripts/convert_labels_to_pose.py
```

This creates `data/sudoku_pose/{train,valid,test}/{images,labels}/` plus
`data/sudoku_pose/data.yaml`. The script is idempotent.

> The dataset is tiny — **18 train / 5 valid / 2 test** images. Add more annotated
> sudoku images to `data/sudoku_pose/` (or re-run the converter on a larger
> source set) for meaningfully better corner-detection accuracy.

### Training

CLI (from `backend/`):

```bash
.venv/Scripts/python.exe scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 8 --device auto
```

In-app: open the **Train** page → **Grid Pose** tab, set epochs / image size / batch,
and stream box loss, pose loss, and mAP50 live per epoch.

Training starts from the seed weights `models/yolov8s-pose.pt`. Best weights are
written to `models/yolo/sudoku_pose_*.pt`, and the pointer file
`models/yolo/latest_yolo.txt` records the latest trained checkpoint.

### Detector selection

- The **pose detector becomes the default once trained weights exist** in
  `models/yolo/` (resolved via `latest_yolo.txt`). The untrained seed
  `yolov8s-pose.pt` is *not* used as a detector, so until you train, the default
  stays classical.
- On the **Solve** page, the **Classical / YOLO** toggle overrides the detector
  per request.
- Set `YOLO_MODEL_PATH` to force a specific pose checkpoint.
- If pose inference fails or no weights are available, the backend
  **auto-falls back to the classical** pipeline.

### Per-request model selection

- **Solve** page exposes two dropdowns: the digit **CNN** model and (when YOLO is
  selected) the grid **pose** model.
- **Optimize** page exposes the **CNN** model dropdown.

In both cases, leaving a dropdown on *Default (server)* uses the latest trained
checkpoint resolved by the backend.

## Screenshots

Add screenshots before release:

| Page | Screenshot |
|---|---|
| Solve | TODO |
| Train | TODO |
| Optimize | TODO |

## License

MIT. See [LICENSE](./LICENSE).
