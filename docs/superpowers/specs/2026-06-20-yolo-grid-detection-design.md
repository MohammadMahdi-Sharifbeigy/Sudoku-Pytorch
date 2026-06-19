# YOLOv8 Grid Detection + Model Selection — Design

Date: 2026-06-20
Branch: `fullapp`
Status: Approved

## Goal

Upgrade the Sudoku solver's grid-extraction step from classical CV (adaptive
threshold + contours) to an optional YOLOv8 detector that is robust to shadows,
noise, and warped paper. Add per-request model selection (CNN + YOLO) across the
backend and frontend, plus YOLO training from both a CLI script and the web UI.

Classical detection remains as a fallback and a selectable mode.

## Decisions (locked)

- **Detector mode:** selectable `classical | yolo`. Default = `yolo` when a YOLO
  model is loaded, else `classical`. YOLO path auto-falls-back to classical on
  detection failure.
- **YOLO training:** both a CLI script and an in-app SSE endpoint + frontend tab.
- **Model picker:** two dropdowns — CNN checkpoints and YOLO weights — on the
  solve page. Optimize page gets the CNN dropdown only.
- **Restructure:** reorganize `vision` and model-file helpers into packages.
- **Dataset location:** `data/Sudoku-Detector.yolov8/` (moved under `data/`).

## Architecture

### Backend package reorg

```
backend/app/core/
  vision/
    __init__.py     # re-exports every symbol inference.py imports (back-compat)
    common.py       # detector-agnostic helpers
    classical.py    # contour-based detection (existing logic, relocated)
    yolo.py         # YOLO detection + cell extraction
  registry/
    __init__.py
    model_files.py  # relocated from core/ (filenames, pointers, artifact paths)
    cnn_models.py   # list CNN checkpoints + path-keyed cached loader
    yolo_models.py  # list YOLO weights + path-keyed cached loader
  train_yolo.py     # core ultralytics trainer (shared by CLI + endpoint)
  model.py, solver.py, train.py, optimize_model.py, data_utils.py, report_utils.py
```

`vision/__init__.py` re-exports all names so existing
`from app.core.vision import ...` imports keep working. `registry/model_files.py`
move requires updating imports in `main.py`, `routers/training.py`,
`routers/models.py`, `routers/optimization.py`.

### vision/common.py (detector-agnostic)

Holds everything both detectors share:
`resize_and_maintain_aspect_ratio`, `apply_grayscale_blur_and_threshold`,
`get_quadrilateral_points_in_order`, `perform_four_point_transform`,
`center_and_resize_digit`, `check_for_digit_in_cell_image`, `_bbox_iou`, `_nms`,
`_contour_to_quad`, `slice_grid_into_cells`, `sort_cells_into_grid`,
`build_grid_from_partial_cells`, `_clear_border_components`,
`get_predicted_sudoku_grid_torch`, `get_per_cell_predictions`,
`plot_cell_images_in_grid`, `generate_solution_image`.

### vision/classical.py

`find_grid_contour_candidates`, `locate_cells_within_grid`, and
`get_cells_classical` (renamed from `get_valid_cells_from_image`; old name kept
as an alias in `__init__.py`).

### vision/yolo.py

```python
def load_yolo_model(weights_path: str, device: torch.device) -> "YOLO": ...

def get_grid_corners_yolo(
    img: np.ndarray, yolo_model: "YOLO", conf: float = 0.25
) -> np.ndarray:
    """Detect grid bbox, refine to 4 exact corners, return ordered TL,TR,BR,BL."""

def get_cells_yolo(
    img: np.ndarray, yolo_model: "YOLO", grid_size: int = 576
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """corners -> square warp -> cells. Same tuple shape as classical path."""
```

**`get_grid_corners_yolo` logic:**
1. `yolo_model.predict(img, conf=conf, verbose=False)` → pick highest-confidence
   box. Raise `ValueError("YOLO found no grid")` if none.
2. bbox `(x1,y1,x2,y2)` → pad ~3% of bbox size, clamp to image bounds.
3. **Corner refinement within bbox** (per rule 3): crop the padded region →
   grayscale → `apply_grayscale_blur_and_threshold` → largest external contour →
   `cv2.minAreaRect` (or `_contour_to_quad` convex-hull extreme points) → 4 exact
   corners mapped back to full-image coords. If the contour is too small/weak
   (area < ~40% of bbox), fall back to the 4 axis-aligned bbox corners.
4. Order via `get_quadrilateral_points_in_order`, return `np.float32` (4,2).

**`get_cells_yolo` logic:**
- `M, grid_sq = perform_four_point_transform(img, corners, size=grid_size)`
- Try `locate_cells_within_grid(grid_sq)`; if 81 cells → use it (true centroids).
  Else if ≥54 → `build_grid_from_partial_cells`. Else → `slice_grid_into_cells`
  (deterministic fallback, primary per task description).
- Return `(cells, M, grid_sq)`.

### Detector selection in the pipeline

`routers/inference._run_pipeline` becomes detector-aware:

```python
def _run_pipeline(img_bytes, cnn_model, device, detector, yolo_model) -> dict:
    ...
    if detector == "yolo" and yolo_model is not None:
        try:
            cells, M, board = get_cells_yolo(img, yolo_model)
            detector_used = "yolo"
        except Exception:
            cells, M, board = get_cells_classical(img)
            detector_used = "classical (yolo fallback)"
    else:
        cells, M, board = get_cells_classical(img)
        detector_used = "classical"
```

`/api/solve` (multipart) gains optional form fields: `detector`, `cnn_model`,
`yolo_model`. Default detector resolved server-side:
`yolo` if `app.state.yolo_model` is loaded, else `classical`.

CNN/YOLO models resolved by id through the registry's cached loaders
(path-keyed, so repeated requests reuse the in-memory model). Unset ids → use
`app.state` defaults. `SolveResponse` gains `detector_used: str`.

### Model storage convention

- CNN checkpoints: `models/*.pt` (excluding `yolov8s.pt`). Pointer
  `models/latest_model.txt` (existing).
- YOLO weights: `models/yolo/*.pt` plus seed `models/yolov8s.pt`. Pointer
  `models/yolo/latest_yolo.txt`.
- `GET /api/models/cnn` and `GET /api/models/yolo` →
  `[{id, filename, size_mb, created_at, is_default}]`.
  `id` = filename (validated: no separators / traversal).

### YOLO training — CLI + in-app

`core/train_yolo.py`:
```python
def train_yolo(
    data_yaml: str, seed_weights: str, epochs: int, imgsz: int, batch: int,
    device: str, models_dir: str, on_epoch=None,
) -> str:  # returns best-weights path
```
Uses `ultralytics.YOLO(seed_weights)`, registers an `on_train_epoch_end`
callback that pulls `box_loss/cls_loss/mAP50` from the trainer and calls
`on_epoch(metrics)`. Copies `runs/.../best.pt` → `models/yolo/sudoku_yolo_*.pt`
and writes the pointer.

`scripts/train_yolo.py`: argparse wrapper. Defaults: `--data
data/Sudoku-Detector.yolov8/data.yaml`, `--model models/yolov8s.pt`,
`--epochs 50`, `--imgsz 640`, `--batch 16`, `--device auto`. Device auto-detect:
`cuda` → `mps` → `cpu`.

`GET /api/train/yolo/stream` (SSE): mirrors the existing CNN trainer pattern —
thread-pool worker, `asyncio.Queue`, sentinel, 30s keepalive ping, concurrency
guard (`_is_yolo_training`). Emits `{type:"epoch", box_loss, cls_loss, map50,
epoch, epochs}`, `{type:"best_model"}`, `{type:"complete"}`, `{type:"error"}`.
On completion reloads `app.state.yolo_model` from the new best weights.

Optimize endpoint (`POST /api/optimize`) gains optional `cnn_model` (CNN-only).

### data.yaml fix

Current `data/Sudoku-Detector.yolov8/data.yaml` uses `train: ../train/images`,
which resolved relative to the old repo-root location. Now nested under `data/`,
rewrite to be self-rooted:
```yaml
path: .            # dir of this file
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['Sudoku-Detector']
```

### App state / lifespan (`main.py`)

- Load default CNN as today.
- Resolve default YOLO weights: `YOLO_MODEL_PATH` env, else
  `models/yolo/latest_yolo.txt` pointer, else newest `models/yolo/*.pt`. If none,
  `app.state.yolo_model = None` and the solve default falls to classical.
- `app.state.cnn_cache = {}`, `app.state.yolo_cache = {}` (path-keyed loaders).

### Schemas (`schemas.py`)

Add `ModelEntry`, `ModelList`, `YoloTrainingConfig`. Extend `SolveResponse` with
`detector_used: str`.

## Frontend

Skills: `frontend-design` + `ui-ux-pro-max` for new components; `caveman-review`
as a final pass. Existing dark/cyan theme and Space-Mono accents preserved.

- `lib/api.ts`: `listCnnModels`, `listYoloModels`,
  `solveSudoku(file, {detector, cnnModel, yoloModel})`, `yoloTrainingStreamUrl`,
  `runOptimization({cnnModel})`. Extend `SolveResponse` with `detector_used`.
- New `components/ui/ModelSelect.tsx` — themed dropdown (filename, size, "default"
  badge), reused by solve + optimize.
- **Solve page:** segmented detector toggle (Classical │ YOLO) + CNN dropdown +
  YOLO dropdown (shown only when detector = YOLO). Show `detector_used` in the
  result header.
- **Optimize page:** CNN dropdown feeding `runOptimization`.
- **Train page:** CNN │ YOLO sub-tab. YOLO form (epochs, imgsz, batch) + live
  box-loss / mAP50 chart, reusing the SSE stream hook pattern.

## Testing / verification

- Backend: `python -c "import app.main"` import-smoke after the reorg;
  `uvicorn` boot; `GET /api/models/cnn|yolo`; `/api/solve` with each detector on a
  sample from `data/sudoku_images/`. `get_grid_corners_yolo` unit check on one
  labeled image (corners inside bounds, 4 points).
- `scripts/train_yolo.py --epochs 1` smoke (CPU) to confirm wiring + best-weights
  copy + pointer.
- Frontend: `npm run build` / lint; Playwright specs updated for new controls.

## Risks

- `ultralytics` is a heavy dependency (pulls its own torch pin) → add to
  `backend/requirements.txt`, verify it coexists with the current torch.
- YOLO SSE training is GPU/CPU intensive → single-flight concurrency guard.
- `yolov8s.pt` (21 MB) is untracked (training seed) — keep local, document in
  README; trained weights (`models/yolo/*.pt`) are small.
- Back-compat: the `vision` package must re-export every symbol the routers
  currently import; covered by the import-smoke test.

## Out of scope

- Replacing the backtracking solver or the DigitCNN architecture.
- Retraining / benchmarking YOLO accuracy beyond a 1-epoch wiring smoke.
- Dockerfile changes beyond adding the new dependency.
</content>
</invoke>
