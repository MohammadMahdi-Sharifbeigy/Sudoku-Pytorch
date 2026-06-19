# YOLOv8-Pose Grid Corner Detection + Model Selection — Design

Date: 2026-06-20 (revised — switched detection → pose)
Branch: `fullapp`
Status: Approved (pose revision)

## Goal

Upgrade the Sudoku solver's grid-extraction step to a **YOLOv8-pose** model that
predicts the **4 corner keypoints** of the Sudoku grid directly. The 4 corners
feed straight into the existing `perform_four_point_transform` — no bounding-box
contour refinement needed. Add per-request CNN/YOLO model selection across
backend + frontend, plus pose training from a CLI script and an in-app SSE
endpoint.

Classical contour detection is kept as a selectable fallback (auto-fallback when
pose finds no grid).

## Why pose (key finding)

The dataset labels (`data/Sudoku-Detector.yolov8/*/labels/*.txt`) are **already
4-corner annotations** — each line is 9 columns:
`class x1 y1 x2 y2 x3 y3 x4 y4` (normalized polygon, one grid per image; 18 train
/ 5 val / 2 test). Pose training needs a different label layout
(`class cx cy w h px1 py1 v1 …`), so a one-time **label conversion** produces a
pose dataset. The seed weights `models/yolov8s-pose.pt` are already present.

## Decisions (locked)

- **Detector:** YOLOv8-pose predicts 4 corners directly. Default = `yolo` when
  pose weights are loaded, else `classical`. Pose auto-falls-back to classical on
  detection failure. Classical toggle stays.
- **Label conversion:** `scripts/convert_labels_to_pose.py` converts the existing
  4-point labels into pose format under a **new** `data/sudoku_pose/` dataset
  (original dataset untouched).
- **Corner order:** geometrically canonicalize each label to **TL, TR, BR, BL**
  (sum/diff of coords) so the pose model learns stable keypoint slots.
- **Seed weights:** fine-tune from `models/yolov8s-pose.pt`.
- **Model picker:** two dropdowns (CNN checkpoints + pose weights) on solve;
  CNN-only on optimize.
- **Restructure:** `vision/` package (`common`, `classical`, `yolo`) + `registry/`
  package (model files, CNN list, pose list).
- **Dataset source:** `data/Sudoku-Detector.yolov8/` (4-point polygon labels).

## Label conversion

`scripts/convert_labels_to_pose.py`:
1. Read each `Sudoku-Detector.yolov8/{train,valid,test}/labels/*.txt`.
2. Parse `class` + 4 normalized `(x,y)` points.
3. **Canonicalize order** to TL,TR,BR,BL:
   - `s = x + y` → TL = argmin(s), BR = argmax(s)
   - `d = x - y` → TR = argmax(d), BL = argmin(d)
4. **Bbox** from the 4 points: `xmin..xmax, ymin..ymax` →
   `cx, cy, w, h` (normalized, clamped to [0,1]).
5. Write pose label line (kpt_shape `[4,3]`, visibility `2`):
   `class cx cy w h  x1 y1 2  x2 y2 2  x3 y3 2  x4 y4 2` (17 cols).
6. Copy/symlink images into `data/sudoku_pose/{train,valid,test}/images/` and
   write `data/sudoku_pose/data.yaml`:
   ```yaml
   path: .
   train: train/images
   val: valid/images
   test: test/images
   kpt_shape: [4, 3]
   flip_idx: [1, 0, 3, 2]   # horizontal flip swaps TL<->TR, BR<->BL
   names:
     0: sudoku
   ```
   `flip_idx` keeps left/right-mirror augmentation consistent with the corner
   order.

The script is idempotent (overwrites the output dataset) and prints per-split
counts.

## Architecture

### Backend package reorg

```
backend/app/core/
  vision/
    __init__.py     # re-exports public API (back-compat)
    common.py       # detector-agnostic helpers
    classical.py    # contour detector (relocated, unchanged logic)
    yolo.py         # pose corner detection + cell extraction
  registry/
    __init__.py
    model_files.py  # relocated from core/
    cnn_models.py   # list + cached-load CNN checkpoints
    yolo_models.py  # list + cached-load pose weights
  train_yolo.py     # ultralytics pose trainer (shared CLI + endpoint)
  model.py, solver.py, train.py, optimize_model.py, data_utils.py, report_utils.py
```

`vision/__init__.py` re-exports every symbol the routers import.
`registry/model_files.py` move updates imports in `main.py`,
`routers/{training,models,optimization}.py`.

### vision/common.py (detector-agnostic)

`resize_and_maintain_aspect_ratio`, `apply_grayscale_blur_and_threshold`,
`get_quadrilateral_points_in_order`, `perform_four_point_transform`,
`center_and_resize_digit`, `check_for_digit_in_cell_image`, `_bbox_iou`, `_nms`,
`_contour_to_quad`, `locate_cells_within_grid`, `sort_cells_into_grid`,
`build_grid_from_partial_cells`, `_clear_border_components`,
`slice_grid_into_cells`, `get_predicted_sudoku_grid_torch`,
`get_per_cell_predictions`, `plot_cell_images_in_grid`, `generate_solution_image`.

### vision/classical.py

`find_grid_contour_candidates`, `get_cells_classical` (renamed from
`get_valid_cells_from_image`; legacy alias kept in `__init__.py`).

### vision/yolo.py (pose)

```python
def load_yolo_model(weights_path: str, device: torch.device) -> "YOLO": ...

def get_grid_corners_yolo(
    img: np.ndarray, yolo_model: "YOLO", conf: float = 0.25
) -> np.ndarray:
    """Run pose inference, take the highest-confidence instance's 4 keypoints,
    order them TL,TR,BR,BL, return float32 (4,2). Raise ValueError if none."""

def get_cells_yolo(
    img: np.ndarray, yolo_model: "YOLO", grid_size: int = 576
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """corners -> square warp -> 81 cells. Same tuple shape as classical."""
```

**`get_grid_corners_yolo` logic (simpler than bbox approach):**
1. `yolo_model.predict(img, conf=conf, verbose=False)`.
2. From `results[0].keypoints`, pick the instance with the highest box conf.
   Extract its 4 `(x,y)` keypoints in pixel coords (`keypoints.xy`).
3. If no instance / fewer than 4 keypoints → `ValueError("YOLO-pose found no grid")`.
4. Order via `get_quadrilateral_points_in_order` (defensive — model is trained on
   canonical order, but ordering guarantees correctness). Return float32 (4,2).

**`get_cells_yolo`:** `M, grid_sq = perform_four_point_transform(img, corners,
size=grid_size)`; then `locate_cells_within_grid` → (81 → use; ≥54 →
`build_grid_from_partial_cells`; else `slice_grid_into_cells`). Returns
`(cells, M, grid_sq)`.

> The bbox→corner `refine_corners_within_bbox` from the detection design is
> **removed** — pose gives corners directly.

### Detector selection (unchanged from prior design)

`routers/inference._run_pipeline(img_bytes, cnn_model, device, detector,
yolo_model)`: `detector == "yolo"` and a pose model loaded → `get_cells_yolo`
with classical fallback on exception; else `get_cells_classical`. `/api/solve`
gains optional form fields `detector`, `cnn_model`, `yolo_model`; default detector
= `yolo` if pose model loaded else `classical`. `SolveResponse` gains
`detector_used`.

### Model storage convention

- CNN checkpoints: `models/*.pt` (excluding pose seed).
- Pose weights: `models/yolo/*.pt` + seed `models/yolov8s-pose.pt`. Pointer
  `models/yolo/latest_yolo.txt`.
- `GET /api/models/cnn`, `GET /api/models/yolo` →
  `[{id, filename, size_mb, created_at, is_default}]`.

### Pose training — CLI + in-app

`core/train_yolo.py: train_yolo(data_yaml, seed_weights, epochs, imgsz, batch,
device, models_dir, on_epoch=None) -> best_path`. Uses
`ultralytics.YOLO(seed_weights)` (task inferred = pose). Callback on
`on_fit_epoch_end` emits `{epoch, epochs, box_loss, pose_loss, map50}` (pose mAP
key `metrics/mAP50(P)`). Copies `best.pt` → `models/yolo/sudoku_pose_*.pt`,
writes pointer.

`scripts/train_yolo.py` (argparse): defaults `--data
data/sudoku_pose/data.yaml`, `--model models/yolov8s-pose.pt`, `--epochs 100`
(small dataset), `--imgsz 640`, `--batch 8`, `--device auto`. Device auto:
`cuda → mps → cpu`.

`GET /api/train/yolo/stream` (SSE): mirrors the CNN trainer pattern
(thread-pool, queue, sentinel, keepalive, single-flight guard). Emits per-epoch
pose metrics; on completion reloads `app.state.yolo_model`. Requires the pose
dataset to exist (`data/sudoku_pose/data.yaml`); if missing → emit an error event
telling the user to run the conversion script first.

Optimize endpoint gains optional `cnn_model` (CNN-only).

### App state / lifespan

Load default CNN as today. Resolve default pose weights: `YOLO_MODEL_PATH` env →
`models/yolo/latest_yolo.txt` → newest `models/yolo/*.pt` → none. Add
`app.state.yolo_model`, `yolo_model_id`, `cnn_cache`, `yolo_cache`.

### Schemas

Add `ModelEntry`, `ModelList`, `YoloTrainingConfig` (epochs, imgsz, batch,
model_seed). Extend `SolveResponse` with `detector_used`.

## Frontend

Skills: `frontend-design` + `ui-ux-pro-max`; `caveman-review` final pass. Preserve
the dark/cyan theme.

- `lib/api.ts`: `listCnnModels`, `listYoloModels`,
  `solveSudoku(file, {detector, cnnModel, yoloModel})`, `yoloTrainingStreamUrl`,
  `runOptimization(cnnModel?)`, `detector_used` plumbed through.
- New `components/ui/ModelSelect.tsx` — themed dropdown reused by solve + optimize.
- **Solve page:** Classical│YOLO toggle + CNN dropdown + pose-weights dropdown
  (shown when YOLO). Show `detector_used` badge.
- **Optimize page:** CNN dropdown.
- **Train page:** Digit-CNN │ Grid-Pose sub-tab. Pose form (epochs, imgsz, batch)
  + live box/pose-loss + mAP50 chart via SSE.

## Testing / verification

- `convert_labels_to_pose.py`: unit test on a synthetic 4-point label →
  asserts 17-col output, TL,TR,BR,BL order, bbox encloses all keypoints.
- `get_grid_corners_yolo`: unit test with a fake pose result object exposing
  `keypoints.xy` → returns ordered (4,2) corners; empty result raises.
- vision back-compat import test; registry listing tests (CNN excludes pose seed;
  pose list includes `yolo/` + seed).
- Backend import-smoke + uvicorn boot + curl `/api/models/*` and `/api/solve`.
- `scripts/convert_labels_to_pose.py` real run (tiny dataset) then
  `scripts/train_yolo.py --epochs 1 --imgsz 320 --batch 2 --device cpu` wiring
  smoke (best-weights copy + pointer).
- Frontend `npm run lint && npm run build`; Playwright specs with mocked
  `/api/models/*`.

## Risks

- **Tiny dataset** (18 train / 5 val / 2 test) → pose accuracy will be limited;
  rely on ultralytics augmentation, document that more data improves results.
  This is a wiring/feature deliverable, not an accuracy guarantee.
- `ultralytics` already in `requirements.txt` (8.4.71 installed) — no new dep.
- Pose SSE training is GPU/CPU heavy → single-flight guard.
- Corner-order canonicalization must match `flip_idx` `[1,0,3,2]`; both derived
  from TL,TR,BR,BL — covered by the conversion unit test.
- Back-compat: `vision` package must re-export every router import — covered by
  the compat test.

## Out of scope

- Replacing the backtracking solver or DigitCNN.
- Achieving a target pose mAP (dataset is tiny; only a 1-epoch wiring smoke).
- Auto-running label conversion at server start (explicit script step).
