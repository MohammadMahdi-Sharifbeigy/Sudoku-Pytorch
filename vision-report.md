# Sudoku-Pytorch — Full Technical Report

Complete technical reference for the project: every module, every class,
every function, every tunable parameter, and how they connect into the
end-to-end pipeline (image → grid extraction → digit recognition → solving
→ solution overlay), plus the model-training and model-optimization
sub-systems built around it.

This report supersedes the previous `vision.py`-only report. It now covers
the whole codebase: `src/` (core logic) and `app/` (Streamlit UI), plus the
CLI entry points.

---

## 1. Project Map

```
main.py                    Streamlit entry point — mode router
run_training_cli.py         CLI training runner (no Streamlit)

src/
  vision.py                 Grid detection, cell extraction, solution overlay
  model.py                  All CNN architectures + loss functions
  data_utils.py              Dataset loaders, augmentation pipelines
  train.py                   Epoch-level train/validate loops
  train_runner.py             High-level training orchestration (config → trained model)
  optimize_model.py           TorchScript / ONNX export + CPU benchmarking
  report_utils.py             Plain-text report generation (training & inference)
  solver.py                   Backtracking Sudoku solver

app/
  inference_page.py           Mode 1 — "Solve Sudoku" Streamlit page
  training_page.py            Mode 2 — "Model Training" Streamlit page
  optimization_page.py        Mode 3 — "Model Optimization" Streamlit page
  preprocess_config.py         Shared preprocessing sidebar + config state
  plot_utils.py                Dark-themed Matplotlib plotting helpers
  debug_utils.py               Debug-output dumper (writes to debug_outputs/)
  onnx_wrapper.py              Adapters giving ONNX sessions a PyTorch-like API
  cancel.py                    Thread-safe cooperative cancellation

tests/
  test_vision_a.py             Legacy test file — targets a removed module
                               (`vision_a.py`, deleted; see §8, Known Issues)
```

**Note on pipeline history.** Earlier versions of this project had three
interchangeable vision pipelines ("Original", "B", "C"). Pipelines B and C
(`src/vision_b.py`, `src/vision_c.py`) were removed in the commit
*"removing pipeline b and c of vision and adding more test samples"*.
`src/vision.py` (the former "Original" pipeline) is now the **only** vision
pipeline in the project. The pipeline `selectbox` in
`app/preprocess_config.py` still lists `["Original", "B", "C"]`, but
`app/inference_page.py` no longer branches on it — it always uses
`vision_get_cells()`, which calls `src.vision.get_valid_cells_from_image`.
Selecting "B" or "C" in that dropdown currently has no effect.

---

## 2. End-to-End Pipeline

```
                              ┌─────────────────────────┐
                              │   Uploaded photo (JPG)   │
                              └────────────┬─────────────┘
                                           ▼
                     resize_and_maintain_aspect_ratio(new_width=1000)
                                           ▼
                      sharpen_image()  (optional, Laplacian ± NLM denoise)
                                           ▼
              ┌──────────────────────────────────────────────────┐
              │        find_grid_contour_candidates(img)          │
              │  multi-threshold adaptive binarize → contours →   │
              │  NMS → perspective-warp each surviving candidate  │
              └───────────────────────┬────────────────────────┘
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │       get_valid_cells_from_image  (orchestrator)  │
              │                                                    │
              │  1. Contour path:  locate_cells_within_grid        │
              │     exactly 81 cells found? → sort_cells_into_grid │
              │                                                    │
              │  2. Partial recovery: ≥54 cells found?              │
              │     → build_grid_from_partial_cells (KMeans)       │
              │                                                    │
              │  3. Slice fallback: square-warp + slice_grid_into_ │
              │     cells (deterministic 9×9, no contour needed)   │
              └───────────────────────┬────────────────────────┘
                                       ▼
                     list of 81 cell dicts (img, contains_digit,
                     grid_row, grid_col, x_centroid, y_centroid)
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │   get_predicted_sudoku_grid_torch(model, cells)    │
              │   batch all "contains_digit" crops through the CNN │
              │   (or MultiTask / Unified / ONNX variant)          │
              └───────────────────────┬────────────────────────┘
                                       ▼
                              9×9 int grid (0 = empty)
                                       ▼
                      SudokuSolver(board).solve()  (bitmask backtracking)
                                       ▼
                    generate_solution_image(...)  — render solved digits
                    on a blank board, un-warp with the inverse perspective
                    matrix, paint in red onto the original photo
                                       ▼
                              Annotated solved image
```

Each **cell dict** produced by the extraction stage has this shape:

| key | type | meaning |
|---|---|---|
| `img` | `uint8` 28×28 | centred binary digit crop (all-zero if empty) |
| `contains_digit` | `bool` | whether a digit was detected in the cell |
| `grid_row` | `int` 0–8 | row position in the 9×9 layout |
| `grid_col` | `int` 0–8 | column position in the 9×9 layout |
| `x_centroid` | `int` | x of cell centre, in warped-grid pixels |
| `y_centroid` | `int` | y of cell centre, in warped-grid pixels |

---

## 3. Training Pipeline (all model types)

```
TrainingConfig (dataclass: model/purpose/dataset/hyperparameters)
        │
        ▼
run_training(config, device, on_epoch, stop_event)      [src/train_runner.py]
        │
        ├─ model_choice == "EfficientNetDigit" ─────► _run_efficientnet
        │        two-phase: freeze_backbone() → head-only epochs
        │                   unfreeze_last_blocks(n=3) → fine-tune epochs
        │
        ├─ model_choice == "MultiTaskCNN" ──────────► _run_multitask
        │        shared backbone, two heads (digit 0-9, language Persian/English)
        │
        ├─ model_choice ∈ {"UnifiedCNN — MobileNetV3","UnifiedCNN — ShuffleNetV2"}
        │        ────────────────────────────────────► _run_unified
        │        single 20-class head (0-9 English ∪ 10-19 Persian)
        │
        ├─ purpose == "Persian" ────────────────────► _run_lang_specific(is_persian=True)
        ├─ purpose == "English" ────────────────────► _run_singletask
        └─ purpose == "Multi" + "Separate models" ──► _run_lang_specific ×2 (Persian, English)
        │
        ▼
 per-epoch: train_epoch()/train_epoch_multitask() → validate()/validate_multitask()
 scheduler.step()  → history dict updated → on_epoch(update) callback fired
        │
        ▼
 best-val-loss checkpoint saved to models/best_model_<variant>.pt
        │
        ▼
 save_training_report(...) / save_training_report_multitask(...)  [src/report_utils.py]
 writes models/training_report_<variant>.txt (epoch history, confusion matrix,
 per-class accuracy, classification report)
```

The Streamlit **Model Training** page (`app/training_page.py`) runs this
exact call graph inside a background `threading.Thread`, polling progress
through `queue.Queue` objects so the UI stays responsive and the **Stop**
button can interrupt training via `threading.Event` (`app/cancel.py`). The
CLI (`run_training_cli.py`) calls the same `run_training()` function
synchronously with a `tqdm` progress bar instead.

---

## 4. Class Reference

### 4.1 `src/vision.py` — no classes (pure functions, see §5)

### 4.2 `src/model.py` — CNN architectures & losses

#### `FocalLoss(nn.Module)`
Focal loss for class-imbalanced digit classification (empty-cell class is
much more frequent than any single digit class).
```
FL(p_t) = alpha * (1 - p_t)^gamma * CE(p_t)
```
| Constructor arg | Default | Effect of higher | Effect of lower |
|---|---|---|---|
| `alpha` | `0.25` | Scales overall loss magnitude up (stronger gradient signal) | Weaker gradient signal |
| `gamma` | `2.0` | Down-weights easy (already-confident) examples more aggressively, focusing training on hard/misclassified cells | Behaves closer to plain cross-entropy; less focus on hard examples |
| `reduction` | `"mean"` | — (`"sum"` scales loss with batch size; a raw per-sample tensor is returned for `"none"`) | — |

#### `DigitCNN(nn.Module)`
Primary architecture. 3 convolutional blocks (BatchNorm + ReLU, doubling
channels 32→64→128) each ending in `MaxPool2d`/`GlobalAvgPool`, followed by
a `128 → 256 → num_classes` classifier head. ~250K parameters.
- Block 1: `1→32` channels, `28×28 → 14×14`, `Dropout2d(p=0.1)`.
- Block 2: `32→64` channels, `14×14 → 7×7`, `Dropout2d(p=0.1)`.
- Block 3: `64→128` channels, `AdaptiveAvgPool2d((1,1))` → `128×1×1`.
- Classifier: `Linear(128,256)` → `BatchNorm1d` → `ReLU` → `Dropout(p=0.4)` → `Linear(256, num_classes)`.
- `num_classes` default `10` (digits 0–9, 0 = empty).

#### `LegacyDigitCNN(nn.Module)`
Original 2-conv architecture (`conv1`/`conv2`/`fc` keys), kept **only** so
older checkpoints can still be loaded/exported. Not used for new training.
`Dropout(p=0.5)` before the single `Linear(64*7*7, num_classes)` head.

#### `MultiTaskDigitCNN(nn.Module)`
Same 3-block backbone as `DigitCNN`, but with **two** heads sharing the
128-d pooled feature:
- `digit_head`: `128 → 256 → BN → ReLU → Dropout(0.4) → num_digit_classes` (default 10).
- `lang_head`: `128 → 64 → ReLU → num_lang_classes` (default 2: Persian=0, English=1).
- `forward()` returns `(digit_logits, lang_logits)`.
- Empty cells must be labeled `lang_target = -1` so the language loss ignores them.

#### `MultiTaskFocalLoss(nn.Module)`
Combines both heads' losses:
```
total = digit_weight * FocalLoss(digit_logits, digit_target)
      + lang_weight  * CrossEntropyLoss(lang_logits, lang_target, ignore_index=-1)
```
| Constructor arg | Default | Effect of higher | Effect of lower |
|---|---|---|---|
| `digit_weight` | `0.7` | Digit accuracy prioritized more in the combined gradient | Language accuracy relatively prioritized more |
| `lang_weight` | `0.3` | Language-head training signal strengthened | Language head trains more slowly |
| `focal_alpha` | `0.25` | See `FocalLoss` | |
| `focal_gamma` | `2.0` | See `FocalLoss` | |
| `lang_class_weights` | computed via `compute_lang_class_weights` | Corrects for Persian/English sample imbalance (Hoda is the minority source) | Uniform weighting; imbalance not corrected |
| `lang_ignore_index` | `-1` | — sentinel for "no language" (empty cells) | |
Returns `(total_loss_tensor, digit_loss_float, lang_loss_float)`.

#### `UnifiedDigitCNN(nn.Module)`
20-class model that predicts digit **and** script in a single softmax head,
built on a torchvision backbone with its stem adapted from 3-channel to
1-channel input:
- `backbone="mobilenet_v3_small"` (~2.5M params) or `"shufflenet_v2_x0_5"` (~0.35M params).
- `pretrained: bool` — load ImageNet weights for the (adapted) backbone.
- `num_classes = UNIFIED_NUM_CLASSES = 20` (0=Eng-empty, 1–9=Eng digits, 10=Per-empty [unused], 11–19=Per digits).
- `param_count()` → `(total_params, trainable_params)`.

`decode_unified_class(cls_idx)` maps a winning class index back to
`(digit 0-9, lang_name_or_None)`; classes 0 and 10 both decode to digit 0
with no language (both represent "empty").

#### `EfficientNetDigitCNN(nn.Module)`
EfficientNet-B0 (torchvision) adapted for grayscale 28×28 input — the input
is repeated to 3 channels inside `forward()` so pretrained stem weights
apply correctly, then resized externally to 224×224 by the data pipeline.
- Classifier head replaces the stock 1280→1000 head with:
  `Dropout(0.3) → Linear(1280,512) → SiLU → Dropout(0.2) → Linear(512,256) → SiLU → Dropout(0.1) → Linear(256,num_classes)`.
- `freeze_backbone()` — Phase 1: freezes all `features` params, unfreezes `classifier`, freezes BatchNorm running stats (`_freeze_batchnorm()`).
- `unfreeze_last_blocks(n=3)` — Phase 2: unfreezes the last `n` feature blocks (default 3) in addition to the head; BatchNorm stays frozen (correct behavior for pretrained transfer learning — re-estimating BN stats on a small dataset destabilizes training).
- `param_count()` → `(total_params, trainable_params)`.

Free functions: `probe_digit_cnn_architecture(path)` (peeks at a
checkpoint's `state_dict` keys to distinguish `DigitCNN` from
`LegacyDigitCNN` without instantiating either) and
`load_digit_cnn_checkpoint(path, device, num_classes=10)` (tries `DigitCNN`
first, falls back to `LegacyDigitCNN` on a `RuntimeError` key mismatch;
returns `(model, architecture_name)`).

### 4.3 `src/data_utils.py` — datasets, augmentation, loaders

#### `RandomShadow`
Custom torchvision-style transform simulating a shadow cast across part of
a digit cell (real photos often have shadows from book bindings or uneven
lighting). Darkens a random half (vertical or horizontal strip, chosen 50/50)
by multiplying pixel values by a random factor.
| Constructor arg | Default | Effect of higher | Effect of lower |
|---|---|---|---|
| `intensity_range` | `(0.35, 0.70)` | Values closer to 1.0 = lighter/subtler shadow | Values closer to 0.0 = darker, more aggressive shadow |
| `p` | `0.25` | Shadow applied more often (stronger regularization) | Shadow applied less often |

#### `AugmentedDataset(Dataset)`
Wraps precomputed `(x, y)` tensors (`x`: float32 `[N,1,H,W]` in `[0,1]`,
`y`: long `[N]`) and applies a `transform` on `__getitem__` (per-sample, at
load time — so augmentation differs every epoch).

#### `MultitaskAugmentedDataset(Dataset)`
Same idea but three-output: `(x, digit_y, lang_y)` for the multi-task loader.

### 4.4 `src/solver.py`

#### `SudokuSolver`
Bitmask-constraint backtracking solver with an MRV (Minimum-Remaining-Values)
heuristic.
- `__init__(board)` — builds `rows`/`cols`/`boxes` bitmasks (bit `k` set = digit `k` used in that row/col/box). Detects conflicting **givens** immediately (e.g. a vision misread that put the same digit twice in a row) and sets `self.valid = False` — `solve()` then fails fast with zero search.
- `_box_idx(row, col)` — maps `(row, col)` to one of the 9 (`box_size = sqrt(n) = 3`) 3×3 sub-grids.
- `_candidates(row, col)` — legal digits for an empty cell given current constraints.
- `_find_best_empty()` — MRV: scans all empty cells, returns the one with the **fewest legal candidates** (ties broken by scan order); short-circuits immediately on 0 candidates (dead end — prune early) or exactly 1 candidate (forced move — no need to keep scanning). This dramatically reduces backtracking versus naive first-empty-cell selection.
- `_place`/`_remove` — set/clear a digit and update the three bitmasks.
- `solve()` — recursive backtracking over MRV-selected cells; returns `bool`.
- `is_valid_number(board, number, position)` — standalone legality check (bitmask lookup).

### 4.5 `app/cancel.py`

#### `RunCancelled(Exception)`
Marker exception used to unwind a running pipeline when cancellation is
requested. Caught in each Streamlit page's `except RunCancelled` block.
(`src/train_runner.py` defines its **own** identically-named `RunCancelled`
class, raised internally by `_check_stop()` — the two classes are distinct
but structurally interchangeable; each page catches the one relevant to the
code path it drives.)

### 4.6 `app/onnx_wrapper.py`

#### `OnnxInferenceSession`
Wraps an `onnxruntime.InferenceSession` (forced to
`providers=['CPUExecutionProvider']` — **ONNX inference always runs on CPU**
regardless of the app's selected device) so it exposes a PyTorch-model-like
`__call__`/`eval()`/`to()` interface. `n_outputs` (1 or 2) tells callers
whether this is a standard or multi-head export. `infer_multitask(x)`
returns both output tensors for dual-head models.

#### `DigitOnlyModelWrapper`
Wraps a `MultiTaskDigitCNN`, discarding the language-head output so the
model can be fed directly into `get_predicted_sudoku_grid_torch` (which
expects a single-logits-tensor-returning callable).

#### `UnifiedDigitOnlyWrapper`
Wraps a `UnifiedDigitCNN` (20-class), folding English+Persian logits into a
10-class digit-only view: `logit[0] = max(class_0, class_10)`,
`logit[d] = class_d + class_{d+10}` for `d` in 1–9. Used only for the vision
pipeline's digit-filling step; the raw (unwrapped) model is used elsewhere
in `inference_page.py` to also recover the predicted language.

---

## 5. Function Reference — `src/vision.py`

### Basic image helpers

#### `resize_and_maintain_aspect_ratio(input_image, new_width)`
Resize keeping aspect ratio (`cv2.INTER_AREA`, good for downscaling).
`new_width` — target width in px; height is scaled to match.

#### `sharpen_image(img, use_nlm=False, center=5)`
Sharpens to recover detail lost to camera/scanner blur, via a Laplacian
kernel `[[0,-1,0],[-1,center,-1],[0,-1,0]]`.
- `center` — kernel centre weight. Higher = stronger sharpening (risk of ringing artifacts on already-sharp images); lower = milder sharpening.
- `use_nlm=False` — Laplacian only, instant.
- `use_nlm=True` — runs `fastNlMeansDenoising(Colored)` first (heavier, ~1–3s extra), then sharpens — better on noisy photos.

#### `apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7, blur_k=3)`
Blur → grayscale → adaptive threshold → invert, so foreground (digits/lines)
becomes white on black.
- `method` — `"mean"` → `ADAPTIVE_THRESH_MEAN_C`; anything else → `ADAPTIVE_THRESH_GAUSSIAN_C`.
- `blocksize` — odd neighbourhood size for the local threshold. Larger = smoother, ignores fine texture (risks missing thin/faint strokes); smaller = more locally adaptive, picks up thin strokes but is more sensitive to noise.
- `c` — constant subtracted from the local mean. Higher = stricter threshold (less white kept); lower = looser threshold (more white/noise kept).
- `blur_k` — Gaussian blur kernel size (odd int ≥1; `1` skips blur). Higher = smoother/less noisy threshold input but can blur thin digit strokes.

#### `get_quadrilateral_points_in_order(approx_arr)`
Orders 4 corner points as **[top-left, top-right, bottom-right, bottom-left]**
using distance-from-origin heuristics — required because
`getPerspectiveTransform` needs source/destination corners in matching order.

#### `perform_four_point_transform(input_img, src_corners, pad=10, size=None)`
Warps a quadrilateral region to a flat top-down view.
- `pad` — inner margin (px), only used in **aspect-preserving** mode. Higher = more blank border around the warped grid.
- `size`:
  - `None` → aspect-preserving warp to `max_w × max_h` (used for the contour path).
  - `int` → **square** warp to `size × size`, so the grid can be cut into a perfect 9×9 lattice (used for the slice fallback).
- Returns `(M, warped)` — `M` (3×3 perspective matrix) is needed later to un-warp the solution overlay.

#### `center_and_resize_digit(cell_img, digit_scale=20)`
Isolates the largest blob (the digit), scales its long side to `digit_scale`
px, and centres it on a **28×28** canvas — matching the MNIST-style input the
CNN expects.
- `digit_scale` — controls zoom. Lower = more padding (safer for odd/thin strokes, e.g. some Persian digits); higher = fills more of the canvas (closer to raw MNIST scale, risk of clipping strokes at the edges). Practical range 10–26.

#### `check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False)`
Decides if a cell holds a digit by measuring the largest contour's area as a
percentage of cell area.
- `area_threshold` — minimum % of cell area the largest blob must cover. Lower = more sensitive (catches thin/faint strokes but risks counting line scraps as digits); higher = fewer false positives but may miss faint digits.
- `apply_border` — if `True`, blanks a 7% frame around the cell before measuring, to remove grid-line bleed at the edges. Used by the contour path; the slice path relies on `_clear_border_components` instead and passes `False`.
- Returns `(has_digit: bool, processed_cell)`.

### NMS utilities

#### `_bbox_iou(b1, b2)`
Intersection-over-Union of two `(x, y, w, h)` boxes, `0.0`–`1.0`.

#### `_nms(candidates, iou_threshold=0.3)`
Greedy Non-Maximum Suppression: keeps the highest-scoring box, drops
anything overlapping it beyond the threshold, repeats.
- `iou_threshold` — overlap above which the lower-scored box is suppressed. Higher = more permissive (keeps more near-duplicate detections); lower = more aggressive merging (may drop legitimate close-together detections).

### Quad helper

#### `_contour_to_quad(contour)`
Reduces any contour to 4 corners via convex-hull extreme points
(TL = min x+y, BR = max x+y, TR = max x−y, BL = min x−y). Lets 5–14-sided
polygons (wavy/hand-drawn outlines) still yield a usable quadrilateral.

### Grid boundary detection

#### `find_grid_contour_candidates(img, threshold_combos=None, blur_k=3, should_stop=None)`
Locates the outer grid boundary across multiple threshold parameter
combinations, dedupes with NMS, and warps each surviving candidate.
- `threshold_combos` — list of `(blocksize, C)` pairs to sweep. Default `DEFAULT_GRID_THRESHOLD_COMBOS = [(31,6),(21,5),(11,6),(61,10),(41,8),(81,12)]` — a spread from fine to coarse so at least one setting catches the boundary regardless of lighting. More combos = more robustness to unusual lighting but proportionally slower (each combo re-runs full contour detection).
- Applies a **3×3 morphological close** after thresholding to bridge gaps in thin/broken (e.g. hand-drawn) grid lines so the outer boundary forms one closed contour.
- Tries both `RETR_EXTERNAL` and `RETR_LIST` retrieval modes.
- Keeps contours whose area is **4%–97%** of the image (`0.04 < area_ratio < 0.97`) — rejects tiny noise and full-frame borders.
- Accepts 4-corner contours directly; reduces 5–14-corner ones via `_contour_to_quad`.
- Scores each candidate by `squareness × area_ratio`; NMS at `iou_threshold=0.5`.
- Returns `(M_matrices, warped_images, contour_list)` or `(None, None, None)` if nothing survived.

### Cell extraction

#### `locate_cells_within_grid(grid_img, threshold_combos=None, blur_k=3, area_threshold=4, erode_enabled=True, erode_kernel_size=3, erode_iterations=1, digit_scale=20, should_stop=None)`
Finds up to 81 individual cells inside one warped grid by detecting each
cell's own 4-corner contour.
- `threshold_combos` — default `DEFAULT_CELL_THRESHOLD_COMBOS = [(91,7),(51,5),(71,9),(31,4),(111,10),(41,6)]`.
- Keeps contours with area fraction **0.3%–2.2%** of the grid (`0.003 < frac < 0.022`) — the expected size band for 1 cell out of 81.
- Requires a clean 4-corner `approxPolyDP` result.
- Composite score = `squareness × area_regularity × fill`, where `area_regularity` peaks when the contour's area matches `grid_area / 81`.
- NMS at `iou_threshold=0.3`; iterates all `threshold_combos`, keeps whichever pass yields the most cells, and stops early once 81 are found.
- `erode_enabled`/`erode_kernel_size`/`erode_iterations` — morphological erosion applied to digit-containing cells after border-line removal. Larger kernel / more iterations = more aggressive line/noise stripping but risks eroding thin digit strokes; disable entirely with `erode_enabled=False`.
- `digit_scale` — passed through to `center_and_resize_digit`.
- Returns a list of cell dicts (centroids from image moments; **no** `grid_row`/`grid_col` yet — that's `sort_cells_into_grid`'s job).
- **Known limitation:** on a slightly skewed printed grid this typically finds 75–78 of 81 cells, not all — this drove the partial-recovery tier described in §7.

#### `sort_cells_into_grid(cells)`
Assigns `grid_row`/`grid_col` by dividing centroids by an estimated cell
pitch, then sorts row-major. **Only correct when exactly 81 cells are
present** (the divide-by-pitch estimate assumes a complete, evenly-spaced grid).

#### `_clear_border_components(binary)`
Drops white connected components that touch the cell border (leftover grid
lines), keeping the centred digit. Uses `connectedComponentsWithStats`. This
is the slice path's line-removal strategy — gentler than blanking a fixed
border frame, since it only removes components that actually touch the edge.

#### `slice_grid_into_cells(grid_img, n=9, pad_frac=0.02, area_threshold=4, erode_enabled=True, erode_kernel_size=2, erode_iterations=3, digit_scale=20, should_stop=None)`
**Deterministic fallback.** Cuts a square warped grid into an exact `n × n`
lattice with no contour detection — survives faint/wavy/hand-drawn lines
that defeat `locate_cells_within_grid`.
- `grid_img` — must be a **square** warped grid (from `perform_four_point_transform(..., size=...)`).
- `n` — grid dimension, 9.
- `pad_frac` — fraction trimmed inward from each cell before digit detection, to shed surrounding grid lines. Too high clips real digits at the edges; too low lets grid lines bleed into the digit-detection area. Historically `0.08`, lowered to `0.02` once `_clear_border_components` took over primary line removal (see §7, Problem A).
- Each cell: trim → threshold (fixed `blocksize=31, c=7`) → `_clear_border_components` → `check_for_digit_in_cell_image(area_threshold=4, apply_border=False)` → `center_and_resize_digit`.
- Returns exactly `n*n` cell dicts in row-major order (centroids = geometric slice centres).

#### `build_grid_from_partial_cells(cells, n=9)`
Reconstructs a full `n × n` grid from a **partial** set of detected cells by
clustering their centroids — the core of the "Problem B" fix (§7).
- `cells` — unordered cells from `locate_cells_within_grid` (may be `<81`).
- Runs **1-D KMeans** (`n_clusters=n`) separately on x-centroids (→ columns) and y-centroids (→ rows); cluster centres are sorted and mapped to a 0–8 rank each.
- Places each cell in its `(row, col)` slot; on collision, **prefers the cell that contains a digit** — a real digit never loses its slot to a spuriously-detected blank.
- Fills any empty slot with a blank cell whose centroid is the cluster-centre intersection (so `generate_solution_image` can still place an answer there).
- Requires `len(cells) >= n` (KMeans needs at least `n` samples per axis). Returns exactly `n*n` row-major cell dicts, or `None` if that minimum isn't met.

#### `get_valid_cells_from_image(img, grid_size=576, grid_threshold_combos=None, cell_threshold_combos=None, blur_k=3, area_threshold=4, erode_enabled=True, contour_erode_kernel_size=3, contour_erode_iterations=1, slice_erode_kernel_size=2, slice_erode_iterations=3, digit_scale=20, should_stop=None)`
**Orchestrator.** Full pipeline: detect grid boundary → warp → extract 81
cells, via a three-tier strategy:
1. **Contour path** — for each grid candidate, run `locate_cells_within_grid`; return immediately (with `sort_cells_into_grid`) if exactly 81 cells are found.
2. **Partial-grid recovery** — if the best candidate found **≥54** cells (two-thirds of 81 — the threshold empirically found reliable for KMeans reconstruction), `build_grid_from_partial_cells` reconstructs the full 9×9 layout, preserving every digit the contour detector already found.
3. **Slice fallback** — otherwise, square-warp the best candidate (`grid_size` px side, fresh perspective matrix `M`) and `slice_grid_into_cells`.
- `grid_size` — side length (px) of the square warp used only in the fallback tier. Higher = more pixel detail per cell (finer digit strokes preserved) at the cost of more compute; lower = faster but coarser.
- Raises an `Exception` only if **no** grid boundary is found at all.
- Returns `(cells, M, board_image)`.

### Prediction & visualisation

#### `get_predicted_sudoku_grid_torch(model, cells, device, should_stop=None)`
Batches every `contains_digit` crop through the CNN in one forward pass,
scatters predictions back into an `81`-length array (0 = empty at every
non-digit position), reshapes to `(9,9)`.

#### `plot_cell_images_in_grid(cells)`
Stitches the 81 cell crops into a single 9×9 montage `matplotlib` figure —
used by the "Grid Extraction Preview" panel and debug-output dumps.

#### `generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix)`
Renders solved digits onto a blank white board at each empty cell's
centroid (`cv2.putText`, red channel via a mask), un-warps that overlay with
`M_matrix` via `WARP_INVERSE_MAP`, and paints the digits onto the original
photo wherever the un-warped overlay is dark. Returns the annotated image.

---

## 6. Function Reference — Remaining `src/` Modules

### `src/train.py` — epoch-level loops

| Function | Purpose |
|---|---|
| `train_epoch(model, train_loader, criterion, optimizer, device, should_stop=None)` | One training epoch (single-task). Shows a `tqdm` bar only when stdout is a TTY (`_is_tty()` — avoids log spam under Streamlit). Returns `(avg_loss, accuracy_pct)`. |
| `validate(model, val_loader, criterion, device, should_stop=None)` | One no-grad validation pass. Returns `(avg_loss, accuracy_pct)`. |
| `collect_predictions(model, loader, device, should_stop=None)` | Runs inference over a full loader, returns `(all_true_labels, all_pred_labels)` for building a classification report / confusion matrix. |
| `train_epoch_multitask(...)` | Multi-task variant; loader yields `(data, digit_target, lang_target)`. Returns `(total_loss, digit_acc_pct, lang_acc_pct, digit_loss, lang_loss)`. Language accuracy is computed only over non-empty cells (`lang_target != -1`). |
| `validate_multitask(...)` | Multi-task validation counterpart. |
| `collect_predictions_multitask(...)` | Returns `(digit_true, digit_pred, lang_true, lang_pred)` — `lang_*` contain only non-empty-cell samples. |
| `run_twophase_training(model, train_loader, val_loader, criterion, device, epochs_phase1=6, epochs_phase2=14, lr_phase1=1e-3, lr_phase2=1e-5, weight_decay=1e-4, unfreeze_blocks=3, should_stop=None, on_epoch_end=None)` | Drives `EfficientNetDigitCNN`'s two-phase training. Phase 1: `model.freeze_backbone()`, trains `epochs_phase1` epochs at `lr_phase1`. Phase 2: `model.unfreeze_last_blocks(unfreeze_blocks)`, trains `epochs_phase2` epochs at `lr_phase2` (typically ~100× smaller — fine-tuning shouldn't move pretrained weights far). BatchNorm is re-frozen every epoch (`model._freeze_batchnorm()`) because `model.train()` would otherwise re-enable running-stat updates. Uses `CosineAnnealingLR` internally for each phase, `T_max = phase length`. Tracks and restores the best-val-loss `state_dict` across both phases. Returns `(history, best_val_loss)`. |

Parameter effects for `run_twophase_training`:

| Parameter | Default | Higher | Lower |
|---|---|---|---|
| `epochs_phase1` | 6 | Longer head-only warmup before touching backbone weights | Shorter warmup, backbone unfrozen sooner |
| `epochs_phase2` | 14 | Longer fine-tuning of unfrozen blocks + head | Shorter fine-tuning |
| `lr_phase1` | 1e-3 | Faster head convergence, risk of head instability | Slower head convergence |
| `lr_phase2` | 1e-5 | More aggressive backbone updates (risk of destroying pretrained features) | Gentler backbone updates (safer but slower adaptation) |
| `unfreeze_blocks` | 3 | More of the backbone becomes trainable in phase 2 (more capacity to adapt, more overfitting risk on small datasets) | Less of the backbone adapts (safer, less capacity) |

### `src/train_runner.py` — high-level orchestration

#### `TrainingConfig` (dataclass)
All fields and their defaults are documented in the cross-cutting parameter
table in §9. This is the single source of truth for every training
hyperparameter in the project — both the CLI (`run_training_cli.py`) and the
Streamlit training page construct one of these and pass it to `run_training`.

#### `RunCancelled(Exception)`
Raised internally when `stop_event.is_set()` — distinct class from, but
functionally mirrors, `app.cancel.RunCancelled`.

#### Private helpers
- `_check_stop(stop_event)` — raises `RunCancelled` if the event is set.
- `_make_optimizer(model, lr, wd)` — always `AdamW`.
- `_make_scheduler(optimizer, schedule, epochs, train_loader, cosine_t0, cosine_t_mult, cosine_eta_min)` — dispatches on `schedule` string to one of `CosineAnnealingWarmRestarts`, `CosineAnnealing`, `OneCycleLR`, or (default) `ReduceLROnPlateau(factor=0.5, patience=3)`.
- `_scheduler_step(scheduler, val_loss, schedule)` — `OneCycleLR` steps per-batch internally (no-op here); `CosineAnnealingWarmRestarts`/other cosine variants step once per epoch unconditionally; `ReduceLROnPlateau` steps on `val_loss`.
- `_map_dataset_to_loader_key(dataset_choice)` — normalizes several historical UI label strings (including legacy pre-refactor labels, kept for backward compatibility with old session state) down to one of `mnist_only`/`mnist_fonts`/`all`/`mnist_hoda`/`persian`.
- `_train_loop(...)` — the generic single-task epoch loop shared by `_run_singletask`, `_run_lang_specific`, `_run_unified`: runs `epochs` epochs of `train_epoch`/`validate`, tracks best-val-loss checkpoint (deep-copied `state_dict`), fires `on_epoch` with a structured update dict after every epoch, saves the best checkpoint to `save_path` at the end.

#### Router
`run_training(config, device, on_epoch=None, stop_event=None) -> dict` —
dispatches on `config.model_choice` / `config.purpose` / `config.multi_mode`
to one of `_run_singletask`, `_run_lang_specific` (×1 or ×2 for "Separate
models"), `_run_multitask`, `_run_unified`, `_run_efficientnet`. Each
sub-runner builds its own `DataLoader`s (via `src/data_utils.py`), model,
loss, optimizer/scheduler, runs training, evaluates on the held-out test
split, writes a report via `src/report_utils.py`, and returns a result dict
containing the trained model, test metrics, and history — consumed by both
the CLI printer and the Streamlit results page.

### `src/optimize_model.py` — export & benchmarking

#### `run_optimization_and_benchmark(base_model, model_path, cpu_device, output_prefix='models/best_model', should_stop=None)`
Exports a single-output model to **TorchScript** (`torch.jit.trace` +
`.save()`) and **ONNX** (`torch.onnx.export`, `opset_version=11`,
dynamic batch axis), then benchmarks CPU inference latency for all three
formats (PyTorch, TorchScript, ONNX via `onnxruntime` if installed) using a
`(1,1,28,28)` dummy tensor, `1000` timed iterations after a `10`-iteration
warmup. Reports size (MB), ms/cell, and extrapolated ms/81-cells (a full
Sudoku grid). ONNX benchmarking is skipped gracefully (`"N/A"`) if
`onnxruntime` isn't installed.

#### `run_optimization_and_benchmark_multitask(base_model, model_path, cpu_device, output_prefix='models/best_model_multitask', should_stop=None)`
Same idea for dual-output models (`MultiTaskDigitCNN` / `UnifiedDigitCNN`
wrapped appropriately). ONNX export uses `output_names=['digit_output',
'lang_output']`. After export, **verifies** ONNX output matches PyTorch
output within `atol=1e-5` (`np.allclose`) for both heads before
benchmarking — returns a `verification` dict (`digit_match`, `lang_match`,
`error`) alongside the results list, surfaced as an error in the UI if
either match fails (a real correctness signal, not just a benchmark).

Both functions hardcode `iters=1000` and a `10`-iteration warmup; these are
not exposed as tunable parameters.

### `src/report_utils.py` — plain-text report generation

| Function | Purpose |
|---|---|
| `save_training_report(history, test_loss, test_acc, y_true, y_pred, dataset_mode, epochs, learning_rate, batch_size, best_val_loss, model, output_path=...)` | Single-task report: header, config, full epoch-by-epoch history table, test summary, `sklearn.classification_report`, text-rendered confusion matrix, per-class accuracy table. |
| `save_training_report_multitask(...)` | Multi-task report: adds a Persian/English sample-count imbalance audit section (flags `>60/40` splits), digit accuracy **broken down by true language** (catches the case where aggregate accuracy looks fine but the Persian-minority class is failing), plus separate classification reports/confusion matrices for the digit and language heads. |
| `save_inference_report(image_name, cells, per_cell_info, grid_array, solved_board, output_path=...)` | Per-image inference report: cell-by-cell table (row, col, has-digit, predicted digit, confidence, and language+confidence if a multitask/unified model was used), the raw predicted grid, the solved grid (or a "could not be solved" notice), and a low-confidence-cell summary (`<50%` confidence) to help diagnose misreads. |
| `_format_grid(grid)` | Shared helper — renders a 9×9 grid as a bordered text block with `.` for empty cells and `---+---+---`-style box separators. |

---

## 7. Function Reference — `app/` (Streamlit UI)

### `app/inference_page.py` — Mode 1: Solve Sudoku

- `vision_get_cells(**kwargs)` — thin wrapper forwarding to `get_valid_cells_from_image`, applying the same defaults documented in §5.
- `_get_per_cell_predictions(model, cells, device, is_multitask=False, is_unified=False)` — runs the classifier over all 81 cells; empty cells short-circuit to `{'label':0,'confidence':1.0,'has_digit':False}` without a forward pass; branches on unified (20-class softmax + `decode_unified_class`) vs. multitask (independent digit/lang softmax) vs. standard (single softmax) model shape.
- `_render_debug_steps(...)` — visual, display-only breakdown of the CV pipeline (sharpening → grayscale/blur → threshold preview → all-combos preview) inside an `st.expander`.
- `_render_cell_grid(cells, per_cell, is_multitask, is_unified)` — renders a 9×9 grid of 60px thumbnails with predicted label, confidence band (High ≥80% / Medium 50–80% / Low <50%), and language badge if applicable.
- `render_inference_page(device)` — top-level entry point called from `main.py`. Wires together: sidebar model selection (Persian/multi-model/ONNX/manual-file toggles), `render_preprocess_sidebar()` from `app/preprocess_config.py`, two tabs ("Solve Sudoku", "Preprocessing Debug"), the full extract→predict→solve→overlay flow with `RunCancelled`/generic-exception handling, and debug-output dumping via `app/debug_utils.py`.

### `app/training_page.py` — Mode 2: Model Training

State machine in `st.session_state['_train_state']`:
`idle → running → done` (or `cancelled`/`error`, looping back to `idle`).

- `render_training_page(device)` — entry point; dispatches on state.
- `_render_config_form(device)` — the hyperparameter form (see §9 for the full widget table); on submit builds a `TrainingConfig` and calls `_launch`.
- `_launch(config, device)` — clears the cancel event, creates `epoch_q`/`result_q` queues, starts a daemon `threading.Thread` running `run_training(...)`, sets session state to `"running"`.
- `_render_running()` — drains `epoch_q` into history, shows a **Stop Training** button, renders live `st.progress`/`st.metric`/`st.line_chart`, polls `result_q` non-blockingly, sleeps `0.4s` between reruns (the polling interval).
- `_render_done(device)` — final results: training curves, test-set metrics (branches on `model_type`: multitask/separate/single-task), confusion matrix (`_draw_cm` local helper calling `app/plot_utils.py`), download button for the `.txt` report.
- `_render_reset_button()` / `_reset()` — clears all training session-state keys, returns to `idle`.

### `app/optimization_page.py` — Mode 3: Model Optimization

- `render_optimization_page(device)` — sole function. Discovers all trained `.pt` checkpoints under `models/` (fixed candidates for single/multitask/unified + a glob for `best_model_efficientnet_*.pt`), lets the user pick one and (for Unified models) the matching backbone, then runs `run_optimization_and_benchmark[_multitask]` from `src/optimize_model.py` and renders the results table, load-snippet, and (for dual-head models) the ONNX/PyTorch parity verification.

### `app/preprocess_config.py` — Preprocessing sidebar & config state

Two-tier session-state design: a **draft** (live widget edits,
`st.session_state.draft_*`) and an **applied** config
(`st.session_state.preprocess_applied`), with an Apply/Reset workflow so
image processing only reacts to explicitly committed settings.

- `format_threshold_combos(combos)` / `threshold_combo_label(index, combo)` — serialize a `(blocksize, C)` list to text / a human label.
- `parse_threshold_combos(raw_text, fallback)` — parses back, validating `blocksize >= 3` and odd, `C >= 0`; falls back to `fallback` if zero valid combos parsed (defends against an empty/corrupted text field disabling grid detection entirely).
- `threshold_combos_to_frame(combos)` — combos as a 2-column `pandas.DataFrame`.
- `PREPROCESS_DEFAULTS` — the full default config dict (values listed in §9).
- `applied_preprocess_config()` — lazily initializes and range-clamps the applied config (defensive: resets any out-of-range/corrupted session-state value back to default rather than crashing).
- `init_preprocess_draft(config)` — initializes `draft_*` keys from `config`, with the same clamping applied to any stale pre-existing draft values.
- `current_preprocess_draft()` — reads `draft_*` keys back into a plain dict; forces `use_nlm=False` if `enable_sharpen=False` (NLM is a sub-option of sharpening).
- `render_preprocess_sidebar()` — the full sidebar UI (see §9 for the widget table); returns `(active: dict, dirty: bool)`.

### `app/plot_utils.py` — dark-themed Matplotlib helpers

- `_style_ax(ax, title, ylabel='', xlabel='Epoch')` — applies the shared dark theme (facecolor, white text, gray spines, subtle grid) to one `Axes`.
- `plot_confusion_matrix(y_true, y_pred, class_names)` — row-normalized heatmap with `"{count}\n{pct}%"` annotations; text flips to dark when the normalized cell value exceeds `0.55` (contrast against the light colormap).
- `plot_lang_confusion_matrix(lang_true, lang_pred)` — fixed 2×2 Persian/English variant.
- `plot_multitask_training_history(history)` — 2×3 subplot dashboard: total/digit/language loss, digit/language accuracy, plus a stacked "Loss Composition" area chart that visualizes the fixed `0.7×digit + 0.3×lang` weighting baked into `MultiTaskFocalLoss`'s defaults.
- `plot_lr_history(lr_history)` — log-scale LR-per-epoch line plot.
- `plot_time_per_epoch(time_history)` — dual-axis bar (per-epoch seconds) + line (cumulative time).
- `plot_singletask_summary(history, lr_history, time_history)` — 2×2 dashboard: Loss, Accuracy, LR (log scale), Time/epoch.
- `make_run_dir(model_tag, epochs, batch_size, lr, weight_decay)` — creates and returns a timestamped `runs/<tag>_ep..._bs..._lr..._wd..._<timestamp>/` directory, encoding hyperparameters into the folder name.
- `save_fig(fig, run_dir, filename)` — saves a figure at `dpi=120`, closes it, returns the path.
- `save_run_metadata(run_dir, meta)` — dumps `meta` as indented JSON.

### `app/debug_utils.py`

- `save_debug_outputs(image_name, img_rgb, cells_selected, board_selected, pipeline_name, error=None, out_root="debug_outputs")` — dumps the original image, warped grid, a full cell mosaic, and every individual cell crop (named `r{row}c{col}_{D|E}.png`) plus a `summary.json` to a timestamped directory under `debug_outputs/`, for offline troubleshooting independent of the Streamlit session.

### `app/onnx_wrapper.py` and `app/cancel.py`

See the Class Reference in §4.5–4.6 — both modules are class-only (no
standalone functions besides the small `get_stop_event`/`request_cancel`/
`clear_cancel`/`raise_if_cancelled`/`is_cancelled`/`render_stop_button`
helpers in `cancel.py`, described inline there).

---

## 8. Known Issues / Out-of-Scope Notes

1. **`tests/test_vision_a.py` targets a removed module.** It imports from a
   top-level `vision_a` module (`preprocess_clahe`, `detect_grid_canny`,
   `refine_corners_hough`, `remove_grid_lines_hough`,
   `slice_cells_from_lines`) that no longer exists anywhere in the repository
   (only stale `.pyc` cache files remain). This file was not removed in the
   same commit that deleted `src/vision_b.py`/`src/vision_c.py` and will
   currently fail to collect under `pytest`. It documents the shape of an
   older, no-longer-present pipeline variant and should be either deleted or
   rewritten against `src/vision.py`'s current function names.

2. **Stale "B"/"C" pipeline selector.** `app/preprocess_config.py` still
   offers a `Pipeline` selectbox with options `["Original", "B", "C"]` and
   descriptive help text referencing CLAHE/Hough-based alternative
   pipelines. Since `src/vision_b.py` and `src/vision_c.py` were deleted,
   `app/inference_page.py` no longer reads this value — every run uses the
   Original (`src/vision.py`) pipeline regardless of the selection. The
   selectbox should either be removed or clearly marked as inactive.

3. **CNN misclassification can still make an otherwise-perfectly-extracted
   puzzle unsolvable.** This is a model-accuracy limitation, not a
   vision-pipeline bug: two visually similar digits (historically, `9`↔`8`
   and `6`↔`8` on some printed fonts) can be predicted with moderate
   confidence (60–75%) but land in the same row/column/box, making the
   puzzle's constraints unsatisfiable. `SudokuSolver.valid` will be `False`
   for such boards. Mitigations live entirely in `src/train.py`/
   `src/data_utils.py` (retraining, font augmentation, confidence-threshold
   flags surfaced via `save_inference_report`'s low-confidence-cell list) —
   not in `src/vision.py`.

---

## 9. Cross-Cutting Parameter Reference

Every numeric/enum knob in the project, in one place.

### 9.1 Vision pipeline parameters (`src/vision.py`, surfaced via `app/preprocess_config.py`)

| Parameter | Where | Default | Range (UI) | Higher value effect | Lower value effect |
|---|---|---|---|---|---|
| Sharpening enabled | sidebar toggle | On | on/off | Sharper edges before detection | Raw (softer) image |
| NLM denoise first | sidebar toggle (sub-option) | Off | on/off | Cleaner denoised base, much slower | Faster, keeps existing noise |
| Sharpen kernel centre (`sharpen_center`) | slider | 5 | 3–13, step 2 | Stronger sharpening / edge contrast, more noise amplification, risk of ringing | Gentler sharpening |
| Gaussian blur kernel (`blur_k`) | select | 3 | {1,3,5,7} | Smoother, less noisy input; can blur thin strokes | Sharper/noisier; better fine detail |
| Adaptive-threshold blocksize (`blocksize`/`thresh_bs`) | slider / per-combo editor | grid: from `DEFAULT_GRID_THRESHOLD_COMBOS[0]`; cell: fixed per-combo list | sidebar 11–111; combo editor 3–301 (must be odd) | Smoother threshold, less locally sensitive, ignores fine texture | More locally adaptive, more sensitive to noise |
| Adaptive-threshold constant (`c`/`thresh_c`) | slider / per-combo editor | varies per combo | sidebar 1–25; combo editor 0–100 | Darker/stricter threshold, fewer pixels kept | Lighter threshold, more pixels/noise kept |
| Min contour area % (`area_threshold`) | slider | 3.5 (sidebar) / 4 (function default) | 0.5–10.0, step 0.5 | Stricter — fewer false positives, may miss faint/thin digits | Looser — catches thin strokes (e.g. Persian), more false positives |
| Grid-threshold combo list | text editor / add-remove rows | `[(31,6),(21,5),(11,6),(61,10),(41,8),(81,12)]` | 3–301 (odd) / 0–100 per row | More combos = more robust to odd lighting, N× slower | Fewer combos = faster, may fail on unusual images |
| Grid loop mode | toggle | "Use selected combo only" | selected-only / try-all | Tries every combo, more robust, N× slower | Fast, deterministic, may fail on tricky images |
| Erosion enabled | toggle | On | on/off | — | — |
| Contour-path erosion kernel | select_slider | 3 | {1,2,3,4,5} | More aggressive border/noise removal, risk of eroding thin strokes | Gentler erosion |
| Contour-path erosion iterations | slider | 1 | 0–5 | Cumulative stronger erosion | Less erosion |
| Slice-fallback erosion kernel | select_slider | 2 | {1,2,3,4,5} | Stronger erosion on the fallback path | Gentler |
| Slice-fallback erosion iterations | slider | 3 | 0–5 | Stronger cumulative erosion on fallback path | Gentler |
| `pad_frac` (slice fallback) | function default only | 0.02 | — | Clips more of the grid line but risks clipping real digits | Lets more grid-line bleed into digit detection |
| Digit scale (px in 28×28 canvas) | slider | 22 (UI default) / 20 (`vision_get_cells`/function default) | 10–26 | Tighter crop, fills more of canvas, risk of clipping strokes at edges | More padding, safer for odd/thin digits, less signal filling the canvas |
| NMS IoU threshold — grid candidates | function-internal | 0.5 | — | More permissive, keeps near-duplicate grid candidates | More aggressive merging |
| NMS IoU threshold — cell candidates | function-internal | 0.3 | — | More permissive, keeps near-duplicate cells | More aggressive merging |
| Partial-recovery threshold (`≥54` cells) | function-internal | 54 (⅔ of 81) | — | (Raising it would fall back to slicing more often, discarding more contour-detected digits) | (Lowering it risks feeding KMeans too few points to reconstruct a reliable 9×9 layout) |
| Square-warp `grid_size` (slice fallback) | function default | 576 px | — | More pixel detail per cell, slower | Coarser detail, faster |

### 9.2 Training hyperparameters (`TrainingConfig`, `src/train_runner.py`, surfaced via `app/training_page.py` and `run_training_cli.py`)

| Parameter | CLI flag | UI widget | Default | Range | Higher value effect | Lower value effect |
|---|---|---|---|---|---|---|
| `epochs` | `--epochs` | number_input | 20 | 1–100 | Longer training, more convergence chance, more overfitting risk | Faster, may underfit |
| `learning_rate` | `--lr` | number_input | 1e-3 | unbounded | Faster initial convergence, risk of instability/divergence | Slower, more stable, risk of stalling |
| `batch_size` | `--batch` | selectbox | 128 | {32,64,128,256} | Smoother gradients, more memory, may need higher LR | Noisier gradients, less memory, sometimes better generalization |
| `weight_decay` | `--wd` | number_input | 1e-4 | unbounded | Stronger L2 regularization, less overfitting, risk of underfitting | Weaker regularization |
| `lr_schedule` | `--schedule` | selectbox | `ReduceLROnPlateau` | 4 options (see below) | — qualitative — | — qualitative — |
| `cosine_t0` | `--cosine-t0` | number_input | 10 | 1–100 | Longer first cycle (WarmRestarts) / slower full decay (CosineAnnealing) | Shorter cycle / faster decay |
| `cosine_t_mult` | `--cosine-mult` | number_input | 1 | 1–4 | Each restart cycle grows longer (2× per restart) | Equal-length cycles |
| `cosine_eta_min` | `--eta-min` | number_input | 1e-6 | unbounded | Higher LR floor (less fine-tuning at cycle end) | Lower floor, finer late-cycle updates |
| `eff_pretrained` | `--no-pretrained` (inverts) | toggle | True | on/off | Faster convergence, less data needed (ImageNet init) | Random init, needs more data/epochs |
| `eff_phase1_epochs` | `--phase1` | number_input | 6 | 1–30 | Longer head-only warmup | Shorter warmup |
| `eff_phase2_epochs` | `--phase2` | number_input | 14 | 1–50 | Longer full-network fine-tuning | Shorter fine-tuning |
| `unified_backbone` | `--backbone` | selectbox | `mobilenet_v3_small` | `mobilenet_v3_small` / `shufflenet_v2_x0_5` | MobileNetV3: ~2.5M params, higher capacity/accuracy | ShuffleNetV2: ~0.35M params, faster/smaller, lower capacity |
| `unified_pretrained` | (`--no-pretrained`) | toggle | False | on/off | Faster convergence | Random init |
| `aug_preset` | `--aug` | selectbox | `full` (UI) / `none` (CLI default) | `none` / `light` / `full` | `full`: strongest regularization, slower convergence, risk of over-distorting digits; `light`: affine only; `none`: no augmentation, fastest, most overfitting risk | (inverse) |
| `multi_mode` | `--multi-mode` | radio | `Unified model (20-class)` | `Unified model` / `Separate models` | Separate: 2 independent models, doubles training time, may reach higher per-language accuracy | Unified: one model, faster to train/deploy, shares capacity across languages |
| `data_path` | `--data` | text_input | `"data"` | any valid path | — | — |
| `model_choice` | `--model` | selectbox | `DigitCNN` | `DigitCNN`/`LegacyDigitCNN`/`MultiTaskCNN`/`UnifiedCNN — MobileNetV3`/`UnifiedCNN — ShuffleNetV2`/`EfficientNetDigit` | — architecture choice — | — |
| `purpose` | `--purpose` | selectbox | `English` | `English`/`Persian`/`Multi` | — dataset/model routing — | — |
| `dataset_choice` | `--dataset` | selectbox | `MNIST + Fonts` | varies by purpose | More source datasets merged = more training data/diversity | Fewer sources = faster but less diverse |

**LR schedule options** (qualitative choice, not a magnitude):
- `ReduceLROnPlateau` — halves LR (`factor=0.5`) after `patience=3` epochs without validation-loss improvement. Simple, reactive, no schedule to tune in advance.
- `CosineAnnealingWarmRestarts` — cosine decay with periodic restarts every `cosine_t0` epochs (growing by `cosine_t_mult` each restart); floor at `cosine_eta_min`. Good for escaping local minima.
- `CosineAnnealing` — single smooth cosine decay over `cosine_t0` epochs (used as `T_max`) down to `cosine_eta_min`. Predictable, no restarts.
- `OneCycleLR` — warms up then decays LR within one run (`pct_start=0.3`, peak `max_lr = base_lr × 10`), steps per-batch. Often fastest convergence for a fixed epoch budget.

Fixed (non-tunable) training internals worth noting:
- `FocalLoss(alpha=0.25, gamma=2.0)` used for all single-task and multi-task digit losses.
- `MultiTaskFocalLoss(digit_weight=0.7, lang_weight=0.3)` — fixed weighting between the two heads' contributions to the combined loss (also visualized as the "Loss Composition" chart in `plot_multitask_training_history`).
- Optimizer is always `AdamW`.
- `generate_empty_cells(num_samples=3000)` — synthetic all-black "empty cell" samples added to every dataset mix; increasing this ratio would bias the model toward predicting "empty" more readily, decreasing it would do the opposite (not currently exposed as a UI/CLI parameter).
- Data-loader `num_workers=0` always (Windows `spawn` multiprocessing can't pickle the dataset classes from inside the background training thread — documented in `data_utils.py`'s `_make_loader`).

### 9.3 Augmentation transform parameters (`src/data_utils.py`, `build_train_transform`)

| Parameter | Default | Higher value effect | Lower value effect |
|---|---|---|---|
| `brightness` (ColorJitter) | 0.0 (off) | Wider random brightness swings — more robust to lighting variance, risk of unrealistic samples | Narrower swings / disabled at 0 |
| `contrast` (ColorJitter) | 0.0 (off) | Wider random contrast swings | Narrower / disabled at 0 |
| `shadow_p` (`RandomShadow` probability) | 0.0 (off) | Shadow simulation applied more often | Less often / disabled at 0 |
| `shadow_intensity_min`/`max` | 0.35 / 0.70 | Values closer to 1.0 = subtler shadow | Values closer to 0.0 = darker, harsher shadow |
| `RandomAffine(degrees=8, translate=(0.08,0.08), scale=(0.88,1.12), shear=6)` | fixed in "full"/"light" presets | More rotation/translation/scale/shear jitter = more geometric robustness, risk of illegible/ambiguous digits | Less jitter = closer to clean input, less robust |
| `RandomPerspective(distortion_scale=0.18, p=0.35)` | fixed ("full" preset only) | More perspective distortion applied more often | Less distortion |
| `GaussianBlur(kernel_size=3, sigma=(0.1,1.5))` | fixed ("full" preset only) | Blurrier training samples, more robust to blur, risk of losing fine strokes | Sharper training samples |
| `RandomAdjustSharpness(sharpness_factor=2.0, p=0.3)` | fixed ("full" preset only) | Stronger occasional sharpening | Milder |
| `RandomErasing(p=0.25, scale=(0.02,0.12), ratio=(0.3,3.3))` | fixed ("full" preset only) | Larger/more frequent random erased patches — more robust to occlusion, risk of erasing the whole digit on small crops | Smaller/less frequent erasing |

**Augmentation presets** (`aug_preset` / `build_train_transform_preset`):
- `"none"` — `ToTensor` + normalize only. Fastest, most overfitting-prone, but the "original winning config" per an inline code comment (i.e. empirically best for some runs — augmentation isn't strictly better on every dataset).
- `"light"` — `RandomAffine` only, no blur/erasing. A middle ground, safer for small/simple models.
- `"full"` — every transform above. Strongest regularization, current default in the UI.

`build_efficientnet_transform(augment=True)` — separate, milder pipeline for
224×224 3-channel input: `RandomAffine(degrees=10, translate=(0.05,0.05),
scale=(0.90,1.10), shear=5)`, `ColorJitter(brightness=0.2, contrast=0.2)`,
`RandomPerspective(distortion_scale=0.12, p=0.25)`. Milder than the 28×28
pipeline because EfficientNet's pretrained features are more sensitive to
aggressive distortion.

Normalization constants (not meant to be tuned — must match what a
checkpoint was trained with):
- `MNIST_MEAN=0.1307`, `MNIST_STD=0.3081` — all 28×28 grayscale models.
- `EFFICIENTNET_MEAN=[0.485,0.456,0.406]`, `EFFICIENTNET_STD=[0.229,0.224,0.225]` — ImageNet standard, required for the pretrained EfficientNet/Unified backbones.

### 9.4 Model architecture dimensions (not runtime-tunable, listed for reference)

| Model | Key dims | Param count (approx) |
|---|---|---|
| `DigitCNN` | 1→32→64→128 conv channels, 128→256→num_classes head | ~250K |
| `LegacyDigitCNN` | 1→32→64 conv channels, 64×7×7→num_classes head | smaller, legacy only |
| `MultiTaskDigitCNN` | same backbone as `DigitCNN`; digit head 128→256→10, lang head 128→64→2 | ~260K |
| `UnifiedDigitCNN` (mobilenet_v3_small) | adapted 1-channel stem, 20-class head | ~2.5M |
| `UnifiedDigitCNN` (shufflenet_v2_x0_5) | adapted 1-channel stem, 20-class head | ~0.35M |
| `EfficientNetDigitCNN` | EfficientNet-B0 backbone (3-channel via repeat), head 1280→512→256→num_classes | ~5.3M |

### 9.5 Optimization/export parameters (`src/optimize_model.py`)

| Parameter | Value | Notes |
|---|---|---|
| Dummy input shape | `(1,1,28,28)` | Fixed — matches all digit-model inputs |
| Warmup iterations | 10 | Not exposed as a parameter |
| Timed iterations | 1000 | Not exposed as a parameter |
| ONNX `opset_version` | 11 | Fixed |
| ONNX dynamic axes | batch dimension only | Input/output spatial dims are fixed at export time |
| ONNX↔PyTorch parity tolerance | `atol=1e-5` (`np.allclose`) | Multi-head export verification only |

---

## 10. File-Size / Structure Summary

| File | Lines | Role |
|---|---|---|
| `src/vision.py` | 668 | Grid detection, cell extraction, solution overlay |
| `src/data_utils.py` | 861 | Dataset loaders, augmentation |
| `src/train_runner.py` | 591 | High-level training orchestration |
| `src/model.py` | 414 | CNN architectures & losses |
| `src/report_utils.py` | 460 | Text report generation |
| `src/train.py` | 274 | Epoch-level train/validate loops |
| `src/optimize_model.py` | 182 | Export & benchmarking |
| `src/solver.py` | 89 | Sudoku backtracking solver |
| `app/inference_page.py` | 617 | "Solve Sudoku" Streamlit page |
| `app/training_page.py` | 507 | "Model Training" Streamlit page |
| `app/preprocess_config.py` | 372 | Preprocessing sidebar & config state |
| `app/plot_utils.py` | 251 | Dark-themed plotting helpers |
| `app/optimization_page.py` | 148 | "Model Optimization" Streamlit page |
| `app/onnx_wrapper.py` | 79 | ONNX/multi-head model adapters |
| `app/debug_utils.py` | 62 | Debug-output dumper |
| `app/cancel.py` | 42 | Cooperative cancellation |
| `run_training_cli.py` | 172 | CLI training entry point |
| `main.py` | 43 | Streamlit entry point / mode router |
