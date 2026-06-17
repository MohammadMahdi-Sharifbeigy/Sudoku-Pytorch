# `vision.py` — Function Reference & Problem-Solving Report

Computer-vision front end of the Sudoku solver. Takes a raw photo, finds the
grid, warps it flat, cuts it into 81 cells, and hands clean 28×28 digit
crops to the CNN. This document describes every function and parameter, then
walks through the bugs we hit and how each was fixed.

---

## 1. Pipeline at a glance

```
raw photo
   │
   ▼
find_grid_contour_candidates ──► list of warped grid candidates + transforms
   │
   ▼
get_valid_cells_from_image  (orchestrator)
   │   ├─ 1. locate_cells_within_grid      (contour cells)  ── exactly 81? → return
   │   ├─ 2. build_grid_from_partial_cells  (KMeans recovery) ── ≥54 cells? → return
   │   └─ 3. slice_grid_into_cells          (deterministic 9×9 fallback)
   ▼
list of 81 cell dicts  →  get_predicted_sudoku_grid_torch  →  9×9 int grid
   │
   ▼
solver → generate_solution_image (overlay answer back on the photo)
```

Each **cell dict** has this shape:

| key              | type        | meaning                                            |
| ---------------- | ----------- | -------------------------------------------------- |
| `img`            | `uint8` 28×28 | centred binary digit crop (zeros if empty)       |
| `contains_digit` | `bool`      | whether a digit was detected in the cell           |
| `grid_row`       | `int` 0–8   | row position in the 9×9 layout                     |
| `grid_col`       | `int` 0–8   | column position in the 9×9 layout                  |
| `x_centroid`     | `int`       | x of cell centre, in warped-grid pixels            |
| `y_centroid`     | `int`       | y of cell centre, in warped-grid pixels            |

---

## 2. Function reference

### Basic image helpers

#### `resize_and_maintain_aspect_ratio(input_image, new_width)`
Resize keeping aspect ratio.
- `input_image` — BGR/RGB or grayscale array.
- `new_width` — target width in px; height scaled to match.
- Returns the resized image (`cv2.INTER_AREA`, good for downscaling).

#### `apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7)`
Blur → grayscale → **adaptive threshold** → invert, so foreground (digits/lines)
becomes white on black.
- `img` — 3-channel or single-channel input (auto-detected).
- `method` — `"mean"` → `ADAPTIVE_THRESH_MEAN_C`; anything else → `ADAPTIVE_THRESH_GAUSSIAN_C`.
- `blocksize` — odd neighbourhood size for the local threshold. Larger = smoother, ignores fine texture; smaller = picks up thin strokes but more noise.
- `c` — constant subtracted from the local mean. Higher = stricter (less white).
- Returns an inverted binary image (`uint8`, 0/255).

#### `get_quadrilateral_points_in_order(approx_arr)`
Order 4 corner points as **[top-left, top-right, bottom-right, bottom-left]**
using distance-from-origin heuristics. Required because `getPerspectiveTransform`
needs source and destination corners in matching order.
- `approx_arr` — array of 4 points, shape `(4,1,2)` or `(4,2)`.
- Returns the 4 points reordered TL, TR, BR, BL.

#### `perform_four_point_transform(input_img, src_corners, pad=10, size=None)`
Warp a quadrilateral region to a flat top-down view.
- `input_img` — image to warp.
- `src_corners` — 4 source corners (any order; reordered internally).
- `pad` — inner margin (px) in **aspect-preserving** mode only.
- `size` — **key parameter**:
  - `None` → aspect-preserving warp to `max_w × max_h` (original behaviour; used for the contour path).
  - `int`  → **square** warp to `size × size`, so the grid can be cut into a perfect 9×9 lattice (used for the slice fallback).
- Returns `(M, warped)` where `M` is the 3×3 perspective matrix (needed later to un-warp the solution).

#### `center_and_resize_digit(cell_img)`
Take a binary cell, isolate the largest blob (the digit), scale its long side
to 20 px, and centre it on a **28×28** canvas — matching MNIST-style input the
CNN expects.
- `cell_img` — binary cell crop.
- Returns a 28×28 `uint8` image.

#### `check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False)`
Decide if a cell actually holds a digit by measuring the largest contour's area.
- `img` — binary cell crop.
- `area_threshold` — min % of cell area the largest blob must cover to count as a digit. Lower = more sensitive (risks counting line scraps); higher = misses faint digits.
- `apply_border` — if `True`, blanks a 7% frame around the cell before measuring, to kill grid-line bleed. Used by the contour path; the slice path uses `_clear_border_components` instead and passes `False`.
- Returns `(has_digit: bool, processed_cell)`.

### NMS utilities

#### `_bbox_iou(b1, b2)`
Intersection-over-Union of two `(x, y, w, h)` boxes. Returns `0.0`–`1.0`.

#### `_nms(candidates, iou_threshold=0.3)`
Greedy **Non-Maximum Suppression**: keep the highest-scoring box, drop anything
overlapping it beyond the threshold, repeat.
- `candidates` — list of `(score, bbox, payload)`.
- `iou_threshold` — overlap above which the lower-scored box is suppressed. Higher = more permissive (keeps near-duplicates); lower = more aggressive merging.
- Returns the kept subset.

### Quad helper

#### `_contour_to_quad(contour)`
Reduce any contour to 4 corners via **convex-hull extreme points** (TL = min x+y,
BR = max x+y, etc.). Lets 5–14-sided polygons (wavy/hand-drawn outlines) still
yield a usable quadrilateral.

### Grid boundary detection

#### `find_grid_contour_candidates(img)`
Locate the outer grid boundary across **multiple threshold settings**, dedupe with
NMS, and warp each surviving candidate.
- Sweeps `(blocksize, c)` ∈ `[(41,8),(21,5),(61,10),(31,6),(11,3),(81,12)]` — a spread from fine to coarse so at least one setting catches the boundary regardless of lighting.
- Applies a **3×3 morphological close** after thresholding to bridge gaps in thin/broken hand-drawn lines so the outer boundary closes into one contour.
- Tries both `RETR_EXTERNAL` and `RETR_LIST` retrieval modes.
- Keeps contours whose area is **4%–97%** of the image (`0.04 < area_ratio < 0.97`) — rejects tiny noise and full-frame borders.
- Accepts 4-corner contours directly; reduces 5–14-corner ones via `_contour_to_quad`.
- Scores each by `squareness × area_ratio`; NMS at `iou_threshold=0.5`.
- Returns `(M_matrices, warped_images, contour_list)` or `(None, None, None)`.

### Cell extraction

#### `locate_cells_within_grid(grid_img)`
Find the 81 individual cells inside one warped grid by detecting each cell's own
4-corner contour.
- Sweeps `(blocksize, c)` ∈ `[(91,7),(51,5),(71,9),(31,4),(111,10),(41,6)]`.
- Keeps contours with area fraction **0.3%–2.2%** of the grid (`0.003 < frac < 0.022`) — the expected size band for one cell out of 81.
- Requires a 4-corner `approxPolyDP` (a clean square).
- Composite score = `squareness × area_regularity × fill`, where `area_regularity` peaks when the contour matches `grid_area/81`.
- NMS at `iou_threshold=0.3`; keeps the `(blocksize, c)` pass yielding the most cells; stops early at 81.
- Returns a list of cell dicts (centroids from image moments; **no** `grid_row`/`grid_col` yet).
- **Limitation that drove the fix:** on a slightly skewed printed grid it typically finds 75–78 of 81 — not all — so the result needs reconstruction, not rejection.

#### `sort_cells_into_grid(cells)`
Assign `grid_row`/`grid_col` by dividing centroids by an estimated cell pitch,
then sort row-major. **Only correct when exactly 81 cells are present** (the
divide-by-pitch estimate assumes a full grid).

#### `_clear_border_components(binary)`
Drop white connected components that **touch the cell border** (leftover grid
lines), keeping the centred digit. Uses `connectedComponentsWithStats`. This is
the slice path's line-removal strategy and is gentler than blanking a fixed
border frame.

#### `slice_grid_into_cells(grid_img, n=9, pad_frac=0.02)`
**Deterministic fallback.** Cut a square grid into an exact `n × n` lattice with
no contour detection — survives faint/wavy/hand-drawn lines.
- `grid_img` — **square** warped grid (from `perform_four_point_transform(..., size=...)`).
- `n` — grid dimension (9).
- `pad_frac` — fraction trimmed inward from each cell before digit detection, to shed surrounding grid lines. Too high clips real digits; too low lets lines bleed in. Lowered from `0.08` → `0.02` once `_clear_border_components` took over line removal.
- Each cell: trim → threshold (`blocksize=31, c=7`) → `_clear_border_components` → `check_for_digit_in_cell_image(area_threshold=4, apply_border=False)` → `center_and_resize_digit`.
- Returns exactly `n*n` cell dicts in row-major order (centroids = slice centres).

#### `build_grid_from_partial_cells(cells, n=9)` *(new — core of the fix)*
Reconstruct a full `n × n` grid from a **partial** set of detected cells by
clustering their centroids.
- `cells` — unordered cells from `locate_cells_within_grid` (may be <81).
- `n` — grid dimension (9).
- Runs **1-D KMeans** (`n` clusters) separately on x-centroids (→ columns) and
  y-centroids (→ rows). Cluster centres are sorted to map each label to a 0–8 rank.
- Places each cell in its `(row, col)` slot; on collision, **prefers the cell that contains a digit** (so a real digit never loses its slot to a blank).
- Fills any empty slot with a blank cell whose centroid is the cluster-centre
  intersection (so `generate_solution_image` can still place an answer there).
- Returns exactly `n*n` row-major cell dicts, or `None` if `len(cells) < n`.

#### `get_valid_cells_from_image(img, grid_size=576)` *(orchestrator — rewired)*
Full extraction with a three-tier strategy:
1. **Contour path** — for each grid candidate run `locate_cells_within_grid`; if exactly 81 cells, `sort_cells_into_grid` and return.
2. **Partial-grid recovery** — if the best candidate has **≥54** cells (two-thirds of 81), `build_grid_from_partial_cells` reconstructs the 9×9 and returns. Preserves every digit the contour detector found.
3. **Slice fallback** — otherwise square-warp the best candidate (fresh `M` via `size=grid_size`) and `slice_grid_into_cells`.
- `grid_size` — side length (px) of the square warp used only in the fallback.
- Raises only if **no** grid boundary is found at all.
- Returns `(cells, M, board_image)`.

### Prediction & visualisation

#### `get_predicted_sudoku_grid_torch(model, cells, device)`
Batch the `contains_digit` crops through the CNN, scatter predictions back into
a 9×9 int array (0 = empty). Returns the `(9,9)` grid.

#### `plot_cell_images_in_grid(cells)`
Stitch the 81 cell crops into a single 9×9 montage `matplotlib` figure — used by
the Streamlit "Cell Extraction" panel and for debugging.

#### `generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix)`
Render solved digits onto a blank board at each empty cell's centroid, un-warp
that overlay with `M_matrix` (inverse perspective), and paint the digits in red
onto the original photo. Returns the annotated image.

---

## 3. Our road to solving the problems

### Problem A — hand-drawn grid `m3.jpg` extracted poorly
**Symptom.** Thin, wavy, faint hand-drawn lines with no thick outer border.
Per-cell contour detection (`locate_cells_within_grid`) needs each cell to be a
clean 4-corner square; wavy/broken lines defeat the `approxPolyDP` 4-corner test,
so far fewer than 81 cells were found and the old code **raised an exception** —
all-or-nothing.

**Steps taken.**
1. Added a **square-warp mode** (`size=` in `perform_four_point_transform`) so a
   grid can be cut into an exact lattice.
2. Added `slice_grid_into_cells` — a deterministic 9×9 cut that ignores per-cell
   contours entirely.
3. Added a **3×3 morphological close** in `find_grid_contour_candidates` to bridge
   broken hand-drawn lines so the outer boundary closes.
4. First slice attempt produced **68 "digits"** (real ≈28) — wavy line scraps were
   bleeding into sliced cells. Added `_clear_border_components` to drop
   border-touching components (lines) while keeping centred digits, and switched
   the slice path to `apply_border=False`. False positives cleared.

### Problem B — printed grids `16/17/4` lost digit cells, solved the *wrong* puzzle
**Symptom.** `16-extraction.png` and the `17.jpg` inference report showed only
**6 detected digits** out of ~26 — the grid was clean but most digits vanished,
so the solver filled in and solved a *different* puzzle.

**Root cause.** On a slightly skewed printed grid, `locate_cells_within_grid`
found **76 of 81** cells (with all 26 digits). The old orchestrator discarded any
result that wasn't *exactly* 81 and fell through to the **slice fallback** — but a
small warp skew misaligned the deterministic lattice, clipping digits onto cell
borders. Net: 26 → ~6 digits.

**Diagnosis path.**
- Traced `17.jpg`: contour path = **76 cells / 26 digits**; final pipeline output =
  **10 digits** → confirmed the loss happened in the *slice* path, not detection.
- Dumped the square warp + slice montage: the grid was fine, but digits landed
  off-centre / on borders and got trimmed.
- Conclusion: the detector was already succeeding; the orchestrator was **throwing
  away good data**.

**Fix.** Added `build_grid_from_partial_cells` + a middle recovery tier in
`get_valid_cells_from_image`:
- KMeans-cluster the detected centroids into 9 row-bands and 9 column-bands,
  place each detected cell in its slot, fill gaps as empty.
- Keeps **every** digit the contour detector found; only genuinely missing cells
  become blanks.
- Slice fallback now only fires when detection is truly poor (<54 cells).

**Results (digits detected, was → now):**

| image | before | after | notes                                   |
| ----- | ------ | ----- | --------------------------------------- |
| 16    | 6      | **26**, solvable ✓ | printed; was solving wrong puzzle |
| 17    | 6      | **26**, solvable ✓ | printed; was solving wrong puzzle |
| 3     | —      | **26**, solvable ✓ | printed                           |
| m3    | 68 (false) | **24** | hand-drawn; false positives gone      |
| 4     | 6      | **26 extracted ✓** | extraction perfect; see below     |

### Problem C — `main.py` Streamlit deprecation
**Symptom.**
```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
```
**Fix.** Replaced all 9 occurrences of `use_container_width=True` with
`width='stretch'`.

---

## 4. Known remaining issue (out of scope of extraction)

**Image `4.jpg` is extracted perfectly (all 26 digits in the right cells) but is
reported unsolvable.** Cause is a **CNN misclassification**, not vision:
- cell (1,1) is a **9** but predicted **8** (conf 0.66)
- cell (1,6) is a **6** but predicted **8** (conf 0.72)

Two 8s in the same row → no valid solution. This is a model-accuracy problem on
that printed font, fixable by retraining / font augmentation — separate from the
cell-extraction pipeline documented here.

---

## 5. Summary of changes made

| area | change |
| ---- | ------ |
| `perform_four_point_transform` | added `size=` square-warp mode |
| `find_grid_contour_candidates` | added 3×3 morphological close for broken lines |
| `_clear_border_components` | **new** — removes border-touching line scraps |
| `slice_grid_into_cells` | **new** — deterministic 9×9 fallback; `pad_frac` 0.08→0.02 |
| `build_grid_from_partial_cells` | **new** — KMeans reconstruction of partial grids |
| `get_valid_cells_from_image` | rewired to 3-tier strategy (contour → recovery → slice); no longer raises on <81 cells |
| `import` | added `from sklearn.cluster import KMeans` |
| `main.py` | `use_container_width=True` → `width='stretch'` (×9) |
