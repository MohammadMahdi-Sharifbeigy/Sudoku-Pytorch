"""
vision.py – Contour-based Sudoku grid detection and cell extraction with NMS.

Pipeline:
  Grid detection  : multi-param adaptive threshold → contours → NMS → perspective warp
  Cell extraction : multi-param adaptive threshold → contours → NMS → 81 best cells
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imutils
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────────────────────────────────────
# Basic image helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_and_maintain_aspect_ratio(input_image, new_width):
    ratio = new_width / float(input_image.shape[1])
    new_height = int(input_image.shape[0] * ratio)
    return cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7):
    """Accepts RGB 3-channel or single-channel grayscale input."""
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY) if len(blurred.shape) == 3 else blurred
    am = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(gray, 255, am, cv2.THRESH_BINARY, blocksize, c)
    return cv2.bitwise_not(thresh)


def get_quadrilateral_points_in_order(approx_arr):
    if approx_arr.shape == (4, 1, 2):
        approx_arr = np.squeeze(approx_arr, axis=1)
    max_x = int(1.1 * np.max(approx_arr[:, 0]))
    d1 = [np.linalg.norm(p - [0, 0])     for p in approx_arr]
    d2 = [np.linalg.norm(p - [max_x, 0]) for p in approx_arr]
    tl_idx, br_idx = int(np.argmin(d1)), int(np.argmax(d1))
    d2_tmp = d2.copy(); d2_tmp[tl_idx] = np.inf;  d2_tmp[br_idx] = np.inf
    tr_idx = int(np.argmin(d2_tmp))
    d2_tmp = d2.copy(); d2_tmp[tl_idx] = -np.inf; d2_tmp[br_idx] = -np.inf
    bl_idx = int(np.argmax(d2_tmp))
    return np.array([approx_arr[tl_idx], approx_arr[tr_idx],
                     approx_arr[br_idx], approx_arr[bl_idx]])


def perform_four_point_transform(input_img, src_corners, pad=10, size=None):
    """
    Warp the quadrilateral defined by src_corners to a top-down view.

    size=None : aspect-preserving warp (max_w x max_h) — original behaviour.
    size=int  : square warp to (size x size). Use this for the grid so cells
                can be sliced deterministically into a perfect 9x9.
    """
    src = get_quadrilateral_points_in_order(src_corners).astype('float32')
    if size is not None:
        dst = np.array([[0, 0], [size-1, 0],
                        [size-1, size-1], [0, size-1]], dtype='float32')
        M = cv2.getPerspectiveTransform(src, dst)
        return M, cv2.warpPerspective(input_img, M, (size, size))
    tl, tr, br, bl = src
    max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_h = max(int(np.linalg.norm(tl - bl)), int(np.linalg.norm(tr - br)))
    dst = np.array([[pad, pad], [max_w-1-pad, pad],
                    [max_w-1-pad, max_h-1-pad], [pad, max_h-1-pad]], dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(input_img, M, (max_w, max_h))
    return M, warped


def center_and_resize_digit(cell_img):
    contours, _ = cv2.findContours(cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(cell_img, (28, 28), interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = cell_img[y:y+h, x:x+w]
    max_side = max(w, h)
    if max_side == 0:
        return cv2.resize(cell_img, (28, 28))
    scale = 20.0 / max_side
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    sy, sx = (28 - nh) // 2, (28 - nw) // 2
    canvas[sy:sy+nh, sx:sx+nw] = resized
    return canvas


def check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False):
    cell = img.copy()
    if apply_border:
        bf = 0.07
        yb, xb = max(1, int(bf * cell.shape[0])), max(1, int(bf * cell.shape[1]))
        cell[:yb, :] = 0; cell[-yb:, :] = 0
        cell[:, :xb] = 0; cell[:, -xb:] = 0
    contours = imutils.grab_contours(
        cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
    if contours:
        area = cv2.contourArea(max(contours, key=cv2.contourArea))
        pct  = 100 * area / max(1, cell.shape[0] * cell.shape[1])
        return pct > area_threshold, cell
    return False, cell


# ─────────────────────────────────────────────────────────────────────────────
# NMS utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_iou(b1, b2):
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(w1*h1 + w2*h2 - inter, 1)


def _nms(candidates, iou_threshold=0.3):
    """
    Non-Maximum Suppression.
    Input : list of (score, bbox=(x,y,w,h), payload)
    Output: subset list with overlapping boxes removed, highest score kept.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
    kept, suppressed = [], set()
    for i, item in enumerate(ordered):
        if i in suppressed:
            continue
        kept.append(item)
        for j in range(i + 1, len(ordered)):
            if j not in suppressed and _bbox_iou(item[1], ordered[j][1]) > iou_threshold:
                suppressed.add(j)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Quad helper
# ─────────────────────────────────────────────────────────────────────────────

def _contour_to_quad(contour):
    """Reduce any contour to 4 corners via convex-hull extreme points."""
    hull = cv2.convexHull(contour)[:, 0, :]
    tl = hull[np.argmin( hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmax( hull[:, 0] - hull[:, 1])]
    br = hull[np.argmax( hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmin( hull[:, 0] - hull[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Grid boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def find_grid_contour_candidates(img):
    """
    Detect the Sudoku grid boundary using contours across multiple threshold
    parameter combinations. NMS removes duplicate candidates from different params.

    Accepts 4-corner contours (exact) and 5-14-corner polygons (reduced to 4
    via convex-hull extreme points).

    Returns (M_matrices, warped_images, contours) or (None, None, None).
    """
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    all_candidates = []   # (score, bbox, pts, raw_contour)

    for blocksize, c_val in [(41, 8), (21, 5), (61, 10), (31, 6), (11, 3), (81, 12)]:
        thresh = apply_grayscale_blur_and_threshold(img, blocksize=blocksize, c=c_val)

        # Close gaps in thin / broken (e.g. hand-drawn) grid lines so the outer
        # boundary forms a single closed contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        for retr in [cv2.RETR_EXTERNAL, cv2.RETR_LIST]:
            contours = imutils.grab_contours(
                cv2.findContours(thresh.copy(), retr, cv2.CHAIN_APPROX_SIMPLE))
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:8]:
                area       = cv2.contourArea(contour)
                area_ratio = area / img_area
                if not (0.04 < area_ratio < 0.97):
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx    = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                n         = len(approx)

                if n == 4:
                    pts = np.squeeze(approx, axis=1).astype(np.float32)
                elif 4 < n <= 14:
                    pts = _contour_to_quad(contour)
                else:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                squareness  = min(w, h) / max(max(w, h), 1)
                score       = squareness * area_ratio

                all_candidates.append((score, (x, y, w, h), pts, contour))

    # NMS: suppress grid candidates that overlap significantly
    nms_in = [(s, b, (p, c)) for s, b, p, c in all_candidates]
    kept   = _nms(nms_in, iou_threshold=0.5)

    M_matrices, warped_images, contour_list = [], [], []
    for score, bbox, (pts, contour) in kept:
        try:
            M, warped = perform_four_point_transform(img, pts, pad=20)
            if warped.shape[0] >= 50 and warped.shape[1] >= 50:
                M_matrices.append(M)
                warped_images.append(warped)
                contour_list.append(contour)
        except Exception:
            continue

    return (M_matrices, warped_images, contour_list) if warped_images else (None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Cell extraction
# ─────────────────────────────────────────────────────────────────────────────

def locate_cells_within_grid(grid_img):
    """
    Find 81 cells inside a perspective-corrected grid image.

    Algorithm:
      For each of several (blocksize, C) pairs:
        1. Threshold the grid image.
        2. Find all 4-corner contours in the expected cell-area range.
        3. Score each contour: squareness × area-regularity × fill-ratio.
        4. Apply NMS (IoU > 0.3) to remove duplicate / overlapping detections.
        5. Keep the result with the most cells; stop early when 81 are found.
    """
    grid_area         = grid_img.shape[0] * grid_img.shape[1]
    expected_cell_area = grid_area / 81.0
    best_cells        = []

    for blocksize, c_val in [(91, 7), (51, 5), (71, 9), (31, 4), (111, 10), (41, 6)]:
        thresh = apply_grayscale_blur_and_threshold(
            grid_img, method="mean", blocksize=blocksize, c=c_val)

        contours = imutils.grab_contours(
            cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            frac = area / grid_area
            if not (0.003 < frac < 0.022):
                continue

            perimeter = cv2.arcLength(contour, True)
            approx    = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Composite quality score
            squareness      = min(w, h) / max(max(w, h), 1)
            area_regularity = 1.0 / (1.0 + abs(area / max(expected_cell_area, 1) - 1.0))
            fill            = area / max(w * h, 1)
            score           = squareness * area_regularity * fill

            candidates.append((score, (x, y, w, h), contour))

        # Suppress overlapping cell detections
        kept = _nms(candidates, iou_threshold=0.3)

        # Extract each kept cell
        valid_cells = []
        for score, bbox, contour in kept:
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
            y_px, x_px = np.where(mask == 255)
            if len(y_px) == 0:
                continue
            cell_img = thresh[min(y_px):max(y_px)+1, min(x_px):max(x_px)+1]
            has_digit, cell_img = check_for_digit_in_cell_image(
                cell_img, area_threshold=4, apply_border=True)
            cell_img = center_and_resize_digit(cell_img) if has_digit else np.zeros((28, 28), dtype=np.uint8)
            moments  = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            valid_cells.append({
                'img':          cell_img,
                'contains_digit': has_digit,
                'x_centroid':   int(moments['m10'] / moments['m00']),
                'y_centroid':   int(moments['m01'] / moments['m00']),
            })

        if len(valid_cells) > len(best_cells):
            best_cells = valid_cells
        if len(best_cells) == 81:
            break

    return best_cells


def sort_cells_into_grid(cells):
    max_x  = max(c['x_centroid'] for c in cells)
    max_y  = max(c['y_centroid'] for c in cells)
    cell_w = (max_x * 1.1) / 9.0
    cell_h = (max_y * 1.1) / 9.0
    for cell in cells:
        cell['grid_row'] = min(int(cell['y_centroid'] / cell_h), 8)
        cell['grid_col'] = min(int(cell['x_centroid'] / cell_w), 8)
    return sorted(cells, key=lambda c: (c['grid_row'], c['grid_col']))


def _clear_border_components(binary):
    """
    Remove connected white components that touch the image border.

    In a sliced grid cell the digit sits near the centre while leftover grid
    lines run along the edges; dropping border-touching components removes the
    lines without harming a centred digit. Returns a new binary image.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape[:2]
    out = np.zeros_like(binary)
    for lbl in range(1, n):  # 0 = background
        x, y, bw, bh, _ = stats[lbl]
        touches = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
        if not touches:
            out[labels == lbl] = 255
    return out


def slice_grid_into_cells(grid_img, n=9, pad_frac=0.02):
    """
    Deterministic fallback: split a square perspective-corrected grid into an
    exact n x n lattice. Always returns n*n cells in row-major order — no
    contour detection, so it survives faint / wavy / hand-drawn grid lines
    that defeat locate_cells_within_grid.

    pad_frac trims each cell inward to drop the surrounding grid lines before
    digit detection.
    """
    H, W = grid_img.shape[:2]
    ch, cw = H / float(n), W / float(n)
    cells = []
    for r in range(n):
        for c in range(n):
            y0, y1 = int(round(r * ch)), int(round((r + 1) * ch))
            x0, x1 = int(round(c * cw)), int(round((c + 1) * cw))
            cell = grid_img[y0:y1, x0:x1]

            py = max(1, int(pad_frac * cell.shape[0]))
            px = max(1, int(pad_frac * cell.shape[1]))
            inner = cell[py:-py, px:-px]
            if inner.size == 0:
                inner = cell

            thr = apply_grayscale_blur_and_threshold(
                inner, method="mean", blocksize=31, c=7)
            thr = _clear_border_components(thr)
            has_digit, thr = check_for_digit_in_cell_image(
                thr, area_threshold=4, apply_border=False)
            cell_img = center_and_resize_digit(thr) if has_digit \
                else np.zeros((28, 28), dtype=np.uint8)

            cells.append({
                'img':            cell_img,
                'contains_digit': has_digit,
                'grid_row':       r,
                'grid_col':       c,
                'x_centroid':     int((c + 0.5) * cw),
                'y_centroid':     int((r + 0.5) * ch),
            })
    return cells


def build_grid_from_partial_cells(cells, n=9):
    """
    Map an unordered set of detected cells onto an exact n x n grid.

    locate_cells_within_grid usually finds *most* but not all 81 cells on a
    printed grid (e.g. 75-78). The old pipeline discarded that result whenever
    it was not exactly 81 and fell back to deterministic slicing, which loses
    digits when the warp is slightly skewed. Instead, cluster the detected
    cell centroids into n row-bands and n column-bands (1-D KMeans per axis),
    place each cell in its (row, col) slot, and fill any empty slot with a
    blank cell. This keeps every digit the contour detector already found.

    Requires at least n detected cells (KMeans needs >= n samples per axis).
    Returns a row-major list of exactly n*n cell dicts, or None if clustering
    is not possible.
    """
    if len(cells) < n:
        return None

    xs = np.array([c['x_centroid'] for c in cells], dtype=np.float32).reshape(-1, 1)
    ys = np.array([c['y_centroid'] for c in cells], dtype=np.float32).reshape(-1, 1)

    kx = KMeans(n_clusters=n, n_init=10, random_state=0).fit(xs)
    ky = KMeans(n_clusters=n, n_init=10, random_state=0).fit(ys)

    col_centers = kx.cluster_centers_.ravel()
    row_centers = ky.cluster_centers_.ravel()
    col_rank = {lbl: rank for rank, lbl in enumerate(np.argsort(col_centers))}
    row_rank = {lbl: rank for rank, lbl in enumerate(np.argsort(row_centers))}
    col_sorted = col_centers[np.argsort(col_centers)]
    row_sorted = row_centers[np.argsort(row_centers)]

    # Place cells into slots; on collision prefer the cell that has a digit.
    slots = {}
    for i, c in enumerate(cells):
        r = row_rank[ky.labels_[i]]
        col = col_rank[kx.labels_[i]]
        key = (r, col)
        if key not in slots or (c['contains_digit'] and not slots[key]['contains_digit']):
            slots[key] = c

    out = []
    for r in range(n):
        for col in range(n):
            if (r, col) in slots:
                cell = dict(slots[(r, col)])
                cell['grid_row'] = r
                cell['grid_col'] = col
            else:
                cell = {
                    'img':            np.zeros((28, 28), dtype=np.uint8),
                    'contains_digit': False,
                    'grid_row':       r,
                    'grid_col':       col,
                    'x_centroid':     int(col_sorted[col]),
                    'y_centroid':     int(row_sorted[r]),
                }
            out.append(cell)
    return out


def get_valid_cells_from_image(img, grid_size=576):
    """
    Full pipeline: detect grid boundary -> warp -> extract 81 cells.

    Strategy:
      1. Contour path: for each candidate grid, try locate_cells_within_grid.
         Return immediately if exactly 81 cells are detected.
      2. Partial-grid recovery: if the best candidate found most (but not all)
         cells, cluster their centroids into a 9x9 lattice and fill the gaps.
         This preserves every digit the contour detector found instead of
         throwing the whole result away.
      3. Slice fallback: if too few cells were found (faint / wavy / hand-drawn
         grids), square-warp the best candidate and slice a deterministic 9x9.
    """
    M_matrices, warped_images, contour_list = find_grid_contour_candidates(img)
    if not warped_images:
        raise Exception(
            "No grid boundary detected. Make sure the Sudoku grid is clearly "
            "visible and fills most of the image."
        )

    # ── 1. Contour path ──────────────────────────────────────────────────
    # Track the best candidate (most cells) for recovery / fallback.
    best = None  # (n_cells, cells, M, grid_image, contour)
    for i, grid_image in enumerate(warped_images):
        cells = locate_cells_within_grid(grid_image)
        if len(cells) == 81:
            return sort_cells_into_grid(cells), M_matrices[i], grid_image
        if best is None or len(cells) > best[0]:
            best = (len(cells), cells, M_matrices[i], grid_image, contour_list[i])

    # ── 2. Partial-grid recovery ─────────────────────────────────────────
    # 54 = two-thirds of 81: enough detected cells that centroid clustering
    # reliably reconstructs the 9x9 layout.
    best_n, best_cells, best_M, best_grid, best_contour = best
    if best_n >= 54:
        recovered = build_grid_from_partial_cells(best_cells)
        if recovered is not None:
            return recovered, best_M, best_grid

    # ── 3. Slice fallback ────────────────────────────────────────────────
    # Re-warp the best grid candidate to a square so it slices into an exact
    # 9x9. A fresh M is required because the square warp differs from the
    # aspect-preserving warp used above (generate_solution_image unwarps with it).
    perimeter = cv2.arcLength(best_contour, True)
    approx    = cv2.approxPolyDP(best_contour, 0.03 * perimeter, True)
    pts = (np.squeeze(approx, axis=1).astype(np.float32)
           if len(approx) == 4 else _contour_to_quad(best_contour))

    M_sq, grid_sq = perform_four_point_transform(img, pts, size=grid_size)
    cells = slice_grid_into_cells(grid_sq)
    return cells, M_sq, grid_sq


# ─────────────────────────────────────────────────────────────────────────────
# Prediction & visualisation (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def get_predicted_sudoku_grid_torch(model, cells, device):
    digit_images = np.array([c['img'] for c in cells if c['contains_digit']])
    if len(digit_images) == 0:
        return np.zeros((9, 9), dtype=int)
    tensors = torch.from_numpy(digit_images).float().unsqueeze(1).div(255.0).to(device)
    with torch.no_grad():
        preds = torch.argmax(model(tensors), dim=1).cpu().numpy()
    indices = np.where([c['contains_digit'] for c in cells])[0]
    grid    = np.zeros(81, dtype=int)
    grid[indices] = preds
    return np.reshape(grid, (9, 9))


def plot_cell_images_in_grid(cells):
    canvas = np.zeros((9 * 28, 9 * 28))
    for i, cell in enumerate(cells):
        r, c = divmod(i, 9)
        canvas[r*28:(r+1)*28, c*28:(c+1)*28] = cell['img']
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap='gray')
    ax.set_title('Extracted and Sorted 81 Cells (9×9 Grid Layout)', fontweight='bold')
    ax.axis('off')
    return fig


def generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w  = board_image.shape[:2]
    sol   = np.ones((h, w, 3), dtype=np.uint8) * 255
    flat  = np.array(solved_board_arr).flatten()
    for i, cell in enumerate(cells_list):
        if not cell['contains_digit']:
            text = str(flat[i])
            tw, th = cv2.getTextSize(text, font, 1, 2)[0]
            tx = int(cell['x_centroid'] - tw / 2)
            ty = int(cell['y_centroid'] + th / 2)
            cv2.putText(sol, text, (tx, ty), font, 1.3, (0, 0, 0), 2)
    unwarped = cv2.warpPerspective(
        sol, M_matrix, (full_image.shape[1], full_image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    annotated = full_image.copy()
    annotated[np.all(unwarped < 50, axis=-1)] = (255, 0, 0)
    return annotated