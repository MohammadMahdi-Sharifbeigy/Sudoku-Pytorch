"""
vision_b.py — Pipeline B: complete  grid extraction strategy.

Key differences from vision.py (Pipeline A):
  - CLAHE + median blur + Gaussian blur before thresholding
  - Adaptive threshold with tight params (blocksize=11, C=2)
  - Corner refinement via second-pass line isolation
  - Hough-line fallback for faint outer borders
  - Grid line removal via morphological open (CELL_SIZE-wide kernel)
  - Cell extraction: local quad curves that follow page folds, falling back to
    uniform row/col bounds — no contour NMS needed
  - _find_digit_mask: connected-component filters (size, density, centroid,
    aspect ratio) rather than raw area threshold
  - Digit normalised on grayscale crop (preserves loops in 6/8/9) and output
    resized to 28×28 for compatibility with existing models

Public API matches vision.py:
  get_valid_cells_from_image_b(img_rgb, ...) → (cells, M, warped)
  cells: list of 81 dicts matching vision.py format
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Grid geometry constants (tuned to WARP_SIZE) ──────────────────────────────
WARP_SIZE  = 450
CELL_SIZE  = WARP_SIZE // 9          # 50 px
WARP_MARGIN = CELL_SIZE // 2         # 25 px
GRID_SPAN  = WARP_SIZE - 1 - 2 * WARP_MARGIN
MAX_DETECT_DIM = 1024

MIN_DIGIT_AREA_RATIO   = 0.015
MIN_DIGIT_HEIGHT_RATIO = 0.25
CELL_MARGIN_RATIO      = 0.12
MIN_GRID_AREA_RATIO    = 0.05


# ── Public entry point ────────────────────────────────────────────────────────

def get_valid_cells_from_image_b(
    img_rgb: np.ndarray,
    digit_scale: int = 20,
    should_stop=None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Full Pipeline B: detect grid → warp → extract 81 cells.

    img_rgb : uint8 H×W×3 RGB image (same as vision.py convention).
    digit_scale : ignored (kept for API parity); B always outputs 28×28.
    Returns (cells, M, warped) where:
      cells  — list[81] of dicts with keys img, contains_digit,
               grid_row, grid_col, x_centroid, y_centroid
      M      — 3×3 perspective matrix (source → warped), for overlay
      warped — the 450×450 greyscale warped grid
    """
    _check_stop(should_stop)

    # Pipeline B works internally in BGR; our app uses RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Polarity correction: some scans have white grid on dark background
    if _is_light_on_dark(gray):
        gray = cv2.bitwise_not(gray)

    # ── Preprocessing ────────────────────────────────────────────────────────
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    denoised   = cv2.medianBlur(normalized, 5)
    blurred    = cv2.GaussianBlur(denoised, (5, 5), 0)
    binary     = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    # Downsample for grid detection if image is very large
    detect_scale = min(1.0, MAX_DETECT_DIM / max(gray.shape[:2]))
    if detect_scale < 1.0:
        dh = int(gray.shape[0] * detect_scale)
        dw = int(gray.shape[1] * detect_scale)
        d_gray   = cv2.resize(gray, (dw, dh))
        d_norm   = clahe.apply(d_gray)
        d_blur   = cv2.GaussianBlur(cv2.medianBlur(d_norm, 5), (5, 5), 0)
        d_bin    = cv2.adaptiveThreshold(
            d_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
    else:
        d_blur, d_bin = blurred, binary

    _check_stop(should_stop)

    # ── Grid boundary detection ───────────────────────────────────────────────
    corners = _locate_grid(d_bin, d_blur)
    if corners is None:
        raise Exception(
            "Pipeline B: no Sudoku grid boundary found. "
            "Ensure the grid is clearly visible and well-lit."
        )
    if detect_scale < 1.0:
        corners = corners / detect_scale

    corners = _order_corners(corners)
    corners = _refine_corners(normalized, corners)

    _check_stop(should_stop)

    # ── Perspective warp ──────────────────────────────────────────────────────
    M       = cv2.getPerspectiveTransform(corners, _warp_destination())
    warped  = cv2.warpPerspective(normalized, M, (WARP_SIZE, WARP_SIZE))

    _check_stop(should_stop)

    # ── Cell extraction ───────────────────────────────────────────────────────
    raw_cells = _process_warp(warped, should_stop=should_stop)

    # Convert Cell objects → vision.py dict format
    cw = WARP_SIZE / 9.0
    ch = WARP_SIZE / 9.0
    cells = []
    for idx, cell in enumerate(raw_cells):
        r, c = divmod(idx, 9)
        img_28 = _to_28(cell.image)
        cells.append({
            'img':            img_28,
            'contains_digit': not cell.is_empty,
            'grid_row':       r,
            'grid_col':       c,
            'x_centroid':     int((c + 0.5) * cw),
            'y_centroid':     int((r + 0.5) * ch),
        })

    return cells, M, warped


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _check_stop(should_stop):
    if should_stop is not None:
        should_stop()


def _is_light_on_dark(gray) -> bool:
    h, w = gray.shape[:2]
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    center = gray[y0:y1, x0:x1]
    blur   = cv2.GaussianBlur(center, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(cv2.countNonZero(mask)) / mask.size < 0.5


def _warp_destination() -> np.ndarray:
    low, high = WARP_MARGIN, WARP_SIZE - 1 - WARP_MARGIN
    return np.array(
        [[low, low], [high, low], [high, high], [low, high]], dtype=np.float32)


def _order_corners(points) -> np.ndarray:
    sums  = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array([
        points[sums.argmin()],
        points[diffs.argmin()],
        points[sums.argmax()],
        points[diffs.argmax()],
    ], dtype=np.float32)


# ── Grid location ─────────────────────────────────────────────────────────────

def _locate_grid(binary, blurred) -> np.ndarray | None:
    min_area = MIN_GRID_AREA_RATIO * binary.size
    pad = max(8, int(0.02 * max(binary.shape[:2])))
    binary  = cv2.copyMakeBorder(binary,  pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    blurred = cv2.copyMakeBorder(blurred, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.05, 0.1):
            quad = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                return quad.reshape(4, 2).astype(np.float32) - pad
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw > 0 and rh > 0 and rw * rh >= min_area and 0.5 < rw / rh < 2.0:
            return cv2.boxPoints(rect).astype(np.float32) - pad

    return _corners_from_hough(blurred, min_area, pad)


def _corners_from_hough(blurred, min_area, pad=0) -> np.ndarray | None:
    edges     = cv2.Canny(blurred, 50, 150)
    threshold = max(80, min(blurred.shape) // 4)
    lines     = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return None
    horizontal, vertical = [], []
    for rho, theta in lines[:, 0]:
        if rho < 0:
            rho, theta = -rho, theta - np.pi
        if abs(theta) < np.pi / 4:
            vertical.append((rho, theta))
        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontal.append((rho, theta))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    top, bottom = min(horizontal), max(horizontal)
    left, right = min(vertical),   max(vertical)
    points = []
    for pair in ((top, left), (top, right), (bottom, right), (bottom, left)):
        pt = _intersect_lines(*pair)
        if pt is None:
            return None
        points.append(pt)
    corners = np.array(points, dtype=np.float32)
    if cv2.contourArea(corners) < min_area:
        return None
    if not cv2.isContourConvex(corners.astype(np.int32)):
        return None
    return corners - pad


def _intersect_lines(a, b):
    (ra, ta), (rb, tb) = a, b
    coeffs = np.array([[np.cos(ta), np.sin(ta)], [np.cos(tb), np.sin(tb)]])
    if abs(np.linalg.det(coeffs)) < 1e-8:
        return None
    x, y = np.linalg.solve(coeffs, np.array([ra, rb]))
    return float(x), float(y)


def _refine_corners(normalized, corners) -> np.ndarray:
    M      = cv2.getPerspectiveTransform(corners, _warp_destination())
    warped = cv2.warpPerspective(normalized, M, (WARP_SIZE, WARP_SIZE))
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(warped, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

    length       = WARP_SIZE // 10
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    horizontal   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vertical     = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    grid_lines   = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return corners

    lc = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(lc)
    if w < 0.5 * WARP_SIZE or h < 0.5 * WARP_SIZE:
        return corners
    if w > 0.95 * WARP_SIZE and h > 0.95 * WARP_SIZE and x < 0.05 * WARP_SIZE and y < 0.05 * WARP_SIZE:
        return corners

    perimeter = cv2.arcLength(lc, True)
    Minv = np.linalg.inv(M)
    for eps in (0.02, 0.05, 0.1):
        quad = cv2.approxPolyDP(lc, eps * perimeter, True)
        if len(quad) == 4 and cv2.isContourConvex(quad):
            refined = cv2.perspectiveTransform(
                quad.reshape(-1, 1, 2).astype(np.float32), Minv).reshape(4, 2)
            return _order_corners(refined)

    quad = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    refined = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), Minv).reshape(4, 2)
    return _order_corners(refined)


# ── Cell extraction (local-quad strategy) ────────────────────────────

class _Cell:
    def __init__(self, image, is_empty):
        self.image    = image     # np.ndarray
        self.is_empty = is_empty  # bool


def _process_warp(warped, should_stop=None) -> list[_Cell]:
    warped_blurred = cv2.medianBlur(warped, 3)

    warped_binary = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    warped_binary_clean = cv2.adaptiveThreshold(
        warped_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    line_free, horizontal_lines, vertical_lines = _remove_grid_lines(
        warped_binary, warped_binary_clean)
    row_bounds = _grid_boundaries(horizontal_lines, axis=1)
    col_bounds = _grid_boundaries(vertical_lines,   axis=0)

    _check_stop(should_stop)

    quads, _ = _detect_cell_quads(warped_binary, row_bounds, col_bounds)

    _check_stop(should_stop)

    if quads is not None:
        return [_extract_cell_quad(line_free, warped, q) for q in quads]
    return [
        _extract_cell(line_free, warped, row_bounds, col_bounds, r, c)
        for r in range(9) for c in range(9)
    ]


def _remove_grid_lines(warped_binary, digit_binary=None):
    h_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_SIZE, 1))
    v_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL_SIZE))
    horizontal = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, h_kernel)
    vertical   = cv2.morphologyEx(warped_binary, cv2.MORPH_OPEN, v_kernel)
    lines      = cv2.dilate(
        cv2.bitwise_or(horizontal, vertical),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    target     = digit_binary if digit_binary is not None else warped_binary
    line_free  = cv2.bitwise_and(target, cv2.bitwise_not(lines))
    return line_free, horizontal, vertical


def _grid_boundaries(lines_mask, axis) -> np.ndarray:
    profile = (lines_mask > 0).sum(axis=axis)
    search  = CELL_SIZE // 3
    bounds  = np.empty(10, dtype=int)
    for i in range(10):
        expected = int(round(WARP_MARGIN + i * GRID_SPAN / 9))
        low    = max(0, expected - search)
        window = profile[low: min(WARP_SIZE, expected + search + 1)]
        if window.max() >= 0.3 * WARP_SIZE:
            bounds[i] = low + int(window.argmax())
        else:
            bounds[i] = expected
    return np.maximum.accumulate(bounds)


def _line_mask(binary) -> np.ndarray:
    thicken = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    run     = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_SIZE // 2, 1))
    opened  = cv2.morphologyEx(cv2.dilate(binary, thicken), cv2.MORPH_OPEN, run)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep    = stats[:, cv2.CC_STAT_WIDTH] >= CELL_SIZE
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _trace_curve(mask, expected):
    half = CELL_SIZE // 2 - 4
    lo   = max(0, int(expected) - half)
    hi   = min(WARP_SIZE, int(expected) + half + 1)
    band = mask[lo:hi] > 0
    counts  = band.sum(axis=0)
    sampled = counts > 0
    if sampled.sum() < 0.35 * WARP_SIZE:
        return None
    rows    = np.arange(lo, hi, dtype=np.float64)[:, None]
    centers = (band * rows).sum(axis=0)[sampled] / counts[sampled]
    fit     = np.polyfit(np.nonzero(sampled)[0], centers, 2)
    curve   = np.polyval(fit, np.arange(WARP_SIZE))
    return np.clip(curve, lo, hi - 1)


def _trace_lines(mask, bounds):
    curves, traced = [], 0
    for expected in bounds:
        curve   = _trace_curve(mask, expected)
        traced += curve is not None
        curves.append(curve if curve is not None else np.full(WARP_SIZE, float(expected)))
    return np.maximum.accumulate(np.stack(curves), axis=0), traced


def _detect_cell_quads(warped_binary, row_bounds, col_bounds):
    h_mask = _line_mask(warped_binary)
    v_mask = _line_mask(warped_binary.T).T

    h_curves, h_traced = _trace_lines(h_mask,   row_bounds)
    v_curves, v_traced = _trace_lines(v_mask.T, col_bounds)

    if h_traced + v_traced < 8:
        return None, {}

    nodes = np.empty((10, 10, 2), dtype=np.float32)
    for i in range(10):
        for j in range(10):
            y = float(row_bounds[i])
            for _ in range(2):
                x = v_curves[j][int(y)]
                y = h_curves[i][int(x)]
            nodes[i, j] = (x, y)

    tl, tr = nodes[:-1, :-1], nodes[:-1, 1:]
    bl, br = nodes[1:, :-1],  nodes[1:, 1:]
    quads  = np.stack([tl, tr, br, bl], axis=2).reshape(81, 4, 2)
    return quads, {}


def _extract_cell_quad(line_free, warped, quad) -> _Cell:
    dst = np.array(
        [[0, 0], [CELL_SIZE-1, 0], [CELL_SIZE-1, CELL_SIZE-1], [0, CELL_SIZE-1]],
        dtype=np.float32)
    H             = cv2.getPerspectiveTransform(quad, dst)
    lf_patch      = cv2.warpPerspective(line_free, H, (CELL_SIZE, CELL_SIZE),
                                        flags=cv2.INTER_NEAREST)
    gray_patch    = cv2.warpPerspective(warped, H, (CELL_SIZE, CELL_SIZE))
    return _cell_from_patch(lf_patch, gray_patch)


def _extract_cell(line_free, warped, row_bounds, col_bounds, row, col) -> _Cell:
    y0, y1 = row_bounds[row],   row_bounds[row + 1]
    x0, x1 = col_bounds[col],  col_bounds[col + 1]
    return _cell_from_patch(line_free[y0:y1, x0:x1], warped[y0:y1, x0:x1])


def _cell_from_patch(lf_patch, gray_patch) -> _Cell:
    height, width = lf_patch.shape[:2]
    y0 = int(height * CELL_MARGIN_RATIO)
    y1 = height - y0
    x0 = int(width  * CELL_MARGIN_RATIO)
    x1 = width  - x0
    if y1 - y0 < 8 or x1 - x0 < 8:
        return _Cell(np.zeros((28, 28), np.uint8), True)

    interior   = lf_patch[y0:y1, x0:x1]
    digit_mask = _find_digit_mask(interior)
    if digit_mask is None:
        return _Cell(np.zeros((28, 28), np.uint8), True)

    ys, xs = np.nonzero(digit_mask)
    pad    = 3
    gy0 = max(0, y0 + ys.min() - pad)
    gy1 = min(height, y0 + ys.max() + 1 + pad)
    gx0 = max(0, x0 + xs.min() - pad)
    gx1 = min(width,  x0 + xs.max() + 1 + pad)

    mask_full              = np.zeros((height, width), np.uint8)
    mask_full[y0:y1, x0:x1] = digit_mask
    digit = _normalize_digit(gray_patch[gy0:gy1, gx0:gx1],
                             mask_full[gy0:gy1, gx0:gx1])
    return _Cell(digit, False)


def _find_digit_mask(interior) -> np.ndarray | None:
    height, width = interior.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        interior, connectivity=8)

    anchor, anchor_area = None, 0
    for lbl in range(1, count):
        x, y, w, h, area = stats[lbl]
        if area < MIN_DIGIT_AREA_RATIO * interior.size:
            continue
        if h < MIN_DIGIT_HEIGHT_RATIO * height:
            continue
        if w > 1.2 * h:
            continue
        if area / max(w * h, 1) < 0.15:
            continue
        cx, cy = centroids[lbl]
        if not (0.15 * width < cx < 0.85 * width and
                0.15 * height < cy < 0.85 * height):
            continue
        if area > anchor_area:
            anchor, anchor_area = lbl, area

    if anchor is None:
        return None

    ax, ay, aw, ah, _ = stats[anchor]
    grow = max(2, min(height, width) // 8)
    left, top     = ax - grow,      ay - grow
    right, bottom = ax + aw + grow, ay + ah + grow
    min_frag      = 0.3 * MIN_DIGIT_AREA_RATIO * interior.size

    mask = np.zeros_like(interior)
    for lbl in range(1, count):
        x, y, w, h, area = stats[lbl]
        if lbl != anchor:
            if area < min_frag:
                continue
            if x + w < left or y + h < top or x > right or y > bottom:
                continue
        mask[labels == lbl] = 255
    return mask


def _normalize_digit(gray_crop, mask_crop) -> np.ndarray:
    """Normalise digit crop and resize to 28×28 (model input size)."""
    h, w   = mask_crop.shape
    k      = max(1, round(min(h, w) * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    mask   = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)

    inverted = cv2.bitwise_not(gray_crop)
    digit    = cv2.bitwise_and(inverted, inverted, mask=mask)
    digit    = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)

    dh, dw = digit.shape
    side   = int(max(dh, dw) * 1.3)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - dh) // 2: (side - dh) // 2 + dh,
           (side - dw) // 2: (side - dw) // 2 + dw] = digit
    return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)


def _to_28(img: np.ndarray) -> np.ndarray:
    """Ensure image is 28×28 uint8."""
    if img.shape == (28, 28):
        return img
    return cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
