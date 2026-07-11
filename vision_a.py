"""
vision_a.py — Canny-first Sudoku pipeline with HoughLinesP grid refinement.

Pipeline:
  Pre-process : CLAHE + GaussianBlur(7,7)
  Grid detect : grayscale → Canny(50,150) → dilate → biggest 4-point contour
  Warp        : four-point perspective to 576×576 square
  Refine      : HoughLinesP → cluster into 10H + 10V lines → exact cell bounds
  Line erase  : paint detected lines black (thickness=7)
  Cell slice  : deterministic crop from Hough line positions
"""

import cv2
import numpy as np
import imutils
import torch
import matplotlib.pyplot as plt

from vision import (
    get_quadrilateral_points_in_order,
    perform_four_point_transform,
    apply_grayscale_blur_and_threshold,
    check_for_digit_in_cell_image,
    center_and_resize_digit,
    _clear_border_components,
    _contour_to_quad,
    find_grid_contour_candidates,
    plot_cell_images_in_grid,
    generate_solution_image,
)


def preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """BGR → BGR: apply CLAHE on L channel, then GaussianBlur(7,7)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(bgr, (7, 7), 1)


def detect_grid_canny(preprocessed: np.ndarray) -> np.ndarray | None:
    """
    BGR → float32 (4,2) corner array [TL,TR,BR,BL] or None.

    Uses Canny edge detection — more stable than adaptive threshold at steep angles
    because Canny responds to intensity gradients, not local mean.
    """
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours = imutils.grab_contours(
        cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    )
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
    if len(approx) != 4:
        return None

    pts = np.squeeze(approx, axis=1).astype(np.float32)
    return get_quadrilateral_points_in_order(pts)


def _cluster_positions(positions: list[float], gap: int = 15) -> np.ndarray:
    """
    Group nearby float positions into clusters, return sorted cluster medians.
    Used to collapse many Hough line detections into the 10 true grid lines.
    """
    arr = np.sort(np.array(positions, dtype=np.float32))
    groups: list[list[float]] = [[arr[0]]]
    for p in arr[1:]:
        if p - groups[-1][-1] < gap:
            groups[-1].append(float(p))
        else:
            groups.append([float(p)])
    return np.array([np.median(g) for g in groups], dtype=np.float32)


def _preprocess_for_hough(warped_bgr: np.ndarray) -> np.ndarray:
    """Convert warped BGR to binary image with grid lines as white."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY) if len(warped_bgr.shape) == 3 else warped_bgr
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=18,
    )
    return cv2.bitwise_not(thresh)


def _complete_to_ten(clusters: np.ndarray, size: int) -> np.ndarray:
    """
    Ensure exactly 10 grid border positions spanning [0, size].

    HoughLinesP misses outer borders when they fall on the warped image edge.
    We insert 0 / size at the appropriate end rather than duplicating with
    mode='edge', which would silently shift all rows by one.
    """
    result = list(clusters)
    cell_gap = size / 9.0

    # Insert outer borders if absent (threshold: must be within half a cell of edge)
    if not result or result[0] > cell_gap * 0.5:
        result.insert(0, 0.0)
    if len(result) < 2 or result[-1] < size - cell_gap * 0.5:
        result.append(float(size))

    # Trim any excess (e.g. spurious lines beyond borders)
    result = sorted(result)[:10]

    # If still short, interpolate missing inner lines
    while len(result) < 10:
        gaps = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        biggest = max(range(len(gaps)), key=lambda i: gaps[i])
        mid = (result[biggest] + result[biggest + 1]) / 2.0
        result.insert(biggest + 1, mid)

    return np.array(result[:10], dtype=np.float32)


def refine_corners_hough(warped_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect exact positions of the 10 horizontal and 10 vertical grid lines
    via HoughLinesP on the warped image.

    Returns (h_ys, v_xs) — sorted y-positions of the 10 H lines and sorted
    x-positions of the 10 V lines — or None if detection fails.

    These positions give exact cell boundaries that correct for any residual
    skew left over from the initial four-point perspective transform.
    """
    h, w = warped_bgr.shape[:2]
    binary = _preprocess_for_hough(warped_bgr)

    min_len = int(0.4 * min(h, w))
    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=min_len, maxLineGap=30,
    )
    if lines is None:
        return None

    h_positions, v_positions = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30:
            h_positions.append((y1 + y2) / 2.0)
        elif angle > 60:
            v_positions.append((x1 + x2) / 2.0)

    if len(h_positions) < 6 or len(v_positions) < 6:
        return None

    h_clusters = _cluster_positions(h_positions, gap=10)
    v_clusters = _cluster_positions(v_positions, gap=10)

    if len(h_clusters) < 6 or len(v_clusters) < 6:
        return None

    h_ys = _complete_to_ten(h_clusters, h)
    v_xs = _complete_to_ten(v_clusters, w)

    return h_ys, v_xs


def remove_grid_lines_hough(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Return a grayscale image with Sudoku grid lines painted out (set to 0).

    Strategy: adaptive-threshold the warped image so grid lines are white,
    detect them with HoughLinesP, paint thick black over each detected line.
    Thickness=7 erases the line while leaving digit pixels near the border intact.
    """
    h, w = warped_bgr.shape[:2]
    binary = _preprocess_for_hough(warped_bgr)

    min_len = int(0.35 * min(h, w))
    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=60, minLineLength=min_len, maxLineGap=40,
    )

    result = binary.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), 0, thickness=7)

    return result


def slice_cells_from_lines(
    warped_clean: np.ndarray,
    h_ys: np.ndarray,
    v_xs: np.ndarray,
) -> list[dict]:
    """
    Extract 81 cell dicts using exact grid line positions from Hough detection.

    warped_clean : 2D uint8 grayscale image with grid lines already removed
    h_ys         : sorted (10,) float32 — y-positions of 10 horizontal grid lines
    v_xs         : sorted (10,) float32 — x-positions of 10 vertical grid lines

    Returns list of 81 dicts with keys:
      img, contains_digit, grid_row, grid_col, x_centroid, y_centroid
    """
    H, W = warped_clean.shape[:2]
    cells = []
    for r in range(9):
        for c in range(9):
            y0, y1 = int(h_ys[r]),   int(h_ys[r + 1])
            x0, x1 = int(v_xs[c]),   int(v_xs[c + 1])
            y0, y1 = max(0, y0), min(H, y1)
            x0, x1 = max(0, x0), min(W, x1)

            patch = warped_clean[y0:y1, x0:x1]
            if patch.size == 0:
                cells.append({
                    'img': np.zeros((28, 28), dtype=np.uint8),
                    'contains_digit': False,
                    'grid_row': r, 'grid_col': c,
                    'x_centroid': int((x0 + x1) / 2),
                    'y_centroid': int((y0 + y1) / 2),
                })
                continue

            thr = _clear_border_components(patch)
            has_digit, thr = check_for_digit_in_cell_image(
                thr, area_threshold=4, apply_border=False)
            cell_img = center_and_resize_digit(thr) if has_digit \
                else np.zeros((28, 28), dtype=np.uint8)

            cells.append({
                'img':            cell_img,
                'contains_digit': has_digit,
                'grid_row':       r,
                'grid_col':       c,
                'x_centroid':     int((x0 + x1) / 2),
                'y_centroid':     int((y0 + y1) / 2),
            })
    return cells


def get_valid_cells_from_image(
    img: np.ndarray,
    grid_size: int = 576,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Full pipeline: BGR image → (cells, M, warped_grid).

    Returns same schema as vision.get_valid_cells_from_image so main.py can
    use either pipeline interchangeably.

    Strategy:
      1. CLAHE + GaussianBlur preprocessing
      2. Canny → detect 4-corner grid boundary
         Fallback: multi-param adaptive threshold (existing find_grid_contour_candidates)
      3. Four-point perspective warp → 576×576 square
      4. HoughLinesP → 10H + 10V grid lines → exact cell bounds
         Fallback: uniform 9×9 grid (equivalent to existing slice_grid_into_cells)
      5. Remove grid lines (Hough-based)
      6. Slice 81 cells from cleaned image using Hough line positions
    """
    preprocessed = preprocess_clahe(img)

    # ── 1. Grid boundary detection ──────────────────────────────────────────
    pts = detect_grid_canny(preprocessed)

    if pts is not None:
        M, warped = perform_four_point_transform(img, pts, size=grid_size)
    else:
        # Fallback to existing multi-param contour detection
        M_matrices, warped_images, _ = find_grid_contour_candidates(img)
        if not warped_images:
            raise Exception(
                "No grid boundary detected. Ensure the Sudoku grid fills most "
                "of the image and is well-lit."
            )
        warped = warped_images[0]
        M = M_matrices[0]
        warped = cv2.resize(warped, (grid_size, grid_size))

    # ── 2. Hough refinement → exact cell bounds ──────────────────────────────
    refined = refine_corners_hough(warped)
    if refined is not None:
        h_ys, v_xs = refined
    else:
        # Uniform fallback — divides warped evenly into 9×9
        h_ys = np.linspace(0, warped.shape[0], 10, dtype=np.float32)
        v_xs = np.linspace(0, warped.shape[1], 10, dtype=np.float32)

    # ── 3. Erase grid lines then slice cells ─────────────────────────────────
    warped_clean = remove_grid_lines_hough(warped)
    cells = slice_cells_from_lines(warped_clean, h_ys, v_xs)

    return cells, M, warped
