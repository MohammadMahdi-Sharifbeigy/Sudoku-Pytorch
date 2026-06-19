"""Classical contour-based Sudoku grid detection."""
from typing import Optional
import cv2
import numpy as np
import imutils
from app.core.vision.common import (
    apply_grayscale_blur_and_threshold, perform_four_point_transform,
    _contour_to_quad, _nms, locate_cells_within_grid, sort_cells_into_grid,
    build_grid_from_partial_cells, slice_grid_into_cells,
)


# ─────────────────────────────────────────────────────────────────────────────
# Grid boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def find_grid_contour_candidates(
    img: np.ndarray,
) -> tuple[Optional[list], Optional[list], Optional[list]]:
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


def get_cells_classical(
    img: np.ndarray, grid_size: int = 576
) -> tuple[list[dict], np.ndarray, np.ndarray]:
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
        raise ValueError(
            "No grid boundary detected. Make sure the Sudoku grid is clearly "
            "visible and fills most of the image."
        )

    # ── 1. Contour path ──────────────────────────────────────────────────
    best = None  # (n_cells, cells, M, grid_image, contour)
    for i, grid_image in enumerate(warped_images):
        cells = locate_cells_within_grid(grid_image)
        if len(cells) == 81:
            return sort_cells_into_grid(cells), M_matrices[i], grid_image
        if best is None or len(cells) > best[0]:
            best = (len(cells), cells, M_matrices[i], grid_image, contour_list[i])

    # ── 2. Partial-grid recovery ─────────────────────────────────────────
    best_n, best_cells, best_M, best_grid, best_contour = best  # type: ignore[misc]
    if best_n >= 54:
        recovered = build_grid_from_partial_cells(best_cells)
        if recovered is not None:
            return recovered, best_M, best_grid

    # ── 3. Slice fallback ────────────────────────────────────────────────
    perimeter = cv2.arcLength(best_contour, True)
    approx    = cv2.approxPolyDP(best_contour, 0.03 * perimeter, True)
    pts = (np.squeeze(approx, axis=1).astype(np.float32)
           if len(approx) == 4 else _contour_to_quad(best_contour))

    M_sq, grid_sq = perform_four_point_transform(img, pts, size=grid_size)
    cells = slice_grid_into_cells(grid_sq)
    return cells, M_sq, grid_sq


get_valid_cells_from_image = get_cells_classical
