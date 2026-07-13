"""
vision_c.py — Pipeline C: vision.py grid detection + B's cell extraction.

Hybrid strategy:
  Grid detection  : vision.py multi-combo NMS approach (robust to varied lighting)
  Cell extraction : vision_b.py line-removal + local quad curves + _find_digit_mask
                    (better than contour NMS for cell interiors)

Public API matches vision.py:
  get_valid_cells_from_image_c(img_rgb, ...) → (cells, M, warped)
"""

from __future__ import annotations

import cv2
import numpy as np

from src.vision import (
    find_grid_contour_candidates,
    apply_grayscale_blur_and_threshold,
    sharpen_image,
    DEFAULT_GRID_THRESHOLD_COMBOS,
)
from src.vision_b import (
    _process_warp,
    _to_28,
    WARP_SIZE,
    _order_corners,
    _warp_destination,
    _is_light_on_dark,
    _check_stop,
)


def get_valid_cells_from_image_c(
    img_rgb: np.ndarray,
    grid_size: int = WARP_SIZE,
    grid_threshold_combos=None,
    blur_k: int = 3,
    area_threshold: float = 4.0,
    erode_enabled: bool = True,
    contour_erode_kernel_size: int = 3,
    contour_erode_iterations: int = 1,
    slice_erode_kernel_size: int = 2,
    slice_erode_iterations: int = 3,
    digit_scale: int = 20,
    should_stop=None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Pipeline C: vision.py grid detection → vision_b.py cell extraction.

    Parameters match vision.py's get_valid_cells_from_image for drop-in use.
    Returns (cells, M, warped) in vision.py dict format.
    """
    _check_stop(should_stop)

    combos = grid_threshold_combos or DEFAULT_GRID_THRESHOLD_COMBOS

    # ── Step 1: vision.py multi-combo grid detection ──────────────────────────
    M_matrices, warped_images, contour_list = find_grid_contour_candidates(
        img_rgb, threshold_combos=combos, blur_k=blur_k, should_stop=should_stop)

    if not warped_images:
        raise Exception(
            "Pipeline C: no grid boundary detected. "
            "Make sure the Sudoku grid is clearly visible."
        )

    _check_stop(should_stop)

    # ── Step 2: For each candidate warp, run B's cell extraction ─────────────
    # Pick the candidate that yields the most non-empty cells.
    best_cells, best_M, best_warped = None, None, None
    best_count = -1

    for i, raw_warped in enumerate(warped_images):
        _check_stop(should_stop)

        # Re-warp to square WARP_SIZE so B's line-removal maths holds
        from src.vision import _contour_to_quad, perform_four_point_transform
        import cv2 as _cv2
        contour = contour_list[i]
        perimeter = _cv2.arcLength(contour, True)
        approx    = _cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        pts = (np.squeeze(approx, axis=1).astype(np.float32)
               if len(approx) == 4 else _contour_to_quad(contour))

        M_sq, warped_sq = perform_four_point_transform(img_rgb, pts, size=WARP_SIZE)

        # Pipeline B needs a greyscale, CLAHE-normalised warp
        gray_sq = (cv2.cvtColor(warped_sq, cv2.COLOR_RGB2GRAY)
                   if warped_sq.ndim == 3 else warped_sq)
        if _is_light_on_dark(gray_sq):
            gray_sq = cv2.bitwise_not(gray_sq)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_sq = clahe.apply(gray_sq)

        try:
            raw_cells = _process_warp(norm_sq, should_stop=should_stop)
        except Exception:
            continue

        count = sum(not c.is_empty for c in raw_cells)
        if count > best_count:
            best_count  = count
            best_cells  = raw_cells
            best_M      = M_sq
            best_warped = norm_sq

    if best_cells is None:
        raise Exception("Pipeline C: cell extraction failed for all grid candidates.")

    # Convert to vision.py dict format
    cw = WARP_SIZE / 9.0
    ch = WARP_SIZE / 9.0
    cells = []
    for idx, cell in enumerate(best_cells):
        r, c = divmod(idx, 9)
        cells.append({
            'img':            _to_28(cell.image),
            'contains_digit': not cell.is_empty,
            'grid_row':       r,
            'grid_col':       c,
            'x_centroid':     int((c + 0.5) * cw),
            'y_centroid':     int((r + 0.5) * ch),
        })

    return cells, best_M, best_warped
