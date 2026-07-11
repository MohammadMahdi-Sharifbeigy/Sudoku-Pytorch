"""
vision_a.py — Canny-first Sudoku pipeline with HoughLinesP grid refinement.

Pipeline:
  Pre-process : CLAHE + GaussianBlur(7,7)
  Grid detect : grayscale → Canny(50,150) → dilate → biggest 4-point contour
  Warp        : four-point perspective to 576×576 square
  Refine      : HoughLinesP → cluster into 10H + 10V lines → exact cell bounds
  Line erase  : paint detected lines black (thickness=13)
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
