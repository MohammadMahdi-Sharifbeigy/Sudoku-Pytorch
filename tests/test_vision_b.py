# tests/test_vision_b.py
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from vision_b import refine_corners_hough_b, remove_grid_lines_hough_b, slice_cells_from_lines_b


def _warped_grid_bgr(size=576, n=9):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    step = size // n
    for i in range(n + 1):
        pos = i * step
        cv2.line(img, (0, pos), (size, pos), (180, 180, 180), 2)
        cv2.line(img, (pos, 0), (pos, size), (180, 180, 180), 2)
    return img


class TestVisionBHough:
    def test_refine_blank_returns_none(self):
        img = np.zeros((576, 576, 3), dtype=np.uint8)
        assert refine_corners_hough_b(img) is None

    def test_remove_returns_grayscale(self):
        img = _warped_grid_bgr()
        result = remove_grid_lines_hough_b(img)
        assert len(result.shape) == 2

    def test_slice_returns_81(self):
        img = np.zeros((576, 576), dtype=np.uint8)
        h_ys = np.linspace(0, 576, 10, dtype=np.float32)
        v_xs = np.linspace(0, 576, 10, dtype=np.float32)
        cells = slice_cells_from_lines_b(img, h_ys, v_xs)
        assert len(cells) == 81

    def test_get_valid_cells_api_exists(self):
        from vision_b import get_valid_cells_from_image
        assert callable(get_valid_cells_from_image)
