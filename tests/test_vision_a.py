# tests/test_vision_a.py
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from vision_a import preprocess_clahe, detect_grid_canny, refine_corners_hough, remove_grid_lines_hough, slice_cells_from_lines


def _blank_bgr(h=500, w=500):
    return np.zeros((h, w, 3), dtype=np.uint8)


# Shared fixtures — used by TestDetectGridCanny, TestRefineCorners, etc. below
def _square_bgr(h=500, w=500, margin=50):
    """Black image with white rectangle — simulates a sudoku grid boundary."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (255, 255, 255), 3)
    return img


def _warped_grid_bgr(size=576, n=9):
    """Black BGR image with n+1 gray horizontal and vertical lines — simulates warped grid."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    step = size // n
    for i in range(n + 1):
        pos = i * step
        cv2.line(img, (0, pos), (size, pos), (180, 180, 180), 2)
        cv2.line(img, (pos, 0), (pos, size), (180, 180, 180), 2)
    return img


class TestPreprocessClahe:
    def test_returns_same_shape(self):
        img = _blank_bgr()
        result = preprocess_clahe(img)
        assert result.shape == img.shape

    def test_output_is_bgr_three_channel(self):
        img = _blank_bgr()
        result = preprocess_clahe(img)
        assert len(result.shape) == 3 and result.shape[2] == 3

    def test_dtype_uint8(self):
        img = _blank_bgr(300, 300)
        result = preprocess_clahe(img)
        assert result.dtype == np.uint8

    def test_clahe_modifies_nontrivial_image(self):
        img = _square_bgr()
        result = preprocess_clahe(img)
        assert not np.array_equal(result, img)


class TestDetectGridCanny:
    def test_blank_image_returns_none(self):
        img = _blank_bgr()
        assert detect_grid_canny(img) is None

    def test_square_returns_four_corners(self):
        img = _square_bgr()
        pts = detect_grid_canny(img)
        assert pts is not None, "Should detect 4 corners of the white square"
        assert pts.shape == (4, 2)

    def test_corners_ordered_tl_tr_br_bl(self):
        img = _square_bgr(h=500, w=500, margin=50)
        pts = detect_grid_canny(img)
        assert pts is not None
        tl, tr, br, bl = pts
        assert tl[0] < tr[0], "TL x < TR x"
        assert bl[0] < br[0], "BL x < BR x"
        assert tl[1] < bl[1], "TL y < BL y"
        assert tr[1] < br[1], "TR y < BR y"


class TestRefineCorners:
    def test_blank_returns_none(self):
        img = _blank_bgr()
        assert refine_corners_hough(img) is None

    def test_grid_returns_ten_h_ten_v(self):
        img = _warped_grid_bgr()
        result = refine_corners_hough(img)
        assert result is not None, "Should find lines in synthetic grid"
        h_ys, v_xs = result
        assert len(h_ys) == 10, f"Expected 10 H lines, got {len(h_ys)}"
        assert len(v_xs) == 10, f"Expected 10 V lines, got {len(v_xs)}"

    def test_h_ys_sorted_ascending(self):
        img = _warped_grid_bgr()
        result = refine_corners_hough(img)
        if result is None:
            return
        h_ys, _ = result
        assert list(h_ys) == sorted(h_ys)

    def test_v_xs_sorted_ascending(self):
        img = _warped_grid_bgr()
        result = refine_corners_hough(img)
        if result is None:
            return
        _, v_xs = result
        assert list(v_xs) == sorted(v_xs)


class TestRemoveGridLines:
    def test_returns_2d_grayscale(self):
        img = _warped_grid_bgr()
        result = remove_grid_lines_hough(img)
        assert len(result.shape) == 2, "Must return 2D grayscale"
        assert result.shape == img.shape[:2]

    def test_dtype_uint8(self):
        img = _warped_grid_bgr()
        result = remove_grid_lines_hough(img)
        assert result.dtype == np.uint8


class TestSliceCellsFromLines:
    def _make_clean_gray(self, size=576):
        return np.zeros((size, size), dtype=np.uint8)

    def _make_uniform_lines(self, size=576, n=9):
        h_ys = np.linspace(0, size, n + 1, dtype=np.float32)
        v_xs = np.linspace(0, size, n + 1, dtype=np.float32)
        return h_ys, v_xs

    def test_returns_81_cells(self):
        img = self._make_clean_gray()
        h_ys, v_xs = self._make_uniform_lines()
        cells = slice_cells_from_lines(img, h_ys, v_xs)
        assert len(cells) == 81

    def test_cell_dict_keys(self):
        img = self._make_clean_gray()
        h_ys, v_xs = self._make_uniform_lines()
        cell = slice_cells_from_lines(img, h_ys, v_xs)[0]
        for key in ('img', 'contains_digit', 'grid_row', 'grid_col', 'x_centroid', 'y_centroid'):
            assert key in cell, f"Missing key: {key}"

    def test_cell_img_shape_28x28(self):
        img = self._make_clean_gray()
        h_ys, v_xs = self._make_uniform_lines()
        cells = slice_cells_from_lines(img, h_ys, v_xs)
        for cell in cells:
            assert cell['img'].shape == (28, 28), f"Cell img shape {cell['img'].shape}"

    def test_empty_cells_contain_digit_false(self):
        img = self._make_clean_gray()
        h_ys, v_xs = self._make_uniform_lines()
        cells = slice_cells_from_lines(img, h_ys, v_xs)
        assert all(not c['contains_digit'] for c in cells)

    def test_grid_row_col_range(self):
        img = self._make_clean_gray()
        h_ys, v_xs = self._make_uniform_lines()
        cells = slice_cells_from_lines(img, h_ys, v_xs)
        rows = [c['grid_row'] for c in cells]
        cols = [c['grid_col'] for c in cells]
        assert min(rows) == 0 and max(rows) == 8
        assert min(cols) == 0 and max(cols) == 8
