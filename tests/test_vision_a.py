# tests/test_vision_a.py
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from vision_a import preprocess_clahe


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
