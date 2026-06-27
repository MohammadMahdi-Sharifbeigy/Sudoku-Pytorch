"""YOLOv8-detect grid bounding-box detection + cell extraction."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import torch

from app.core.vision.common import (
    get_quadrilateral_points_in_order, perform_four_point_transform,
    locate_cells_within_grid, sort_cells_into_grid, build_grid_from_partial_cells,
    slice_grid_into_cells,
)

if TYPE_CHECKING:
    from ultralytics import YOLO


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_yolo_model(weights_path: str, device: torch.device) -> "YOLO":
    """Load YOLOv8-detect weights and move the model to the given device."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def _best_bbox(result, img_shape: tuple[int, int]) -> np.ndarray:
    """Extract the highest-confidence bounding box and return its 4 corners (px).

    Returns an array of shape (4, 2) with corners ordered TL, TR, BR, BL
    in pixel coordinates (not normalised), ready for four-point transform.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or not len(boxes):
        raise ValueError("YOLO-detect found no grid")

    conf = getattr(boxes, "conf", None)
    if conf is not None and len(conf):
        idx = int(np.argmax(_to_numpy(conf)))
    else:
        idx = 0

    # boxes.xyxy → (n, 4) absolute pixel coords [x1, y1, x2, y2]
    xyxy = _to_numpy(boxes.xyxy)
    if xyxy.ndim != 2 or xyxy.shape[0] == 0:
        raise ValueError("YOLO-detect found no grid")

    x1, y1, x2, y2 = xyxy[idx]

    # Clamp to image bounds
    H, W = img_shape[:2]
    x1 = float(np.clip(x1, 0, W))
    y1 = float(np.clip(y1, 0, H))
    x2 = float(np.clip(x2, 0, W))
    y2 = float(np.clip(y2, 0, H))

    # Return axis-aligned corners: TL, TR, BR, BL
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)


def get_grid_corners_yolo(img: np.ndarray, yolo_model, conf: float = 0.25) -> np.ndarray:
    """Detect the sudoku grid bounding box and return its four corners ordered."""
    results = yolo_model.predict(img, conf=conf, verbose=False)
    if not results:
        raise ValueError("YOLO-detect found no grid")
    corners = _best_bbox(results[0], img.shape)
    return get_quadrilateral_points_in_order(corners)


def get_cells_yolo(img: np.ndarray, yolo_model, grid_size: int = 576) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Warp the detected grid and slice it into 81 cell entries."""
    corners = get_grid_corners_yolo(img, yolo_model)
    M, grid_sq = perform_four_point_transform(img, corners, size=grid_size)
    cells = locate_cells_within_grid(grid_sq)
    if len(cells) == 81:
        return sort_cells_into_grid(cells), M, grid_sq
    if len(cells) >= 54:
        recovered = build_grid_from_partial_cells(cells)
        if recovered is not None:
            return recovered, M, grid_sq
    return slice_grid_into_cells(grid_sq), M, grid_sq
