"""YOLOv8-pose grid corner detection + cell extraction."""
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
    """Load YOLOv8-pose weights and move the model to the given device."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def _best_keypoints(result) -> np.ndarray:
    kpts = getattr(result, "keypoints", None)
    if kpts is None or getattr(kpts, "xy", None) is None:
        raise ValueError("YOLO-pose found no grid")
    xy = _to_numpy(kpts.xy)                 # (n_instances, n_kpts, 2)
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] < 4:
        raise ValueError("YOLO-pose found no grid")
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
        idx = int(np.argmax(_to_numpy(boxes.conf)))
    else:
        idx = 0
    idx = idx if idx < xy.shape[0] else 0
    return xy[idx][:4].astype(np.float32)  # (4, 2)


def get_grid_corners_yolo(img: np.ndarray, yolo_model, conf: float = 0.25) -> np.ndarray:
    """Detect the sudoku grid's four corner keypoints and return them ordered."""
    results = yolo_model.predict(img, conf=conf, verbose=False)
    if not results:
        raise ValueError("YOLO-pose found no grid")
    corners = _best_keypoints(results[0])
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
