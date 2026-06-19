"""YOLOv8-pose grid corner detection — implemented in Phase 4."""
from __future__ import annotations
import numpy as np
import torch


def load_yolo_model(weights_path: str, device: torch.device):
    raise NotImplementedError


def get_grid_corners_yolo(img: np.ndarray, yolo_model, conf: float = 0.25) -> np.ndarray:
    raise NotImplementedError


def get_cells_yolo(img: np.ndarray, yolo_model, grid_size: int = 576):
    raise NotImplementedError
