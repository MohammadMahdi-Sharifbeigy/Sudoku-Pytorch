"""Offline debug output helpers."""
import json
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.vision import plot_cell_images_in_grid


def save_debug_outputs(
    image_name: str,
    img_rgb: np.ndarray,
    cells_selected,
    board_selected,
    pipeline_name: str,
    error: str = None,
    out_root: str = "debug_outputs",
) -> str:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem    = os.path.splitext(image_name)[0]
    out_dir = os.path.join(out_root, f"{ts}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(out_dir, "00_original.jpg"),
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
    )

    if board_selected is not None:
        path = os.path.join(out_dir, "selected_warped_grid.jpg")
        if len(board_selected.shape) == 3:
            cv2.imwrite(path, cv2.cvtColor(board_selected, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, board_selected)

    if cells_selected is not None:
        fig = plot_cell_images_in_grid(cells_selected)
        fig.savefig(os.path.join(out_dir, "selected_cells_mosaic.png"),
                    dpi=80, bbox_inches="tight")
        plt.close(fig)

        cells_dir = os.path.join(out_dir, "selected_cells")
        os.makedirs(cells_dir, exist_ok=True)
        for c in cells_selected:
            r, col = c['grid_row'], c['grid_col']
            flag   = "D" if c['contains_digit'] else "E"
            cv2.imwrite(os.path.join(cells_dir, f"r{r}c{col}_{flag}.png"), c['img'])

    summary = {
        "image":       image_name,
        "pipeline":    pipeline_name,
        "success":     cells_selected is not None,
        "error":       error,
        "digit_count": sum(c['contains_digit'] for c in cells_selected) if cells_selected else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return out_dir
