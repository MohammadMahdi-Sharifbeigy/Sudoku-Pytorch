"""
report_utils.py
Generates plain-text report files summarising model training results
and per-image inference details.
"""

import os
import datetime
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from typing import Any


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _separator(char: str = "=", width: int = 60) -> str:
    return char * width


def _section(title: str, char: str = "=", width: int = 60) -> str:
    bar = _separator(char, width)
    return f"\n{bar}\n  {title}\n{bar}\n"


# ──────────────────────────────────────────────
# Training report
# ──────────────────────────────────────────────

def save_training_report(
    history: dict[str, list[float]],
    test_loss: float,
    test_acc: float,
    y_true: list[int],
    y_pred: list[int],
    dataset_mode: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    best_val_loss: float,
    model: nn.Module,
    output_path: str = "models/training_report.txt",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_labels = sorted(set(y_true) | set(y_pred))
    class_names = {lbl: ("Empty" if lbl == 0 else str(lbl)) for lbl in all_labels}

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    target_names = [class_names[l] for l in all_labels]
    clf_report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=target_names, digits=4
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines: list[str] = []

    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN – TRAINING REPORT")
    lines.append(f"  Generated : {_timestamp()}")
    lines.append(_separator("="))

    lines.append(_section("Training Configuration"))
    lines.append(f"  Dataset mode   : {dataset_mode}")
    lines.append(f"  Epochs         : {epochs}")
    lines.append(f"  Learning rate  : {learning_rate}")
    lines.append(f"  Batch size     : {batch_size}")
    lines.append(f"  Total params   : {total_params:,}")
    lines.append(f"  Trainable      : {trainable_params:,}")

    lines.append(_section("Epoch History"))
    header = f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Train Acc':>10}  {'Val Acc':>8}"
    lines.append(header)
    lines.append(_separator("-", width=60))
    for i, (tl, vl, ta, va) in enumerate(zip(
        history["Train Loss"], history["Val Loss"],
        history["Train Acc"],  history["Val Acc"]
    ), start=1):
        lines.append(f"  {i:4d}  {tl:11.6f}  {vl:9.6f}  {ta:9.4f}%  {va:7.4f}%")

    lines.append("")
    lines.append(f"  Best validation loss : {best_val_loss:.6f}")

    lines.append(_section("Test Set Results"))
    lines.append(f"  Test Loss     : {test_loss:.6f}")
    lines.append(f"  Test Accuracy : {test_acc:.4f}%")

    lines.append(_section("Per-Class Classification Report"))
    lines.append(clf_report)

    lines.append(_section("Confusion Matrix (rows=True, cols=Predicted)"))

    col_w = 7
    header_row = " " * 8 + "".join(f"{class_names[l]:>{col_w}}" for l in all_labels)
    lines.append(header_row)
    lines.append("  " + _separator("-", width=max(0, len(header_row) - 2)))
    for i, true_lbl in enumerate(all_labels):
        row_label = f"  {class_names[true_lbl]:>5} |"
        row_vals = "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(all_labels)))
        lines.append(row_label + row_vals)

    lines.append("")
    lines.append("  (Diagonal = correct predictions; off-diagonal = confusion pairs)")

    lines.append(_section("Per-Class Accuracy"))
    header = f"  {'Class':>8}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}"
    lines.append(header)
    lines.append("  " + _separator("-", 46))
    per_class_correct = cm.diagonal()
    per_class_total   = cm.sum(axis=1)
    for i, lbl in enumerate(all_labels):
        n_corr = per_class_correct[i]
        n_tot  = per_class_total[i]
        acc    = 100.0 * n_corr / n_tot if n_tot > 0 else 0.0
        lines.append(f"  {class_names[lbl]:>8}  {n_corr:>8}  {n_tot:>7}  {acc:>8.2f}%")

    lines.append("\n" + _separator("="))
    lines.append("  END OF REPORT")
    lines.append(_separator("=") + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ──────────────────────────────────────────────
# Inference report
# ──────────────────────────────────────────────

def save_inference_report(
    image_name: str,
    cells: list[dict[str, Any]],
    per_cell_info: list[dict[str, Any]],
    grid_array: Any,
    solved_board: Any,
    output_path: str = "models/inference_report.txt",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines: list[str] = []

    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN – INFERENCE REPORT")
    lines.append(f"  Generated  : {_timestamp()}")
    lines.append(f"  Image      : {image_name}")
    lines.append(_separator("="))

    lines.append(_section("Cell-by-Cell Extraction & Prediction"))
    lines.append(
        f"  {'Cell':>5}  {'Row':>4}  {'Col':>4}  {'Has Digit':>10}"
        f"  {'Predicted':>9}  {'Confidence':>11}  {'Shape (HxW)':>12}"
    )
    lines.append("  " + _separator("-", 66))

    for idx, (cell, info) in enumerate(zip(cells, per_cell_info)):
        row = idx // 9
        col = idx % 9
        h, w = cell['img'].shape[:2]
        has_digit = "Yes" if info['has_digit'] else "No"
        pred = str(info['label']) if info['has_digit'] else "–"
        conf = f"{info['confidence']*100:.1f}%" if info['has_digit'] else "–"
        lines.append(
            f"  {idx+1:>5}  {row:>4}  {col:>4}  {has_digit:>10}"
            f"  {pred:>9}  {conf:>11}  {h}x{w}"
        )

        if col == 8 and row in (2, 5):
            lines.append("")

    lines.append(_section("Raw Predicted Grid (0 = empty)"))
    lines.append(_format_grid(grid_array))

    if solved_board is not None:
        lines.append(_section("Solved Grid"))
        lines.append(_format_grid(solved_board))
    else:
        lines.append(_section("Solved Grid"))
        lines.append("  *** Grid could not be solved. ***")
        lines.append("  Check the raw predicted grid above for misread digits.")

    n_digits  = sum(1 for i in per_cell_info if i['has_digit'])
    n_empty   = 81 - n_digits
    if n_digits > 0:
        confs = [i['confidence'] for i in per_cell_info if i['has_digit']]
        avg_conf = float(np.mean(confs)) * 100
        min_conf = float(np.min(confs)) * 100
        low_conf_cells = [(idx, per_cell_info[idx]) for idx in range(81)
                          if per_cell_info[idx]['has_digit']
                          and per_cell_info[idx]['confidence'] < 0.50]
    else:
        avg_conf = min_conf = 0.0
        low_conf_cells = []

    lines.append(_section("Prediction Summary"))
    lines.append(f"  Detected digit cells : {n_digits}")
    lines.append(f"  Empty cells          : {n_empty}")
    if n_digits > 0:
        lines.append(f"  Average confidence   : {avg_conf:.1f}%")
        lines.append(f"  Minimum confidence   : {min_conf:.1f}%")
        lines.append(f"  Low-confidence cells : {len(low_conf_cells)} (<50%)")
        if low_conf_cells:
            lines.append("")
            lines.append("  Low-confidence cell details:")
            for idx, info in low_conf_cells:
                r, c = idx // 9, idx % 9
                lines.append(
                    f"    Cell {idx+1:2d} (row {r}, col {c})"
                    f"  predicted={info['label']}  conf={info['confidence']*100:.1f}%"
                )

    lines.append("\n" + _separator("="))
    lines.append("  END OF REPORT")
    lines.append(_separator("=") + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ──────────────────────────────────────────────
# Grid formatter (shared)
# ──────────────────────────────────────────────

def _format_grid(grid: Any) -> str:
    """Returns a nicely formatted 9×9 Sudoku grid string."""
    lines: list[str] = []
    for r in range(9):
        if r in (3, 6):
            lines.append("  ------+-------+------")
        row_str = ""
        for c in range(9):
            if c in (3, 6):
                row_str += " | "
            val = grid[r][c] if grid[r][c] != 0 else "."
            row_str += f" {val}"
        lines.append("  " + row_str.strip())
    return "\n".join(lines)
