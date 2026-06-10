"""
report_utils.py
Generates plain-text report files summarising model training results
and per-image inference details.
"""

import os
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _separator(char="=", width=60):
    return char * width


def _section(title, char="=", width=60):
    bar = _separator(char, width)
    return f"\n{bar}\n  {title}\n{bar}\n"


# ──────────────────────────────────────────────
# Training report
# ──────────────────────────────────────────────

def save_training_report(
    history,
    test_loss,
    test_acc,
    y_true,
    y_pred,
    dataset_mode,
    epochs,
    learning_rate,
    batch_size,
    best_val_loss,
    model,
    output_path="models/training_report.txt",
):
    """
    Write a comprehensive training report to *output_path*.

    Parameters
    ----------
    history       : dict with keys 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'
    test_loss     : float
    test_acc      : float (percentage, 0-100)
    y_true        : list[int] – ground-truth labels from test set
    y_pred        : list[int] – predicted labels from test set
    dataset_mode  : str – description of the training dataset used
    epochs        : int
    learning_rate : float
    batch_size    : int
    best_val_loss : float
    model         : nn.Module – for parameter count
    output_path   : str
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Collect unique labels and build human-readable class names
    all_labels = sorted(set(y_true) | set(y_pred))
    class_names = {lbl: ("Empty" if lbl == 0 else str(lbl)) for lbl in all_labels}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # sklearn classification report
    target_names = [class_names[l] for l in all_labels]
    clf_report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=target_names, digits=4
    )

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = []

    # ── Header ──────────────────────────────────
    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN – TRAINING REPORT")
    lines.append(f"  Generated : {_timestamp()}")
    lines.append(_separator("="))

    # ── Configuration ───────────────────────────
    lines.append(_section("Training Configuration"))
    lines.append(f"  Dataset mode   : {dataset_mode}")
    lines.append(f"  Epochs         : {epochs}")
    lines.append(f"  Learning rate  : {learning_rate}")
    lines.append(f"  Batch size     : {batch_size}")
    lines.append(f"  Total params   : {total_params:,}")
    lines.append(f"  Trainable      : {trainable_params:,}")

    # ── Epoch-by-epoch history ───────────────────
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

    # ── Test set summary ─────────────────────────
    lines.append(_section("Test Set Results"))
    lines.append(f"  Test Loss     : {test_loss:.6f}")
    lines.append(f"  Test Accuracy : {test_acc:.4f}%")

    # ── Classification report ────────────────────
    lines.append(_section("Per-Class Classification Report"))
    lines.append(clf_report)

    # ── Confusion matrix (text) ──────────────────
    lines.append(_section("Confusion Matrix (rows=True, cols=Predicted)"))

    # Column header
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

    # ── Per-class accuracy table ─────────────────
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

    # ── Footer ───────────────────────────────────
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
    image_name,
    cells,
    per_cell_info,
    grid_array,
    solved_board,
    output_path="models/inference_report.txt",
):
    """
    Write a per-cell extraction + prediction report to *output_path*.

    Parameters
    ----------
    image_name     : str – filename / label for the uploaded image
    cells          : list[dict] – sorted cell dicts from vision pipeline
    per_cell_info  : list[dict] – {'label', 'confidence', 'has_digit'} per cell
    grid_array     : np.ndarray shape (9,9) – raw predicted grid
    solved_board   : np.ndarray shape (9,9) or None – solved grid, None if unsolvable
    output_path    : str
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = []

    # ── Header ──────────────────────────────────
    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN – INFERENCE REPORT")
    lines.append(f"  Generated  : {_timestamp()}")
    lines.append(f"  Image      : {image_name}")
    lines.append(_separator("="))

    # ── Cell-by-cell table ───────────────────────
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

        # Blank row between Sudoku box rows (every 3rd row)
        if col == 8 and row in (2, 5):
            lines.append("")

    # ── Raw predicted grid ───────────────────────
    lines.append(_section("Raw Predicted Grid (0 = empty)"))
    lines.append(_format_grid(grid_array))

    # ── Solved grid ──────────────────────────────
    if solved_board is not None:
        lines.append(_section("Solved Grid"))
        lines.append(_format_grid(solved_board))
    else:
        lines.append(_section("Solved Grid"))
        lines.append("  *** Grid could not be solved. ***")
        lines.append("  Check the raw predicted grid above for misread digits.")

    # ── Summary statistics ───────────────────────
    n_digits  = sum(1 for i in per_cell_info if i['has_digit'])
    n_empty   = 81 - n_digits
    if n_digits > 0:
        confs = [i['confidence'] for i in per_cell_info if i['has_digit']]
        avg_conf = np.mean(confs) * 100
        min_conf = np.min(confs) * 100
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

    # ── Footer ───────────────────────────────────
    lines.append("\n" + _separator("="))
    lines.append("  END OF REPORT")
    lines.append(_separator("=") + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ──────────────────────────────────────────────
# Grid formatter (shared)
# ──────────────────────────────────────────────

def _format_grid(grid):
    """Returns a nicely formatted 9×9 Sudoku grid string."""
    lines = []
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
