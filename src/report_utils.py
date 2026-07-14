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
# Multi-task training report
# ──────────────────────────────────────────────

def save_training_report_multitask(
    history,
    test_digit_loss,
    test_digit_acc,
    test_lang_acc,
    digit_true, digit_pred,
    lang_true, lang_pred,
    lang_balance_info,
    dataset_mode,
    epochs,
    learning_rate,
    batch_size,
    best_val_loss,
    model,
    output_path="models/training_report_multitask.txt",
):
    """
    Write a multi-task training report.

    Parameters
    ----------
    history           : dict with keys 'Train Loss','Val Loss',
                        'Train Digit Acc','Val Digit Acc','Train Lang Acc','Val Lang Acc',
                        'Train Digit Loss','Train Lang Loss','Val Digit Loss','Val Lang Loss'
    digit_true/pred   : list[int]  full test set digit labels (includes empty=0)
    lang_true/pred    : list[int]  test set lang labels, non-empty cells only
                        (0=Persian, 1=English)
    lang_balance_info : {'train':…,'val':…,'test':…} from data_utils
    """
    from sklearn.metrics import classification_report, precision_recall_fscore_support

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    all_digit_labels = sorted(set(digit_true) | set(digit_pred))
    digit_cls_names  = {lbl: ("Empty" if lbl == 0 else str(lbl)) for lbl in all_digit_labels}

    digit_cm = confusion_matrix(digit_true, digit_pred, labels=all_digit_labels)
    lang_cm  = confusion_matrix(lang_true, lang_pred, labels=[0, 1])

    lines = []

    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN (MULTI-TASK) — TRAINING REPORT")
    lines.append(f"  Generated : {_timestamp()}")
    lines.append(_separator("="))

    # Config
    lines.append(_section("Training Configuration"))
    lines.append(f"  Dataset mode   : {dataset_mode}")
    lines.append(f"  Epochs         : {epochs}")
    lines.append(f"  Learning rate  : {learning_rate}")
    lines.append(f"  Batch size     : {batch_size}")
    lines.append(f"  Total params   : {total_params:,}")
    lines.append(f"  Trainable      : {trainable_params:,}")

    # Imbalance audit — self-documenting for any future reader
    lines.append(_section("Persian vs English Sample Counts  [imbalance audit]"))
    for sn in ('train', 'val', 'test'):
        b   = lang_balance_info.get(sn, {})
        flag = "  *** IMBALANCED (>60/40) ***" if b.get('imbalanced') else ""
        lines.append(
            f"  {sn:<6}: Persian={b.get('n_persian',0)} ({b.get('ratio_persian',0):.1f}%)  "
            f"English={b.get('n_english',0)} ({b.get('ratio_english',0):.1f}%){flag}"
        )
    lines.append("")
    lines.append("  Mitigation: inverse-frequency class weights on lang CrossEntropyLoss.")

    # Epoch history
    lines.append(_section("Epoch History"))
    lines.append(
        f"{'Ep':>4}  {'TotL':>8}  {'DL':>8}  {'LL':>8}"
        f"  {'DAcc':>7}  {'LAcc':>7}  {'VDL':>8}  {'VLL':>8}  {'VDAcc':>7}  {'VLAcc':>7}"
    )
    lines.append(_separator("-", 80))
    for i in range(len(history.get("Train Loss", []))):
        lines.append(
            f"  {i+1:>3}  {history['Train Loss'][i]:>8.5f}"
            f"  {history['Train Digit Loss'][i]:>8.5f}  {history['Train Lang Loss'][i]:>8.5f}"
            f"  {history['Train Digit Acc'][i]:>6.2f}%  {history['Train Lang Acc'][i]:>6.2f}%"
            f"  {history['Val Digit Loss'][i]:>8.5f}  {history['Val Lang Loss'][i]:>8.5f}"
            f"  {history['Val Digit Acc'][i]:>6.2f}%  {history['Val Lang Acc'][i]:>6.2f}%"
        )
    lines.append(f"\n  Best validation loss : {best_val_loss:.6f}")

    # Headline metrics
    lines.append(_section("Test Set Results — Headline Metrics"))
    lines.append(f"  Digit Accuracy   : {test_digit_acc:.4f}%")
    lines.append(f"  Language Accuracy: {test_lang_acc:.4f}%")

    # CRITICAL: digit accuracy broken down by language
    lines.append(_section("Digit Accuracy by True Language  [CRITICAL — Hoda is minority]"))
    lines.append("  Aggregate accuracy can mask failure on Persian digits (minority).")
    lines.append("")
    non_empty_dt = [dt for dt in digit_true if dt != 0]
    non_empty_dp = [dp for dt, dp in zip(digit_true, digit_pred) if dt != 0]
    lang_names   = {0: "Persian", 1: "English"}
    if len(non_empty_dt) == len(lang_true):
        for lid, lname in lang_names.items():
            idxs = [i for i, lt in enumerate(lang_true) if lt == lid]
            if not idxs:
                lines.append(f"  {lname}: no samples in test set")
                continue
            sub_dt = [non_empty_dt[i] for i in idxs]
            sub_dp = [non_empty_dp[i] for i in idxs]
            n_ok   = sum(t == p for t, p in zip(sub_dt, sub_dp))
            acc    = 100.0 * n_ok / len(sub_dt)
            lines.append(f"  {lname:>8} digits: {acc:.2f}%  ({n_ok}/{len(sub_dt)})")
    else:
        lines.append(f"  WARNING: alignment mismatch — "
                     f"non_empty_dt={len(non_empty_dt)}  lang_true={len(lang_true)}")

    # Language head report
    lines.append(_section("Language Head — Classification Report"))
    lines.append(classification_report(
        lang_true, lang_pred, labels=[0, 1],
        target_names=["Persian", "English"], digits=4,
    ))

    lines.append(_section("Language Head — Confusion Matrix (rows=True, cols=Predicted)"))
    lines.append("             Persian   English")
    lines.append("  " + _separator("-", 26))
    for i, nm in enumerate(["Persian", "English"]):
        lines.append(f"  {nm:>8} | {lang_cm[i,0]:>7}   {lang_cm[i,1]:>7}")

    # Digit head report
    lines.append(_section("Digit Head — Classification Report"))
    lines.append(classification_report(
        digit_true, digit_pred,
        labels=all_digit_labels,
        target_names=[digit_cls_names[l] for l in all_digit_labels],
        digits=4,
    ))

    lines.append(_section("Digit Head — Confusion Matrix"))
    col_w      = 7
    header_row = " " * 8 + "".join(f"{digit_cls_names[l]:>{col_w}}" for l in all_digit_labels)
    lines.append(header_row)
    lines.append("  " + _separator("-", max(0, len(header_row) - 2)))
    for i, tl in enumerate(all_digit_labels):
        row  = f"  {digit_cls_names[tl]:>5} |"
        row += "".join(f"{digit_cm[i,j]:>{col_w}}" for j in range(len(all_digit_labels)))
        lines.append(row)

    lines.append("\n" + _separator("="))
    lines.append("  END OF REPORT")
    lines.append(_separator("=") + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


# ──────────────────────────────────────────────
# Extended inference report (lang-aware)
# ──────────────────────────────────────────────

def save_inference_report(
    image_name,
    cells,
    per_cell_info,
    grid_array,
    solved_board,
    output_path="models/inference_report.txt",
    timing=None,
):
    """Write a per-cell extraction + prediction report.
    per_cell_info items: {'label','confidence','has_digit'} plus optional
    'lang_label' (0=Persian,1=English) and 'lang_confidence' when a
    multi-task model was used.
    timing: optional dict with per-stage seconds (extraction_s, prediction_s,
    solve_s, total_s) — appended as a Pipeline Timing section when provided.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    has_lang = any('lang_label' in info for info in per_cell_info)
    lang_map  = {0: "Persian", 1: "English"}

    lines = []

    lines.append(_separator("="))
    lines.append("  SUDOKU DIGIT CNN — INFERENCE REPORT")
    lines.append(f"  Generated  : {_timestamp()}")
    lines.append(f"  Image      : {image_name}")
    lines.append(_separator("="))

    lines.append(_section("Cell-by-Cell Extraction & Prediction"))
    if has_lang:
        lines.append(
            f"  {'Cell':>5}  {'Row':>4}  {'Col':>4}  {'Has Digit':>10}"
            f"  {'Predicted':>9}  {'Confidence':>11}  {'Language':>9}  {'LangConf':>9}  {'Shape':>8}"
        )
        lines.append("  " + _separator("-", 78))
    else:
        lines.append(
            f"  {'Cell':>5}  {'Row':>4}  {'Col':>4}  {'Has Digit':>10}"
            f"  {'Predicted':>9}  {'Confidence':>11}  {'Shape (HxW)':>12}"
        )
        lines.append("  " + _separator("-", 66))

    for idx, (cell, info) in enumerate(zip(cells, per_cell_info)):
        row = idx // 9
        col = idx % 9
        h, w      = cell['img'].shape[:2]
        has_digit = "Yes" if info['has_digit'] else "No"
        pred      = str(info['label']) if info['has_digit'] else "–"
        conf      = f"{info['confidence']*100:.1f}%" if info['has_digit'] else "–"

        if has_lang:
            ll   = info.get('lang_label')
            lc   = info.get('lang_confidence')
            lnam = lang_map.get(ll, "–") if ll is not None else "–"
            lcon = f"{lc*100:.1f}%" if lc is not None else "–"
            lines.append(
                f"  {idx+1:>5}  {row:>4}  {col:>4}  {has_digit:>10}"
                f"  {pred:>9}  {conf:>11}  {lnam:>9}  {lcon:>9}  {h}x{w}"
            )
        else:
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

    n_digits = sum(1 for i in per_cell_info if i['has_digit'])
    n_empty  = 81 - n_digits
    if n_digits > 0:
        confs         = [i['confidence'] for i in per_cell_info if i['has_digit']]
        avg_conf      = np.mean(confs) * 100
        min_conf      = np.min(confs) * 100
        low_conf_cells = [(i, per_cell_info[i]) for i in range(81)
                          if per_cell_info[i]['has_digit']
                          and per_cell_info[i]['confidence'] < 0.50]
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

    if timing is not None:
        lines.append(_section("Pipeline Timing  [deploy latency]"))
        _fmt = lambda s: f"{s*1000:8.1f} ms  ({s:.3f} s)"
        lines.append(f"  Grid extraction : {_fmt(timing.get('extraction_s', 0.0))}")
        lines.append(f"  Digit prediction: {_fmt(timing.get('prediction_s', 0.0))}")
        lines.append(f"  Solving         : {_fmt(timing.get('solve_s', 0.0))}")
        lines.append("  " + _separator("-", 44))
        lines.append(f"  Total (upload→solve): {_fmt(timing.get('total_s', 0.0))}")

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
