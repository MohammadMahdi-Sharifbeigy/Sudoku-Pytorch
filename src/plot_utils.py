"""Paper-style (white background) Matplotlib + pandas plot helpers.

These are the *saved-to-disk* artifacts written after a full training run —
distinct from app/plot_utils.py's dark-themed *live-preview* charts used
while training is running in the Streamlit UI. This module has no Streamlit
dependency so it works identically from run_training_cli.py and from the
Streamlit training page (both call src.train_runner.run_training, which is
where generate_training_plots() is invoked).

Color choices (validated colorblind-safe categorical pair + single-hue
sequential ramp, per the project's dataviz palette):
  Train = blue  #2a78d6      Val = red  #e34948
  Confusion matrix = single-hue sequential ("Blues" colormap)
"""
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

_SURFACE   = '#fcfcfb'
_INK       = '#0b0b0b'
_INK_SOFT  = '#52514e'
_INK_MUTED = '#898781'
_GRID      = '#e1e0d9'
_AXIS      = '#c3c2b7'
_C_TRAIN   = '#2a78d6'
_C_VAL     = '#e34948'


def _save_clean(fig, save_path: str) -> str:
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.08, facecolor=_SURFACE)
    plt.close(fig)
    return save_path


def _paper_style(ax, title: str, ylabel: str = '', xlabel: str = 'Epoch') -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, color=_INK, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=_INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_SOFT, fontsize=10)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_AXIS)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(_INK_SOFT)
    ax.grid(True, alpha=0.6, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _epochs(history: dict) -> list:
    return list(range(1, len(history.get('Train Loss', history.get('Train Digit Loss', []))) + 1))


def build_class_names(all_labels: list, unified: bool = False) -> list:
    """Human-readable class names for a sorted list of label ids.

    unified=True splits the 20-class unified-model label space at class 10
    into ENG_n (0-9) / PER_n (10-19); otherwise 0 -> "Empty", else str(digit).
    """
    if unified:
        def _n(c):
            return ("ENG_0" if c == 0 else f"ENG_{c}" if c < 10
                    else "PER_0" if c == 10 else f"PER_{c - 10}")
        return [_n(c) for c in all_labels]
    return ["Empty" if l == 0 else str(l) for l in all_labels]


def save_learning_curve(history: dict, save_path: str) -> str:
    """Train Loss vs Val Loss over epoch — paper style."""
    df = pd.DataFrame(history)
    epochs = _epochs(history)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    ax.plot(epochs, df['Train Loss'], color=_C_TRAIN, linewidth=2,
            marker='o', markersize=4, markevery=max(1, len(epochs) // 12), label='Train')
    ax.plot(epochs, df['Val Loss'], color=_C_VAL, linewidth=2,
            marker='s', markersize=4, markevery=max(1, len(epochs) // 12), label='Validation')
    _paper_style(ax, 'Learning Curve', ylabel='Loss')
    ax.legend(frameon=False, labelcolor=_INK, fontsize=9, loc='best')
    return _save_clean(fig, save_path)


def save_train_history(history: dict, save_path: str) -> str:
    """Train/Val Accuracy over epoch — paper style.

    Auto-detects multi-task keys ('Train Digit Acc'/'Train Lang Acc') and
    renders a second panel for language accuracy when present.
    """
    df = pd.DataFrame(history)
    epochs = _epochs(history)
    is_multitask = 'Train Digit Acc' in df.columns

    if is_multitask:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        panels = [
            (axes[0], 'Train Digit Acc', 'Val Digit Acc', 'Digit Accuracy'),
            (axes[1], 'Train Lang Acc',  'Val Lang Acc',  'Language Accuracy'),
        ]
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        panels = [(ax, 'Train Acc', 'Val Acc', 'Training History')]

    fig.patch.set_facecolor(_SURFACE)
    for ax, t_key, v_key, title in panels:
        ax.plot(epochs, df[t_key], color=_C_TRAIN, linewidth=2,
                marker='o', markersize=4, markevery=max(1, len(epochs) // 12), label='Train')
        ax.plot(epochs, df[v_key], color=_C_VAL, linewidth=2,
                marker='s', markersize=4, markevery=max(1, len(epochs) // 12), label='Validation')
        _paper_style(ax, title, ylabel='Accuracy (%)')
        ax.legend(frameon=False, labelcolor=_INK, fontsize=9, loc='best')

    return _save_clean(fig, save_path)


def save_confusion_matrix(y_true, y_pred, class_names, save_path: str,
                           title: str = 'Confusion Matrix — Test Set') -> str:
    """Paper-style confusion matrix (single-hue sequential colormap)."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum
    tick_size = 8 if n > 12 else 9
    diag_size = 7 if n > 12 else 8
    offdiag_size = 6 if n > 12 else 7

    fig, ax = plt.subplots(
        figsize=(max(7.2, n * 0.72), max(6.2, n * 0.62)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_INK_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_INK_SOFT)
    cbar.set_label('Row-normalized accuracy', color=_INK_SOFT, fontsize=9)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, color=_INK_SOFT, fontsize=tick_size, rotation=45, ha='right')
    ax.set_yticklabels(class_names, color=_INK_SOFT, fontsize=tick_size)
    ax.set_xlabel('Predicted', color=_INK, fontsize=11, labelpad=10)
    ax.set_ylabel('True', color=_INK, fontsize=11, labelpad=10)
    ax.set_title(f'{title}\nDiagonal: count / row %, off-diagonal: count',
                 color=_INK, fontsize=12, pad=12)
    ax.set_aspect('equal')
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(_AXIS)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count == 0:
                continue
            pct = cm_norm[i, j] * 100
            txt_color = 'white' if cm_norm[i, j] > 0.55 else _INK
            if i == j:
                label = f'{count}\n{pct:.1f}%'
                fontsize = diag_size
                weight = 'bold'
            else:
                if n > 12 and pct < 0.5:
                    continue
                label = f'{count}'
                fontsize = offdiag_size
                weight = 'normal'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=fontsize, color=txt_color, fontweight=weight, linespacing=1.15)

    return _save_clean(fig, save_path)


def save_lr_schedule(history: dict, save_path: str, phase_boundaries: list = None) -> str:
    """Learning rate vs epoch (log scale) — paper style.

    phase_boundaries: optional list of epoch indices (1-based) at which a
    dashed vertical line marks a phase transition (e.g. EfficientNet's
    head-only -> fine-tune switch).
    """
    lr_history = history.get('LR', [])
    epochs = list(range(1, len(lr_history) + 1))

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    ax.plot(epochs, lr_history, color=_C_TRAIN, linewidth=2,
            marker='o', markersize=4, markevery=max(1, len(epochs) // 15))
    ax.set_yscale('log')
    _paper_style(ax, 'Learning Rate Schedule', ylabel='LR (log scale)')

    if phase_boundaries:
        for i, b in enumerate(phase_boundaries):
            ax.axvline(b + 0.5, color=_INK_MUTED, linestyle='--', linewidth=1,
                       label='Phase change' if i == 0 else None)
        ax.legend(frameon=False, labelcolor=_INK, fontsize=9, loc='best')

    return _save_clean(fig, save_path)


def generate_training_plots(history: dict, y_true, y_pred, class_names, run_dir: str,
                             lang_true=None, lang_pred=None, lang_class_names=None,
                             phase_boundaries: list = None) -> dict:
    """Generate and save the full paper-style plot set + history.csv.

    Returns a dict of saved paths, keyed by artifact name.
    """
    os.makedirs(run_dir, exist_ok=True)
    paths = {
        'learning_curve':   save_learning_curve(history, os.path.join(run_dir, 'learning_curve.png')),
        'training_history': save_train_history(history, os.path.join(run_dir, 'training_history.png')),
        'confusion_matrix': save_confusion_matrix(
            y_true, y_pred, class_names, os.path.join(run_dir, 'confusion_matrix.png')),
        'lr_schedule': save_lr_schedule(
            history, os.path.join(run_dir, 'lr_schedule.png'), phase_boundaries=phase_boundaries),
    }
    if lang_true is not None and lang_pred is not None:
        paths['confusion_matrix_lang'] = save_confusion_matrix(
            lang_true, lang_pred, lang_class_names or ['Persian', 'English'],
            os.path.join(run_dir, 'confusion_matrix_lang.png'),
            title='Language Confusion Matrix — Test Set')

    history_csv = os.path.join(run_dir, 'history.csv')
    pd.DataFrame(history).to_csv(history_csv, index_label='epoch')
    paths['history_csv'] = history_csv

    return paths


def make_run_dir(model_tag: str, epochs: int, batch_size: int,
                 lr: float, weight_decay: float) -> str:
    """Create timestamped run directory under runs/."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{model_tag}"
            f"_ep{epochs}"
            f"_bs{batch_size}"
            f"_lr{lr:.0e}"
            f"_wd{weight_decay:.0e}"
            f"_{ts}")
    path = os.path.join("runs", name)
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, run_dir: str, filename: str) -> str:
    """Save an arbitrary figure to run_dir and close it. Returns saved path."""
    path = os.path.join(run_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_run_metadata(run_dir: str, meta: dict) -> None:
    """Write training metadata as JSON into the run directory."""
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
