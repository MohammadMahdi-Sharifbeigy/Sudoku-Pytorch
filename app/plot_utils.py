"""Dark-themed Matplotlib plot helpers for the Streamlit UI."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

_BG_FIG  = '#0e1117'
_BG_AX   = '#1a1a2e'
_C_TRAIN = '#4fc3f7'
_C_VAL   = '#f48fb1'
_C_GRID  = '#2d2d3f'


def _style_ax(ax, title: str, ylabel: str = '', xlabel: str = 'Epoch') -> None:
    ax.set_facecolor(_BG_AX)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#aaa', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#aaa', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('white')
    ax.grid(True, alpha=0.25, color=_C_GRID, linewidth=0.8)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Dark-theme confusion matrix with count + row-normalised %."""
    n   = len(class_names)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(_BG_FIG)
    ax.set_facecolor(_BG_FIG)

    im   = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.set_label('Row-normalised accuracy', color='white', fontsize=9)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, color='white', fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(class_names, color='white', fontsize=10)
    ax.set_xlabel('Predicted', color='white', fontsize=12, labelpad=10)
    ax.set_ylabel('True',      color='white', fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix — Test Set\n(count  /  row %)',
                 color='white', fontsize=14, pad=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            if count == 0:
                continue
            txt_color = 'white' if cm_norm[i, j] > 0.55 else '#111'
            ax.text(j, i, f'{count}\n{pct:.1f}%',
                    ha='center', va='center', fontsize=8,
                    color=txt_color, fontweight='bold', linespacing=1.4)

    plt.tight_layout()
    return fig


def plot_lang_confusion_matrix(lang_true, lang_pred):
    """2×2 Persian / English confusion matrix."""
    cm  = confusion_matrix(lang_true, lang_pred, labels=[0, 1])
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(_BG_FIG)
    ax.set_facecolor(_BG_FIG)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ['Persian', 'English']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color='white', fontsize=11)
    ax.set_yticklabels(labels, color='white', fontsize=11)
    ax.set_xlabel('Predicted', color='white', fontsize=11)
    ax.set_ylabel('True',      color='white', fontsize=11)
    ax.set_title('Language Confusion Matrix\n(count  /  row %)',
                 color='white', fontsize=12, pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            txt_color = 'white' if cm_norm[i, j] > 0.55 else '#111'
            ax.text(j, i, f'{count}\n{pct:.1f}%',
                    ha='center', va='center', fontsize=10,
                    color=txt_color, fontweight='bold', linespacing=1.4)

    plt.tight_layout()
    return fig


def plot_multitask_training_history(history: dict):
    """6-panel dashboard for MultiTaskDigitCNN training history."""
    epochs = list(range(1, len(history['Train Loss']) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(_BG_FIG)
    fig.suptitle('Multi-Task Training Dashboard', color='white',
                 fontsize=16, fontweight='bold', y=1.01)

    def _plot(ax, train_key, val_key, title, ylabel='', pct=False):
        suffix = ' (%)' if pct else ''
        ax.plot(epochs, history[train_key], color=_C_TRAIN, linewidth=2,
                label='Train', marker='o', markersize=3, markevery=max(1, len(epochs)//10))
        ax.plot(epochs, history[val_key],   color=_C_VAL,   linewidth=2,
                label='Val',   marker='s', markersize=3, markevery=max(1, len(epochs)//10))
        _style_ax(ax, title + suffix, ylabel=ylabel)
        ax.legend(facecolor='#1e1e2e', edgecolor='#444', labelcolor='white', fontsize=8, loc='best')

    _plot(axes[0, 0], 'Train Loss',       'Val Loss',       'Total Loss',         ylabel='Loss')
    _plot(axes[0, 1], 'Train Digit Loss', 'Val Digit Loss', 'Digit Head Loss',    ylabel='Loss')
    _plot(axes[0, 2], 'Train Lang Loss',  'Val Lang Loss',  'Language Head Loss', ylabel='Loss')
    _plot(axes[1, 0], 'Train Digit Acc',  'Val Digit Acc',  'Digit Accuracy',     ylabel='Accuracy', pct=True)
    _plot(axes[1, 1], 'Train Lang Acc',   'Val Lang Acc',   'Language Accuracy',  ylabel='Accuracy', pct=True)

    ax6 = axes[1, 2]
    ax6.set_facecolor(_BG_AX)
    d_contrib = [0.7 * v for v in history['Train Digit Loss']]
    l_contrib = [0.3 * v for v in history['Train Lang Loss']]
    ax6.stackplot(epochs, d_contrib, l_contrib,
                  labels=['Digit ×0.7', 'Lang ×0.3'],
                  colors=[_C_TRAIN, _C_VAL], alpha=0.75)
    _style_ax(ax6, 'Loss Composition (Train)', ylabel='Weighted loss')
    ax6.legend(facecolor='#1e1e2e', edgecolor='#444', labelcolor='white', fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig
