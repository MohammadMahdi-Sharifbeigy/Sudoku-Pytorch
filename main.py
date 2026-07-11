import os
import copy
import json
import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# --- Local Module Imports ---
from model import (DigitCNN, FocalLoss,
                   MultiTaskDigitCNN, MultiTaskFocalLoss,
                   UnifiedDigitCNN, decode_unified_class, UNIFIED_NUM_CLASSES)
from solver import SudokuSolver
from vision import (
    resize_and_maintain_aspect_ratio,
    apply_grayscale_blur_and_threshold,
    get_valid_cells_from_image as vision_orig_get_cells,
    get_predicted_sudoku_grid_torch,
    generate_solution_image,
    plot_cell_images_in_grid,
)
import vision_a
from train import (train_epoch, validate, collect_predictions,
                   train_epoch_multitask, validate_multitask, collect_predictions_multitask)
from data_utils import (get_dataloaders, get_dataloaders_mnist_hoda, get_dataloaders_all,
                        get_dataloaders_mnist_only, get_dataloaders_multitask,
                        get_dataloaders_persian, get_dataloaders_english,
                        get_dataloaders_unified20)
from report_utils import save_training_report, save_inference_report, save_training_report_multitask
from optimize_model import run_optimization_and_benchmark, run_optimization_and_benchmark_multitask

def save_debug_outputs(
    image_name: str,
    img_rgb: np.ndarray,
    cells_a, board_a, err_a,
    cells_b, board_b, err_b,
    cells_selected, board_selected,
    pipeline_choice: str,
    out_root: str = "debug_outputs",
) -> str:
    """Save all intermediate outputs after inference for offline inspection."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(image_name)[0]
    out_dir = os.path.join(out_root, f"{ts}_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    # original image (BGR for cv2.imwrite)
    cv2.imwrite(os.path.join(out_dir, "00_original.jpg"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    def save_board(board, tag):
        if board is None:
            return
        path = os.path.join(out_dir, f"{tag}_warped_grid.jpg")
        if len(board.shape) == 3:
            cv2.imwrite(path, cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, board)

    def save_cells_mosaic(cells, tag):
        if cells is None:
            return
        fig = plot_cell_images_in_grid(cells)
        fig.savefig(os.path.join(out_dir, f"{tag}_cells_mosaic.png"), dpi=80, bbox_inches="tight")
        plt.close(fig)

    def save_cell_images(cells, tag):
        if cells is None:
            return
        cells_dir = os.path.join(out_dir, f"{tag}_cells")
        os.makedirs(cells_dir, exist_ok=True)
        for c in cells:
            r, col = c['grid_row'], c['grid_col']
            flag = "D" if c['contains_digit'] else "E"
            cv2.imwrite(os.path.join(cells_dir, f"r{r}c{col}_{flag}.png"), c['img'])

    save_board(board_a, "A")
    save_board(board_b, "B")
    save_board(board_selected, "selected")
    save_cells_mosaic(cells_a, "A")
    save_cells_mosaic(cells_b, "B")
    save_cells_mosaic(cells_selected, "selected")
    save_cell_images(cells_a, "A")
    save_cell_images(cells_b, "B")

    # summary JSON
    summary = {
        "image": image_name,
        "pipeline_choice": pipeline_choice,
        "pipeline_A": {
            "success": cells_a is not None,
            "error": err_a,
            "digit_count": sum(c['contains_digit'] for c in cells_a) if cells_a else None,
        },
        "pipeline_B": {
            "success": cells_b is not None,
            "error": err_b,
            "digit_count": sum(c['contains_digit'] for c in cells_b) if cells_b else None,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return out_dir


# --- UI Configuration ---
st.set_page_config(
    page_title="AI Sudoku Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stProgress .st-bo { background-color: #4CAF50; }
    .cell-box { border: 1px solid #555; border-radius: 4px; padding: 4px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("Intelligent Sudoku Solver (PyTorch + OpenCV)")
st.markdown("An end-to-end computer vision and deep learning pipeline for detecting and solving Sudoku puzzles.")

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["Inference (Solve)", "Model Training", "Model Optimization"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Compute Device:** `{device}`")


# ==========================================
# HELPER: Per-cell prediction details
# ==========================================
def get_per_cell_predictions(model, cells, device, is_multitask=False, is_unified=False):
    """Returns predicted label, confidence, and optionally language per cell.

    model can be:
      DigitCNN                    → single output tensor (10 classes)
      MultiTaskDigitCNN           → tuple (digit_logits, lang_logits)
      UnifiedDigitCNN             → single output tensor (20 classes)
      OnnxInferenceSession        → __call__ returns first output; .infer_multitask() for both heads
    """
    is_onnx_mt = (isinstance(model, OnnxInferenceSession)
                  and model.n_outputs > 1
                  and is_multitask)

    results = []
    for cell in cells:
        if not cell['contains_digit']:
            entry = {'label': 0, 'confidence': 1.0, 'has_digit': False}
            if is_multitask or is_unified:
                entry['lang_label'] = None
                entry['lang_confidence'] = None
            results.append(entry)
            continue

        img    = cell['img'].astype('float32') / 255.0
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_unified:
                logits20 = model(tensor)                          # (1, 20)
                probs20  = torch.softmax(logits20, dim=1).cpu().numpy()[0]
                cls_pred = int(np.argmax(probs20))
                cls_conf = float(probs20[cls_pred])
                digit, lang_name = decode_unified_class(cls_pred)
                # confidence for reporting: prob of predicted 20-class
                results.append({
                    'label': digit, 'confidence': cls_conf,
                    'has_digit': True,
                    'lang_label':      (0 if lang_name == 'Persian' else 1) if lang_name else None,
                    'lang_confidence': cls_conf,
                })
                continue

            if is_onnx_mt:
                d_out, l_out = model.infer_multitask(tensor)
            elif is_multitask:
                d_out, l_out = model(tensor)
            else:
                d_out = model(tensor)

        if is_multitask or is_onnx_mt:
            d_probs   = torch.softmax(d_out, dim=1).cpu().numpy()[0]
            l_probs   = torch.softmax(l_out, dim=1).cpu().numpy()[0]
            pred      = int(np.argmax(d_probs))
            conf      = float(d_probs[pred])
            lang_pred = int(np.argmax(l_probs))
            lang_conf = float(l_probs[lang_pred])
            results.append({'label': pred, 'confidence': conf, 'has_digit': True,
                             'lang_label': lang_pred, 'lang_confidence': lang_conf})
        else:
            probs = torch.softmax(d_out, dim=1).cpu().numpy()[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            results.append({'label': pred, 'confidence': conf, 'has_digit': True})
    return results


# ──────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────

_BG_FIG  = '#0e1117'   # Streamlit dark bg
_BG_AX   = '#1a1a2e'   # subplot bg
_C_TRAIN = '#4fc3f7'   # blue for train
_C_VAL   = '#f48fb1'   # pink for val
_C_GRID  = '#2d2d3f'


def _style_ax(ax, title, ylabel='', xlabel='Epoch'):
    """Apply dark-theme styling to a single subplot axis."""
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
    """Dark-theme confusion matrix.  Cell shows count + row-normalised %.
    Text colour chosen per cell so it's always readable.
    """
    n   = len(class_names)
    lbl = list(range(n))
    cm  = confusion_matrix(y_true, y_pred, labels=lbl)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum          # row-normalised 0-1

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
            # Dark cell (norm > 0.55) → white text; light cell → near-black
            txt_color = 'white' if cm_norm[i, j] > 0.55 else '#111'
            ax.text(j, i, f'{count}\n{pct:.1f}%',
                    ha='center', va='center', fontsize=8,
                    color=txt_color, fontweight='bold', linespacing=1.4)

    plt.tight_layout()
    return fig


def plot_lang_confusion_matrix(lang_true, lang_pred):
    """2×2 Persian / English confusion matrix, dark-themed."""
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


def plot_multitask_training_history(history):
    """6-panel dark-themed training dashboard for MultiTaskDigitCNN.

    Layout (2 rows × 3 cols):
      [Total Loss]  [Digit Head Loss]  [Lang Head Loss]
      [Digit Acc]   [Language Acc]     [Loss Composition]
    """
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
        leg = ax.legend(facecolor='#1e1e2e', edgecolor='#444',
                        labelcolor='white', fontsize=8, loc='best')

    _plot(axes[0, 0], 'Train Loss',       'Val Loss',       'Total Loss',      ylabel='Loss')
    _plot(axes[0, 1], 'Train Digit Loss', 'Val Digit Loss', 'Digit Head Loss', ylabel='Loss')
    _plot(axes[0, 2], 'Train Lang Loss',  'Val Lang Loss',  'Language Head Loss', ylabel='Loss')
    _plot(axes[1, 0], 'Train Digit Acc',  'Val Digit Acc',  'Digit Accuracy',  ylabel='Accuracy', pct=True)
    _plot(axes[1, 1], 'Train Lang Acc',   'Val Lang Acc',   'Language Accuracy', ylabel='Accuracy', pct=True)

    # Panel 6: stacked-area showing weighted loss composition (train only)
    ax6 = axes[1, 2]
    ax6.set_facecolor(_BG_AX)
    d_contrib = [0.7 * v for v in history['Train Digit Loss']]
    l_contrib = [0.3 * v for v in history['Train Lang Loss']]
    ax6.stackplot(epochs, d_contrib, l_contrib,
                  labels=['Digit ×0.7', 'Lang ×0.3'],
                  colors=[_C_TRAIN, _C_VAL], alpha=0.75)
    _style_ax(ax6, 'Loss Composition (Train)', ylabel='Weighted loss')
    leg6 = ax6.legend(facecolor='#1e1e2e', edgecolor='#444',
                      labelcolor='white', fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Inference model wrappers
# ──────────────────────────────────────────────

class OnnxInferenceSession:
    """Wraps an onnxruntime session to behave like a PyTorch model.

    __call__ always returns digit logits (first output) as a torch.Tensor,
    so get_predicted_sudoku_grid_torch (vision.py) works unchanged.
    For multi-task ONNX, call .infer_multitask(tensor) to get both outputs.
    Runs on CPU regardless of the device argument passed to .to().
    """
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess      = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.n_outputs = len(self.sess.get_outputs())
        self.onnx_path = onnx_path

    def __call__(self, x):
        outputs = self.sess.run(None, {'input': x.cpu().numpy()})
        return torch.from_numpy(outputs[0])   # digit logits only

    def infer_multitask(self, x):
        outputs = self.sess.run(None, {'input': x.cpu().numpy()})
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])

    def eval(self): return self
    def to(self, _device): return self


class DigitOnlyModelWrapper:
    """Wraps MultiTaskDigitCNN to return only digit logits.
    Needed so get_predicted_sudoku_grid_torch (vision.py) gets a single tensor.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, x):
        digit_logits, _ = self._model(x)
        return digit_logits

    def eval(self): self._model.eval(); return self
    def to(self, device): self._model.to(device); return self


class UnifiedDigitOnlyWrapper:
    """Wraps UnifiedDigitCNN (20-class) so vision.py gets digit logits (0-9).

    Folds English + Persian logits per digit position:
      digit_logit[0] = max(class_0, class_10)   ← empty
      digit_logit[d] = class_d + class_{d+10}   ← digit d in either language
    vision.py then argmaxes over 10 classes to get digit 0-9.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, x):
        logits20 = self._model(x)                     # (B, 20)
        logits10 = logits20[:, :10].clone()
        logits10[:, 0] = torch.maximum(logits20[:, 0], logits20[:, 10])   # empty
        logits10[:, 1:] = logits20[:, 1:10] + logits20[:, 11:20]          # sum English+Persian per digit
        return logits10                               # (B, 10)

    def eval(self): self._model.eval(); return self
    def to(self, device): self._model.to(device); return self


# ==========================================
# MODE 1: INFERENCE (SOLVE SUDOKU)
# ==========================================
if app_mode == "Inference (Solve)":
    st.markdown("### Upload Sudoku Image")
    uploaded_file = st.file_uploader("Drag and drop your Sudoku image here", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

        # ── Model paths ───────────────────────────────────────────────
        _paths = {
            'persian_pt':  'models/best_model_persian.pt',
            'english_pt':  'models/best_model_english.pt',
            'default_pt':  'models/best_model.pt',
            'mt_pt':       'models/best_model_multitask.pt',
            'unified_pt':  'models/best_model_unified20.pt',
            'persian_onnx':'models/best_model_persian.onnx',
            'default_onnx':'models/best_model.onnx',
            'mt_onnx':     'models/best_model_multitask.onnx',
            'unified_onnx':'models/best_model_unified20.onnx',
        }

        # ── Sidebar toggles ────────────────────────────────────────────
        st.sidebar.markdown("### Model Selection")

        use_persian = st.sidebar.toggle(
            "Persian image (فارسی)",
            value=False,
            help="ON → loads best_model_persian.pt (DigitCNN trained on Hoda). "
                 "Off → uses English model by default.",
            disabled=not os.path.exists(_paths['persian_pt']),
        )

        available_mt_models = {}
        if os.path.exists(_paths['mt_pt']):
            available_mt_models["Multi-Task CNN (digit + language head)"] = 'mt'
        if os.path.exists(_paths['unified_pt']):
            available_mt_models["Unified 20-Class (MobileNet/ShuffleNet)"] = 'unified'

        use_multimodel = st.sidebar.toggle(
            "Auto-detect language (multi-model)",
            value=False,
            help="ON → predicts digit AND language per cell automatically. "
                 "Overrides Persian toggle.",
            disabled=len(available_mt_models) == 0,
        )

        if use_multimodel and available_mt_models:
            selected_mt_label = st.sidebar.radio(
                "Multi-model to use:",
                list(available_mt_models.keys()),
                index=0,
            )
            selected_mt_key = available_mt_models[selected_mt_label]
            if selected_mt_key == 'unified':
                unified_bb = st.sidebar.selectbox(
                    "Backbone (must match training)",
                    ["mobilenet_v3_small", "shufflenet_v2_x0_5"],
                )
        else:
            selected_mt_key = None
            unified_bb      = "mobilenet_v3_small"

        # ── ONNX acceleration ──────────────────────────────────────────
        use_onnx = False
        onnx_avail = False
        try:
            import onnxruntime  # noqa: F401
            if use_multimodel:
                _onnx_candidate = _paths['mt_onnx'] if selected_mt_key == 'mt' else _paths['unified_onnx']
            elif use_persian:
                _onnx_candidate = _paths['persian_onnx']
            else:
                _onnx_candidate = _paths['default_onnx']
            onnx_avail = os.path.exists(_onnx_candidate)
        except ImportError:
            st.sidebar.caption("onnxruntime not installed — ONNX unavailable.")
            _onnx_candidate = ''

        if onnx_avail:
            use_onnx = st.sidebar.toggle("Use ONNX (faster ~2-5×)", value=True)

        # ── Resolve flags ──────────────────────────────────────────────
        is_multitask = use_multimodel and selected_mt_key == 'mt'
        is_unified   = use_multimodel and selected_mt_key == 'unified'
        is_persian   = use_persian and not use_multimodel

        # ── Load model ─────────────────────────────────────────────────
        if use_onnx and onnx_avail:
            model        = OnnxInferenceSession(_onnx_candidate)
            vision_model = model
            _label = ("multi-task" if is_multitask else
                      "unified-20" if is_unified else
                      "Persian" if is_persian else "English")
            st.sidebar.success(f"ONNX · {_label}")

        elif is_unified:
            _pt = UnifiedDigitCNN(backbone=unified_bb, pretrained=False,
                                  num_classes=UNIFIED_NUM_CLASSES).to(device)
            _pt.load_state_dict(torch.load(_paths['unified_pt'], map_location=device))
            _pt.eval()
            model        = _pt
            vision_model = UnifiedDigitOnlyWrapper(_pt)
            st.sidebar.info(f"PyTorch · unified-20 · {unified_bb}")

        elif is_multitask:
            _pt = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2).to(device)
            _pt.load_state_dict(torch.load(_paths['mt_pt'], map_location=device))
            _pt.eval()
            model        = _pt
            vision_model = DigitOnlyModelWrapper(_pt)
            st.sidebar.info("PyTorch · multi-task")

        elif is_persian and os.path.exists(_paths['persian_pt']):
            model = DigitCNN(num_classes=10).to(device)
            model.load_state_dict(torch.load(_paths['persian_pt'], map_location=device))
            model.eval()
            vision_model = model
            st.sidebar.info("PyTorch · Persian (Hoda)")

        else:
            # Default: English single-task model
            _eng_pt = _paths['english_pt'] if os.path.exists(_paths['english_pt']) else _paths['default_pt']
            if not os.path.exists(_eng_pt):
                st.error("No trained model found. Go to **Model Training** to train one.")
                st.stop()
            model = DigitCNN(num_classes=10).to(device)
            model.load_state_dict(torch.load(_eng_pt, map_location=device))
            model.eval()
            vision_model = model
            st.sidebar.info("PyTorch · English (default)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original Image")
            st.image(img, width='stretch')

        with st.spinner('Processing image and extracting grid...'):
            try:
                # --- Run both pipelines: Original vs A ---
                cells_orig, M_orig, board_orig = None, None, None
                cells_a, M_a, board_a = None, None, None
                err_orig, err_a = None, None

                try:
                    cells_orig, M_orig, board_orig = vision_orig_get_cells(img)
                except Exception as e:
                    err_orig = str(e)

                try:
                    cells_a, M_a, board_a = vision_a.get_valid_cells_from_image(img)
                except Exception as e:
                    err_a = str(e)

                if cells_orig is None and cells_a is None:
                    raise Exception(
                        f"Both pipelines failed.\nOriginal: {err_orig}\nA: {err_a}"
                    )

                # ── Pipeline comparison panel ──────────────────────────────────
                with st.expander("Pipeline Comparison (Original vs A)", expanded=True):
                    cmp_col_orig, cmp_col_a = st.columns(2)

                    count_orig = sum(c['contains_digit'] for c in cells_orig) if cells_orig else -1
                    count_a = sum(c['contains_digit'] for c in cells_a) if cells_a else -1

                    with cmp_col_orig:
                        st.markdown("**Original Pipeline**")
                        if cells_orig is None:
                            st.error(f"Failed: {err_orig}")
                        else:
                            st.image(board_orig, channels="GRAY",
                                     caption=f"Warped grid ({count_orig} digits detected)")
                            fig_orig = plot_cell_images_in_grid(cells_orig)
                            st.pyplot(fig_orig)
                            plt.close(fig_orig)

                    with cmp_col_a:
                        st.markdown("**Pipeline A — Canny + Hough**")
                        if cells_a is None:
                            st.error(f"Failed: {err_a}")
                        else:
                            st.image(board_a, channels="GRAY",
                                     caption=f"Warped grid ({count_a} digits detected)")
                            fig_a = plot_cell_images_in_grid(cells_a)
                            st.pyplot(fig_a)
                            plt.close(fig_a)

                    options = []
                    if cells_orig is not None:
                        options.append(f"Original — {count_orig} digits")
                    if cells_a is not None:
                        options.append(f"Pipeline A (Canny+Hough) — {count_a} digits")

                    default_idx = 0  # default to Original (currently better)
                    pipeline_choice = st.radio(
                        "Select pipeline to use for solving:",
                        options,
                        index=default_idx,
                    )

                # Resolve selection
                if "Pipeline A" in pipeline_choice:
                    cells, M, board_image = cells_a, M_a, board_a
                else:
                    cells, M, board_image = cells_orig, M_orig, board_orig

                # Save debug outputs for offline inspection
                debug_dir = save_debug_outputs(
                    image_name=uploaded_file.name,
                    img_rgb=img,
                    cells_a=cells_a, board_a=board_a, err_a=err_a,
                    cells_b=cells_orig, board_b=board_orig, err_b=err_orig,
                    cells_selected=cells, board_selected=board_image,
                    pipeline_choice=pipeline_choice,
                )
                st.sidebar.info(f"🔬 Debug outputs saved to `{debug_dir}`")

                with st.expander("View Intermediate Processing Steps", expanded=False):
                    step_col1, step_col2, step_col3 = st.columns(3)
                    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)
                    with step_col1:
                        st.markdown("**1. Adaptive Thresholding**")
                        st.image(thresh, width='stretch', channels="GRAY")
                    with step_col2:
                        st.markdown("**2. Perspective Transform (selected)**")
                        if board_image is not None:
                            st.image(board_image, width='stretch', channels="GRAY")
                    with step_col3:
                        st.markdown("**3. Cell Extraction (selected)**")
                        if cells is not None:
                            fig = plot_cell_images_in_grid(cells)
                            st.pyplot(fig)
                            plt.close(fig)

                # --- Per-cell Prediction Detail ---
                per_cell = get_per_cell_predictions(
                    model, cells, device,
                    is_multitask=is_multitask,
                    is_unified=is_unified,
                )

                with st.expander("🔍 Cell-by-Cell Extraction & Prediction Details", expanded=True):
                    st.markdown(
                        "Each cell shows its **extracted image**, whether a digit was **detected**, "
                        "the **predicted digit**, and the model's **confidence score**. "
                        "🟢 = high confidence (≥80%), 🟡 = medium (50–80%), 🔴 = low (<50%)."
                    )
                    st.markdown("---")

                    COLS = 9
                    grid_cols = st.columns(COLS)

                    # Column headers
                    for c in range(COLS):
                        with grid_cols[c]:
                            st.markdown(f"<div style='text-align:center;color:#888;font-size:11px;'>Col {c}</div>",
                                        unsafe_allow_html=True)

                    for row in range(9):
                        grid_cols = st.columns(COLS)
                        for col in range(COLS):
                            idx = row * 9 + col
                            cell = cells[idx]
                            info = per_cell[idx]

                            with grid_cols[col]:
                                # Display cell image
                                cell_img_display = cell['img']
                                st.image(cell_img_display, width=60, channels="GRAY",
                                         caption=None)

                                if not info['has_digit']:
                                    st.markdown(
                                        "<div style='text-align:center;font-size:11px;color:#888;'>Empty</div>",
                                        unsafe_allow_html=True)
                                else:
                                    label = info['label']
                                    conf = info['confidence']
                                    conf_pct = conf * 100

                                    if conf_pct >= 80:
                                        dot = "🟢"
                                    elif conf_pct >= 50:
                                        dot = "🟡"
                                    else:
                                        dot = "🔴"

                                    lang_badge = ''
                                    if is_multitask and info.get('lang_label') is not None:
                                        ln = 'FA' if info['lang_label'] == 0 else 'EN'
                                        lc = info['lang_confidence'] * 100
                                        lang_badge = f"<div style='text-align:center;font-size:9px;color:#888;'>{ln} {lc:.0f}%</div>"
                                    st.markdown(
                                        f"<div style='text-align:center;font-size:13px;font-weight:bold;'>{dot} {label}</div>"
                                        f"<div style='text-align:center;font-size:10px;color:#aaa;'>{conf_pct:.1f}%</div>"
                                        f"{lang_badge}",
                                        unsafe_allow_html=True)

                        # Row separator every 3 rows
                        if row in (2, 5):
                            st.markdown("<hr style='border-color:#444;margin:4px 0;'>", unsafe_allow_html=True)

                # --- Prediction & Solving ---
                # vision_model always returns single digit logits tensor (ONNX or DigitOnlyWrapper)
                grid_array = get_predicted_sudoku_grid_torch(vision_model, cells, device)
                solver = SudokuSolver(board=copy.deepcopy(grid_array))
                solved_board = solver.board if solver.solve() else None

                # --- Save inference text report ---
                os.makedirs('models', exist_ok=True)
                inf_report_path = save_inference_report(
                    image_name=uploaded_file.name,
                    cells=cells,
                    per_cell_info=per_cell,
                    grid_array=grid_array,
                    solved_board=solved_board,
                    output_path=f'models/reports/inference_report_{uploaded_file.name}.txt',
                )
                st.sidebar.success(f"📄 Inference report saved to `{inf_report_path}`")

                if solved_board is not None:
                    final_image = generate_solution_image(
                        full_image=img, board_image=board_image,
                        cells_list=cells, solved_board_arr=solved_board, M_matrix=M
                    )

                    with col2:
                        st.markdown("#### Solved Sudoku")
                        st.image(final_image, width='stretch')
                        st.success("Sudoku solved successfully!")

                    st.markdown("### Digital Representation")
                    matrix_df = pd.DataFrame(solved_board)
                    st.dataframe(
                        matrix_df.style.set_properties(**{'text-align': 'center', 'font-weight': 'bold'}),
                        width='stretch'
                    )

                    # Inline download button for inference report
                    with open(inf_report_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="⬇️ Download Inference Report (.txt)",
                            data=f.read(),
                            file_name='inference_report.txt',
                            mime='text/plain',
                        )
                else:
                    st.error("The extracted grid is invalid or unsolvable. Please ensure the image is clear and well-lit.")
                    st.markdown("**Extracted Grid (before solving):**")
                    st.dataframe(pd.DataFrame(grid_array), width='stretch')

            except Exception as e:
                import traceback
                st.error(f"Computer Vision Pipeline Error: {e}")
                st.code(traceback.format_exc())


# ==========================================
# MODE 2: MODEL TRAINING (CNN)
# ==========================================
elif app_mode == "Model Training":
    st.markdown("### Model Training Dashboard")
    st.markdown("Configure hyperparameters and monitor the CNN training process in real-time.")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    with param_col2:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
    with param_col3:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)

    data_path = st.text_input("Dataset Directory Path", value="data")
    dataset_mode = st.radio(
        "Training Dataset",
        [
            "MNIST Only",
            "MNIST + Fonts (Recommended for printed Sudoku)",
            "MNIST + Hoda",
            "MNIST + Fonts + Hoda (All)",
            "Persian Only (Hoda — saves best_model_persian.pt)",
            "English Only (MNIST + Fonts — saves best_model_english.pt)",
            "Unified 20-Class (MNIST + Fonts + Hoda → saves best_model_unified20.pt)",
        ],
        index=1,
        help=(
            "Persian Only / English Only: dedicated single-language DigitCNN.\n\n"
            "Unified 20-Class: MobileNetV3-Small or ShuffleNet V2 fine-tuned on 20 classes "
            "(English 0-9 + Persian 0-9 in one pass). Handles mixed-language Sudoku images."
        ),
    )

    if dataset_mode.startswith("Unified"):
        unified_backbone = st.selectbox(
            "Backbone",
            ["mobilenet_v3_small", "shufflenet_v2_x0_5"],
            index=0,
            help=(
                "mobilenet_v3_small: ~2.5 M params, stronger features.\n"
                "shufflenet_v2_x0_5: ~0.35 M params, fastest inference."
            ),
        )
        unified_pretrained = st.toggle(
            "Use ImageNet pretrained weights (first conv re-initialised for 1-channel input)",
            value=False,
            help="Pretrained weights help backbone layers but NOT the first conv (grayscale). "
                 "False recommended for 28×28 digit images.",
        )
    else:
        unified_backbone   = "mobilenet_v3_small"
        unified_pretrained = False

    enable_multitask = st.toggle(
        "Enable language classification (multi-task)",
        value=False,
        help=(
            "Trains MultiTaskDigitCNN with a second head for Persian/English detection. "
            "Uses MNIST + Fonts + Hoda + Empty regardless of dataset mode above. "
            "Class-weighted loss compensates for Hoda being the minority source. "
            "Ignored when Persian Only or English Only is selected above."
        ),
    )

    if st.button("Start Training Sequence", width='stretch'):
        if not os.path.exists(data_path):
            st.error(f"Dataset path `{data_path}` does not exist. Please verify the path.")
            st.stop()

        st.info("Initializing DataLoaders. This may take a moment...")

        try:
            os.makedirs('models', exist_ok=True)

            is_persian_only  = dataset_mode.startswith("Persian Only")
            is_english_only  = dataset_mode.startswith("English Only")
            is_unified_mode  = dataset_mode.startswith("Unified")
            is_lang_specific = is_persian_only or is_english_only

            if is_unified_mode:
                # ======================================================
                # UNIFIED 20-CLASS PATH
                # ======================================================
                train_loader, val_loader, test_loader = get_dataloaders_unified20(data_path, batch_size=batch_size)

                model = UnifiedDigitCNN(
                    backbone=unified_backbone,
                    pretrained=unified_pretrained,
                    num_classes=UNIFIED_NUM_CLASSES,
                ).to(device)

                total_p, _ = model.param_count()
                st.info(f"Backbone: **{unified_backbone}** | Total params: **{total_p:,}** | "
                        f"Pretrained: {'Yes' if unified_pretrained else 'No'}")

                criterion = nn.CrossEntropyLoss()   # standard CE for 20-class
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

                save_path     = 'models/best_model_unified20.pt'
                progress_bar  = st.progress(0)
                status_text   = st.empty()

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("#### Loss Curve")
                    loss_ph_u = st.empty()
                with chart_col2:
                    st.markdown("#### Accuracy Curve (20-class)")
                    acc_ph_u = st.empty()

                metrics_table = st.empty()
                best_val_loss = float('inf')
                history = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}

                for epoch in range(int(epochs)):
                    status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")
                    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                    v_loss, v_acc = validate(model, val_loader, criterion, device)
                    scheduler.step(v_loss)

                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        torch.save(model.state_dict(), save_path)

                    history['Train Loss'].append(t_loss); history['Val Loss'].append(v_loss)
                    history['Train Acc'].append(t_acc);   history['Val Acc'].append(v_acc)

                    loss_ph_u.line_chart(pd.DataFrame({'Train': history['Train Loss'], 'Val': history['Val Loss']}))
                    acc_ph_u.line_chart(pd.DataFrame({'Train': history['Train Acc'],  'Val': history['Val Acc']}))
                    metrics_table.markdown(f"""
| Metric | Training | Validation |
|---|---|---|
| **Loss** | {t_loss:.4f} | {v_loss:.4f} |
| **20-class Acc** | {t_acc:.2f}% | {v_acc:.2f}% |
                    """)
                    progress_bar.progress((epoch + 1) / int(epochs))

                status_text.success(f"Done! Saved to `{save_path}` (Best Val Loss: {best_val_loss:.4f})")

                st.markdown("---"); st.markdown("### Test Set Evaluation")
                with st.spinner("Evaluating…"):
                    model.load_state_dict(torch.load(save_path, map_location=device))
                    t_loss, t_acc = validate(model, test_loader, criterion, device)
                    st.metric("20-class Test Accuracy", f"{t_acc:.2f}%",
                              delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

                st.markdown("---"); st.markdown("### Confusion Matrix (20 classes)")
                with st.spinner("Computing…"):
                    y_true_20, y_pred_20 = collect_predictions(model, test_loader, device)
                    all_lbl20 = sorted(set(y_true_20) | set(y_pred_20))
                    # Label names: 0=Eng_empty, 1-9=Eng_d, 10=Per_empty, 11-19=Per_d
                    def _cls_name(c):
                        if c == 0:   return "ENG_0"
                        if c < 10:   return f"ENG_{c}"
                        if c == 10:  return "PER_0"
                        return f"PER_{c-10}"
                    cn20 = [_cls_name(c) for c in all_lbl20]
                    st.pyplot(plot_confusion_matrix(y_true_20, y_pred_20, cn20))
                    plt.close('all')

                    # Digit-level accuracy (fold 20→10)
                    digit_true = [decode_unified_class(c)[0] for c in y_true_20]
                    digit_pred = [decode_unified_class(c)[0] for c in y_pred_20]
                    digit_acc  = 100 * sum(t == p for t, p in zip(digit_true, digit_pred)) / max(len(digit_true), 1)
                    st.metric("Digit Accuracy (folded to 0-9)", f"{digit_acc:.2f}%")

                rpt = save_training_report(
                    history=history, test_loss=t_loss, test_acc=t_acc,
                    y_true=y_true_20, y_pred=y_pred_20,
                    dataset_mode=dataset_mode + f" [{unified_backbone}]",
                    epochs=int(epochs), learning_rate=learning_rate, batch_size=batch_size,
                    best_val_loss=best_val_loss, model=model,
                    output_path='models/training_report_unified20.txt',
                )
                st.success(f"Report saved to `{rpt}`")
                with open(rpt, 'r', encoding='utf-8') as f:
                    st.download_button("⬇️ Download Unified Report (.txt)",
                                       f.read(), 'training_report_unified20.txt', 'text/plain')

            elif is_lang_specific:
                # ======================================================
                # LANGUAGE-SPECIFIC SINGLE-TASK PATH
                # ======================================================
                if is_persian_only:
                    train_loader, val_loader, test_loader = get_dataloaders_persian(data_path, batch_size=batch_size)
                    if train_loader is None:
                        st.error("Hoda dataset not found. Check `data/DigitDB/` exists.")
                        st.stop()
                    save_path  = 'models/best_model_persian.pt'
                    lang_label = "Persian"
                    st.info("Training dedicated Persian model (Hoda + empty cells).")
                else:
                    train_loader, val_loader, test_loader = get_dataloaders_english(data_path, batch_size=batch_size)
                    save_path  = 'models/best_model_english.pt'
                    lang_label = "English"
                    st.info("Training dedicated English model (MNIST + Fonts + empty cells).")

                model     = DigitCNN(num_classes=10).to(device)
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

                progress_bar  = st.progress(0)
                status_text   = st.empty()

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown(f"#### Loss Curve ({lang_label})")
                    loss_ph_lang = st.empty()
                with chart_col2:
                    st.markdown(f"#### Accuracy Curve ({lang_label})")
                    acc_ph_lang = st.empty()

                metrics_table = st.empty()
                best_val_loss = float('inf')
                history       = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}

                for epoch in range(int(epochs)):
                    status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")
                    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                    v_loss, v_acc = validate(model, val_loader, criterion, device)
                    scheduler.step(v_loss)

                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        torch.save(model.state_dict(), save_path)

                    history['Train Loss'].append(t_loss); history['Val Loss'].append(v_loss)
                    history['Train Acc'].append(t_acc);   history['Val Acc'].append(v_acc)

                    loss_ph_lang.line_chart(pd.DataFrame({'Train': history['Train Loss'], 'Val': history['Val Loss']}))
                    acc_ph_lang.line_chart(pd.DataFrame({'Train': history['Train Acc'],  'Val': history['Val Acc']}))
                    metrics_table.markdown(f"""
| Metric | Training | Validation |
|---|---|---|
| **Loss** | {t_loss:.4f} | {v_loss:.4f} |
| **Accuracy** | {t_acc:.2f}% | {v_acc:.2f}% |
                    """)
                    progress_bar.progress((epoch + 1) / int(epochs))

                status_text.success(f"Training complete! Saved to `{save_path}` (Best Val Loss: {best_val_loss:.4f})")

                st.markdown("---"); st.markdown("### Test Set Evaluation")
                with st.spinner("Evaluating…"):
                    model.load_state_dict(torch.load(save_path, map_location=device))
                    t_loss, t_acc = validate(model, test_loader, criterion, device)
                    st.metric(label=f"{lang_label} Test Accuracy", value=f"{t_acc:.2f}%",
                              delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

                st.markdown("---"); st.markdown("### Confusion Matrix")
                with st.spinner("Computing…"):
                    y_true, y_pred = collect_predictions(model, test_loader, device)
                    all_labels  = sorted(set(y_true) | set(y_pred))
                    class_names = ["Empty" if l == 0 else str(l) for l in all_labels]
                    cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
                    st.pyplot(cm_fig); plt.close(cm_fig)
                    cm_arr = confusion_matrix(y_true, y_pred, labels=all_labels)
                    pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
                    st.dataframe(pd.DataFrame({
                        'Class': class_names, 'Correct': cm_arr.diagonal(),
                        'Total': cm_arr.sum(axis=1), 'Accuracy (%)': [f"{a:.1f}" for a in pca],
                    }).set_index('Class'), width='stretch')

                rpt = save_training_report(
                    history=history, test_loss=t_loss, test_acc=t_acc,
                    y_true=y_true, y_pred=y_pred,
                    dataset_mode=dataset_mode, epochs=int(epochs),
                    learning_rate=learning_rate, batch_size=batch_size,
                    best_val_loss=best_val_loss, model=model,
                    output_path=f'models/training_report_{lang_label.lower()}.txt',
                )
                st.success(f"Report saved to `{rpt}`")
                with open(rpt, 'r', encoding='utf-8') as f:
                    st.download_button(f"⬇️ Download {lang_label} Report (.txt)",
                                       f.read(), f'training_report_{lang_label.lower()}.txt', 'text/plain')

            elif enable_multitask:
                # ======================================================
                # MULTI-TASK PATH
                # ======================================================
                train_loader, val_loader, test_loader, balance_info, lang_weights = \
                    get_dataloaders_multitask(data_path, batch_size=batch_size)

                # Surface imbalance warning prominently
                for split_name, b in balance_info.items():
                    if b.get('imbalanced', False):
                        st.warning(
                            f"**Language imbalance in {split_name} split:** "
                            f"Persian {b['ratio_persian']:.1f}% vs English {b['ratio_english']:.1f}%. "
                            f"Class-weighted loss applied automatically."
                        )

                model     = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2).to(device)
                criterion = MultiTaskFocalLoss(
                    digit_weight=0.7, lang_weight=0.3,
                    lang_class_weights=lang_weights.to(device),
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

                progress_bar  = st.progress(0)
                status_text   = st.empty()

                # Live chart placeholders (updated every epoch)
                st.markdown("#### Live Training Curves")
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    st.markdown("**Loss**")
                    loss_ph = st.empty()
                with lc2:
                    st.markdown("**Digit Accuracy (%)**")
                    dacc_ph = st.empty()
                with lc3:
                    st.markdown("**Language Accuracy (%)**")
                    lacc_ph = st.empty()

                metrics_table = st.empty()

                best_val_loss = float('inf')
                history = {
                    'Train Loss': [], 'Val Loss': [],
                    'Train Digit Acc': [], 'Val Digit Acc': [],
                    'Train Lang Acc':  [], 'Val Lang Acc':  [],
                    'Train Digit Loss': [], 'Train Lang Loss': [],
                    'Val Digit Loss':   [], 'Val Lang Loss':   [],
                }

                for epoch in range(int(epochs)):
                    status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")

                    t_loss, t_d_acc, t_l_acc, t_d_loss, t_l_loss = \
                        train_epoch_multitask(model, train_loader, criterion, optimizer, device)
                    v_loss, v_d_acc, v_l_acc, v_d_loss, v_l_loss = \
                        validate_multitask(model, val_loader, criterion, device)
                    scheduler.step(v_loss)

                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        torch.save(model.state_dict(), 'models/best_model_multitask.pt')

                    history['Train Loss'].append(t_loss);        history['Val Loss'].append(v_loss)
                    history['Train Digit Acc'].append(t_d_acc);  history['Val Digit Acc'].append(v_d_acc)
                    history['Train Lang Acc'].append(t_l_acc);   history['Val Lang Acc'].append(v_l_acc)
                    history['Train Digit Loss'].append(t_d_loss); history['Val Digit Loss'].append(v_d_loss)
                    history['Train Lang Loss'].append(t_l_loss);  history['Val Lang Loss'].append(v_l_loss)

                    # Update live charts
                    loss_ph.line_chart(pd.DataFrame({
                        'Train Total': history['Train Loss'],
                        'Val Total':   history['Val Loss'],
                        'Train Digit': history['Train Digit Loss'],
                        'Val Digit':   history['Val Digit Loss'],
                        'Train Lang':  history['Train Lang Loss'],
                        'Val Lang':    history['Val Lang Loss'],
                    }))
                    dacc_ph.line_chart(pd.DataFrame({
                        'Train': history['Train Digit Acc'],
                        'Val':   history['Val Digit Acc'],
                    }))
                    lacc_ph.line_chart(pd.DataFrame({
                        'Train': history['Train Lang Acc'],
                        'Val':   history['Val Lang Acc'],
                    }))

                    metrics_table.markdown(f"""
| Metric | Training | Validation |
|---|---|---|
| **Total Loss** | {t_loss:.4f} | {v_loss:.4f} |
| **Digit Loss** | {t_d_loss:.4f} | {v_d_loss:.4f} |
| **Lang Loss** | {t_l_loss:.4f} | {v_l_loss:.4f} |
| **Digit Acc** | {t_d_acc:.2f}% | {v_d_acc:.2f}% |
| **Lang Acc** | {t_l_acc:.2f}% | {v_l_acc:.2f}% |
                    """)
                    progress_bar.progress((epoch + 1) / int(epochs))

                status_text.success(
                    f"Training complete! Saved to `models/best_model_multitask.pt` "
                    f"(Best Val Loss: {best_val_loss:.4f})"
                )

                # ── Final training dashboard (full 6-panel figure) ────
                st.markdown("---")
                st.markdown("### Training Summary Dashboard")
                dash_fig = plot_multitask_training_history(history)
                st.pyplot(dash_fig)
                plt.close(dash_fig)

                # ── Test evaluation ───────────────────────────────────
                st.markdown("---")
                st.markdown("### Test Set Evaluation")
                with st.spinner("Evaluating on test set…"):
                    model.load_state_dict(torch.load('models/best_model_multitask.pt', map_location=device))
                    t_loss, t_d_acc, t_l_acc, _, _ = validate_multitask(model, test_loader, criterion, device)
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Digit Accuracy",    f"{t_d_acc:.2f}%")
                    mc2.metric("Language Accuracy", f"{t_l_acc:.2f}%")
                    mc3.metric("Total Loss",        f"{t_loss:.4f}")

                # ── Confusion matrices ────────────────────────────────
                st.markdown("---")
                st.markdown("### Confusion Matrices")
                with st.spinner("Computing…"):
                    d_true, d_pred_lst, l_true, l_pred_lst = \
                        collect_predictions_multitask(model, test_loader, device)

                    cm_col1, cm_col2 = st.columns(2)
                    with cm_col1:
                        st.markdown("#### Digit Classification")
                        all_lbl = sorted(set(d_true) | set(d_pred_lst))
                        cn = ["Empty" if l == 0 else str(l) for l in all_lbl]
                        st.pyplot(plot_confusion_matrix(d_true, d_pred_lst, cn))
                        plt.close('all')
                    with cm_col2:
                        st.markdown("#### Language Classification")
                        st.pyplot(plot_lang_confusion_matrix(l_true, l_pred_lst))
                        plt.close('all')

                    # Per-class digit accuracy table
                    cm_arr = confusion_matrix(d_true, d_pred_lst, labels=all_lbl)
                    pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
                    st.markdown("#### Per-Class Digit Accuracy")
                    st.dataframe(pd.DataFrame({
                        'Class': cn,
                        'Correct': cm_arr.diagonal(),
                        'Total':   cm_arr.sum(axis=1),
                        'Accuracy (%)': [f"{a:.1f}" for a in pca],
                    }).set_index('Class'), width='stretch')

                # ── Save report ───────────────────────────────────────
                rpt_path = save_training_report_multitask(
                    history=history,
                    test_digit_loss=t_loss,
                    test_digit_acc=t_d_acc,
                    test_lang_acc=t_l_acc,
                    digit_true=d_true, digit_pred=d_pred_lst,
                    lang_true=l_true,  lang_pred=l_pred_lst,
                    lang_balance_info=balance_info,
                    dataset_mode="MNIST+Fonts+Hoda [multi-task]",
                    epochs=int(epochs),
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    best_val_loss=best_val_loss,
                    model=model,
                    output_path='models/training_report_multitask.txt',
                )
                st.success(f"Report saved to `{rpt_path}`")
                with open(rpt_path, 'r', encoding='utf-8') as f:
                    st.download_button("⬇️ Download Multi-Task Report (.txt)",
                                       f.read(), 'training_report_multitask.txt', 'text/plain')

            else:
                # ======================================================
                # SINGLE-TASK PATH (unchanged)
                # ======================================================
                if dataset_mode.startswith("MNIST + Fonts + Hoda"):
                    train_loader, val_loader, test_loader = get_dataloaders_all(data_path, batch_size=batch_size)
                elif dataset_mode.startswith("MNIST + Hoda"):
                    train_loader, val_loader, test_loader = get_dataloaders_mnist_hoda(data_path, batch_size=batch_size)
                elif dataset_mode.startswith("MNIST Only"):
                    train_loader, val_loader, test_loader = get_dataloaders_mnist_only(batch_size=batch_size)
                else:
                    train_loader, val_loader, test_loader = get_dataloaders(data_path, batch_size=batch_size)

                model     = DigitCNN(num_classes=10).to(device)
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

                progress_bar  = st.progress(0)
                status_text   = st.empty()
                metrics_table = st.empty()

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("#### Loss Curve")
                    loss_placeholder = st.empty()
                with chart_col2:
                    st.markdown("#### Accuracy Curve")
                    acc_placeholder = st.empty()

                best_val_loss = float('inf')
                history = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}

                for epoch in range(int(epochs)):
                    status_text.markdown(f"**Running Epoch {epoch + 1}/{epochs}...**")

                    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc     = validate(model, val_loader, criterion, device)
                    scheduler.step(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), 'models/best_model.pt')

                    history['Train Loss'].append(train_loss); history['Val Loss'].append(val_loss)
                    history['Train Acc'].append(train_acc);   history['Val Acc'].append(val_acc)

                    loss_df = pd.DataFrame({'Train Loss': history['Train Loss'], 'Val Loss': history['Val Loss']})
                    acc_df  = pd.DataFrame({'Train Acc':  history['Train Acc'],  'Val Acc':  history['Val Acc']})
                    loss_placeholder.line_chart(loss_df)
                    acc_placeholder.line_chart(acc_df)

                    metrics_table.markdown(f"""
| Metric | Training | Validation |
|---|---|---|
| **Loss** | {train_loss:.4f} | {val_loss:.4f} |
| **Accuracy** | {train_acc:.2f}% | {val_acc:.2f}% |
                    """)
                    progress_bar.progress((epoch + 1) / int(epochs))

                status_text.success(
                    f"Training Complete! Best model saved to `models/best_model.pt` "
                    f"(Best Val Loss: {best_val_loss:.4f})"
                )

                st.markdown("---")
                st.markdown("### Test Set Evaluation")
                with st.spinner("Evaluating on test set..."):
                    model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
                    test_loss, test_acc = validate(model, test_loader, criterion, device)
                    st.metric(label="Final Test Accuracy", value=f"{test_acc:.2f}%",
                              delta=f"Loss: {test_loss:.4f}", delta_color="inverse")

                st.markdown("---")
                st.markdown("### Confusion Matrix")
                st.markdown(
                    "Diagonal = correct predictions. "
                    "Off-diagonal = confusion pairs. Each cell shows count and row %."
                )
                with st.spinner("Computing confusion matrix on test set..."):
                    y_true, y_pred = collect_predictions(model, test_loader, device)
                    all_labels  = sorted(set(y_true) | set(y_pred))
                    class_names = ["Empty" if lbl == 0 else str(lbl) for lbl in all_labels]
                    cm_fig      = plot_confusion_matrix(y_true, y_pred, class_names)
                    st.pyplot(cm_fig)
                    plt.close(cm_fig)

                    cm_arr        = confusion_matrix(y_true, y_pred, labels=all_labels)
                    per_class_acc = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
                    acc_df_show   = pd.DataFrame({
                        'Class':   class_names,
                        'Correct': cm_arr.diagonal(),
                        'Total':   cm_arr.sum(axis=1),
                        'Accuracy (%)': [f"{a:.1f}" for a in per_class_acc],
                    })
                    st.markdown("#### Per-Class Accuracy")
                    st.dataframe(acc_df_show.set_index('Class'), width='stretch')

                train_report_path = save_training_report(
                    history=history, test_loss=test_loss, test_acc=test_acc,
                    y_true=y_true, y_pred=y_pred, dataset_mode=dataset_mode,
                    epochs=int(epochs), learning_rate=learning_rate, batch_size=batch_size,
                    best_val_loss=best_val_loss, model=model,
                    output_path='models/training_report.txt',
                )
                st.success(f"📄 Training report saved to `{train_report_path}`")
                with open(train_report_path, 'r', encoding='utf-8') as f:
                    st.download_button(label="⬇️ Download Training Report (.txt)",
                                       data=f.read(), file_name='training_report.txt',
                                       mime='text/plain')

        except Exception as e:
            import traceback
            st.error(f"Training Error: {e}")
            st.code(traceback.format_exc())

# ==========================================
# MODE 3: MODEL OPTIMIZATION
# ==========================================
elif app_mode == "Model Optimization":
    import time

    st.markdown("### Model Optimization & Benchmarking")
    st.markdown("Export a trained model to TorchScript + ONNX. Artifacts saved next to the source `.pt` file.")

    cpu_device = torch.device('cpu')

    # ── Auto-discover available trained models ─────────────────────
    _opt_candidates = {
        "English / Default (DigitCNN)":         ("models/best_model.pt",          "single"),
        "English Only (DigitCNN)":               ("models/best_model_english.pt",  "single"),
        "Persian Only (DigitCNN)":               ("models/best_model_persian.pt",  "single"),
        "Multi-Task CNN":                        ("models/best_model_multitask.pt","multi"),
        "Unified 20-Class (MobileNet/Shuffle)":  ("models/best_model_unified20.pt","unified"),
    }
    available = {k: v for k, v in _opt_candidates.items() if os.path.exists(v[0])}

    if not available:
        st.error("No trained models found. Train at least one model first.")
    else:
        opt_choice = st.selectbox("Model to optimize:", list(available.keys()))
        opt_pt_path, opt_type = available[opt_choice]

        # Output prefix = same stem as the .pt file
        opt_prefix = opt_pt_path.replace('.pt', '')

        st.info(f"Source: `{opt_pt_path}`  →  exports to `{opt_prefix}.ts` and `{opt_prefix}.onnx`")

        if opt_type == "unified":
            _opt_bb = st.selectbox("Backbone (must match what was trained)",
                                   ["mobilenet_v3_small", "shufflenet_v2_x0_5"])

        if st.button("Run Optimization & Benchmark"):
            with st.spinner("Loading model and exporting…"):
                if opt_type == "single":
                    base_model = DigitCNN(num_classes=10)
                    base_model.load_state_dict(torch.load(opt_pt_path, map_location=cpu_device))
                    base_model.to(cpu_device).eval()
                    results = run_optimization_and_benchmark(
                        base_model, opt_pt_path, cpu_device,
                        output_prefix=opt_prefix,
                    )
                    last_onnx_ms = results[-1].get("ms/cell", "N/A")
                    if last_onnx_ms == "N/A":
                        st.warning("onnxruntime not installed — ONNX benchmark skipped. `pip install onnxruntime`")
                    else:
                        st.success("Export complete. ONNX verified.")
                    st.dataframe(pd.DataFrame(results), width='stretch')
                    st.markdown(
                        f"**TorchScript**: `torch.jit.load('{opt_prefix}.ts')`  \n"
                        f"**ONNX**: `onnxruntime.InferenceSession('{opt_prefix}.onnx')`"
                    )

                elif opt_type in ("multi", "unified"):
                    if opt_type == "multi":
                        base_model = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2)
                    else:
                        base_model = UnifiedDigitCNN(backbone=_opt_bb, pretrained=False,
                                                     num_classes=UNIFIED_NUM_CLASSES)
                    base_model.load_state_dict(torch.load(opt_pt_path, map_location=cpu_device))
                    base_model.to(cpu_device).eval()
                    results, verif = run_optimization_and_benchmark_multitask(
                        base_model, opt_pt_path, cpu_device,
                        output_prefix=opt_prefix,
                    )
                    if verif.get('error'):
                        st.warning("onnxruntime not installed — ONNX benchmark skipped.")
                    elif not verif['digit_match'] or not verif['lang_match']:
                        st.error(f"ONNX head mismatch — digit={verif['digit_match']} lang={verif['lang_match']}")
                    else:
                        st.success("ONNX verified: both heads match PyTorch.")
                    st.dataframe(pd.DataFrame(results), width='stretch')
                    st.markdown(
                        f"**TorchScript**: `torch.jit.load('{opt_prefix}.ts')`  \n"
                        f"**ONNX**: `onnxruntime.InferenceSession('{opt_prefix}.onnx')` "
                        f"— outputs: `digit_output`, `lang_output`"
                    )