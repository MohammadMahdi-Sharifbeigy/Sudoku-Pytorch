"""Mode 2 — Model Training Dashboard.

Architecture: training runs in a daemon background thread.
Main Streamlit thread stays responsive → stop button works instantly.

State machine (st.session_state['_train_state']):
  'idle'      → show config form
  'running'   → show live progress + stop button (polls with st.rerun())
  'done'      → show results / confusion matrix
  'cancelled' → show warning + reset form
  'error'     → show traceback + reset form
"""
import os
import queue
import threading
import time as _time
import traceback as _traceback

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src.train_runner import TrainingConfig, run_training, RunCancelled
from src.report_utils import save_training_report, save_training_report_multitask
from app.cancel import clear_cancel, get_stop_event, request_cancel
from app.plot_utils import (
    plot_confusion_matrix, plot_lang_confusion_matrix,
    plot_multitask_training_history, plot_singletask_summary,
    make_run_dir, save_fig, save_run_metadata,
)

# ── Session-state keys ────────────────────────────────────────────────────────
_STATE   = "_train_state"
_HISTORY = "_train_history"
_THREAD  = "_train_thread"
_RESULT  = "_train_result"
_CONFIG  = "_train_config"
_ERROR   = "_train_error"
_DEVICE  = "_train_device"

SCOPE = "training"


def render_training_page(device: torch.device) -> None:
    st.markdown("### Model Training Dashboard")

    state = st.session_state.get(_STATE, "idle")

    if state == "idle":
        _render_config_form(device)

    elif state == "running":
        _render_running()

    elif state == "done":
        _render_done(device)

    elif state == "cancelled":
        st.warning("Training stopped by user.")
        if st.button("Start new training"):
            _reset()
            st.rerun()

    elif state == "error":
        st.error("Training error:")
        st.code(st.session_state.get(_ERROR, "unknown error"))
        if st.button("Reset"):
            _reset()
            st.rerun()


# ── Config form ───────────────────────────────────────────────────────────────

def _render_config_form(device: torch.device) -> None:
    st.markdown("Configure hyperparameters and start training.")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    with param_col2:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
    with param_col3:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        weight_decay = st.number_input("AdamW Weight Decay", value=1e-4, format="%.6f")
    with opt_col2:
        lr_schedule = st.selectbox(
            "LR Schedule",
            ["ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "CosineAnnealing", "OneCycleLR"],
            help=(
                "ReduceLROnPlateau: halves LR when val loss stagnates. "
                "CosineAnnealingWarmRestarts: cosine cycles with restarts (T_0, T_mult). "
                "CosineAnnealing: single cosine decay over all epochs. "
                "OneCycleLR: aggressive warmup+decay."
            ),
        )

    # Cosine-specific params (shown only when relevant)
    cosine_t0, cosine_t_mult, cosine_eta_min = 10, 1, 1e-6
    if lr_schedule in ("CosineAnnealingWarmRestarts", "CosineAnnealing"):
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            cosine_t0 = st.number_input(
                "T_0 (first cycle epochs)" if "Warm" in lr_schedule else "T_max (total epochs)",
                min_value=1, max_value=100, value=10,
                help="For WarmRestarts: epochs before first restart. Recommended: epochs÷2 (e.g. 10 for 20 epochs).",
            )
        with cs2:
            cosine_t_mult = st.number_input(
                "T_mult (cycle multiplier)", min_value=1, max_value=4, value=1,
                help="1 = equal cycles. 2 = each restart doubles cycle length (5→10→20).",
            ) if "Warm" in lr_schedule else 1
        with cs3:
            cosine_eta_min = st.number_input(
                "eta_min (LR floor)", value=1e-6, format="%.2e",
                help="Minimum LR at bottom of cosine. Recommended: lr/1000 (e.g. 1e-6 for lr=1e-3).",
            )

    data_path = st.text_input("Dataset Directory Path", value="data")

    st.markdown("---")
    st.markdown("#### Training Configuration")

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        model_choice = st.selectbox("Model", [
            "DigitCNN", "MultiTaskCNN",
            "UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2",
            "EfficientNetDigit",
        ])
    with sel_col2:
        purpose = st.selectbox("Purpose", ["English", "Persian", "Multi"])

    sel_col3, sel_col4 = st.columns(2)
    with sel_col3:
        if purpose == "English":
            dataset_opts = ["MNIST Only", "MNIST + Fonts", "MNIST + Fonts + Hoda (All)"]
            dataset_default = 1
        elif purpose == "Persian":
            dataset_opts = ["Persian Only (Hoda)", "MNIST + Fonts + Hoda (All)"]
            dataset_default = 0
        else:
            dataset_opts = ["MNIST + Fonts + Hoda (All)", "MNIST + Hoda"]
            dataset_default = 0
        dataset_choice = st.selectbox("Dataset", dataset_opts, index=dataset_default)

    with sel_col4:
        aug_choice = st.selectbox(
            "Augmentation", ["full", "light", "none"], index=0,
            help="none: safest for odd digits. light: affine only. full: all transforms.",
        )

    multi_mode = None
    if purpose == "Multi" and model_choice != "MultiTaskCNN":
        multi_mode = st.radio(
            "Multi-script mode",
            ["Unified model (20-class)", "Separate models (Persian + English)"],
            horizontal=True,
        )

    eff_pretrained, eff_phase1_epochs, eff_phase2_epochs = True, 6, 14
    if model_choice == "EfficientNetDigit":
        eff_pretrained = st.toggle("Use ImageNet pretrained weights", value=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            eff_phase1_epochs = st.number_input("Phase 1 epochs (head)", min_value=1, max_value=30, value=6)
        with ec2:
            eff_phase2_epochs = st.number_input("Phase 2 epochs (fine-tune)", min_value=1, max_value=50, value=14)

    unified_backbone, unified_pretrained = "mobilenet_v3_small", False
    if model_choice in ("UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2"):
        unified_backbone   = "mobilenet_v3_small" if "MobileNet" in model_choice else "shufflenet_v2_x0_5"
        unified_pretrained = st.toggle("Use ImageNet pretrained weights", value=False)

    if not st.button("Start Training", use_container_width=True):
        return

    if not os.path.exists(data_path):
        st.error(f"Dataset path `{data_path}` does not exist.")
        return

    config = TrainingConfig(
        model_choice=model_choice, purpose=purpose,
        dataset_choice=dataset_choice, aug_preset=aug_choice,
        epochs=int(epochs), learning_rate=learning_rate,
        batch_size=batch_size, weight_decay=weight_decay,
        lr_schedule=lr_schedule, data_path=data_path,
        eff_pretrained=eff_pretrained,
        eff_phase1_epochs=int(eff_phase1_epochs),
        eff_phase2_epochs=int(eff_phase2_epochs),
        unified_backbone=unified_backbone,
        unified_pretrained=unified_pretrained,
        multi_mode=multi_mode or "Unified model (20-class)",
        cosine_t0=int(cosine_t0),
        cosine_t_mult=int(cosine_t_mult),
        cosine_eta_min=float(cosine_eta_min),
    )

    _launch(config, device)
    st.rerun()


# ── Session-state sub-keys for queues ─────────────────────────────────────────
_EPOCH_Q  = "_train_epoch_q"   # queue.Queue — thread posts epoch dicts
_RESULT_Q = "_train_result_q"  # queue.Queue — thread posts final result dict


# ── Launch training thread ────────────────────────────────────────────────────

def _launch(config: TrainingConfig, device: torch.device) -> None:
    stop_ev = get_stop_event(SCOPE)
    stop_ev.clear()

    epoch_q  = queue.Queue()   # thread → main: per-epoch updates
    result_q = queue.Queue()   # thread → main: final result (one item)

    def on_epoch(update: dict):
        epoch_q.put(update)

    def worker():
        try:
            result = run_training(config, device, on_epoch=on_epoch, stop_event=stop_ev)
            # Strip non-serialisable PyTorch objects before queuing
            result_q.put({k: v for k, v in result.items()
                          if k not in ('model', 'test_loader', 'criterion',
                                       'persian', 'english')
                          } | {
                # keep sub-results but strip their heavy objects too
                'persian': {k: v for k, v in result.get('persian', {}).items()
                            if k not in ('model', 'test_loader', 'criterion')}
                            if 'persian' in result else {},
                'english': {k: v for k, v in result.get('english', {}).items()
                            if k not in ('model', 'test_loader', 'criterion')}
                            if 'english' in result else {},
            })
        except RunCancelled:
            result_q.put({'type': 'cancelled'})
        except Exception:
            result_q.put({'type': 'error', 'error': _traceback.format_exc()})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    st.session_state[_STATE]    = "running"
    st.session_state[_THREAD]   = t
    st.session_state[_EPOCH_Q]  = epoch_q    # thread writes, main drains
    st.session_state[_RESULT_Q] = result_q   # thread writes one item
    st.session_state[_HISTORY]  = []         # main thread accumulates here
    st.session_state[_RESULT]   = {}
    st.session_state[_CONFIG]   = config
    st.session_state[_DEVICE]   = device
    st.session_state[_ERROR]    = None


# ── Running state ─────────────────────────────────────────────────────────────

def _render_running() -> None:
    config: TrainingConfig    = st.session_state.get(_CONFIG)
    epoch_q: queue.Queue      = st.session_state.get(_EPOCH_Q)
    result_q: queue.Queue     = st.session_state.get(_RESULT_Q)
    thread: threading.Thread  = st.session_state.get(_THREAD)

    # Drain epoch queue → accumulate into session_state history
    if epoch_q:
        while not epoch_q.empty():
            try:
                st.session_state[_HISTORY].append(epoch_q.get_nowait())
            except queue.Empty:
                break

    history: list = st.session_state.get(_HISTORY, [])

    # Stop button
    if st.sidebar.button("⏹ Stop Training", type="primary",
                         use_container_width=True, key="stop_training_active"):
        request_cancel(SCOPE)
        st.sidebar.warning("Stopping after current batch…")

    config_epochs = config.epochs if config else 1
    total = (config.eff_phase1_epochs + config.eff_phase2_epochs
             if config and config.model_choice == "EfficientNetDigit"
             else config_epochs)

    n_done = len(history)

    # ── Live progress header ──────────────────────────────────────────────────
    st.progress(min(n_done / max(total, 1), 1.0),
                text=f"Epoch {n_done}/{total}")

    if not history:
        st.info("Initialising — loading data, building model…")
    else:
        latest = history[-1]
        phase_str = (f"  `{latest['phase']}`"
                     if latest.get('phase') not in ('train', None, '') else "")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Train Loss", f"{latest['train_loss']:.4f}")
        col_b.metric("Train Acc",  f"{latest['train_acc']:.1f}%")
        col_c.metric("Val Loss",   f"{latest['val_loss']:.4f}")
        col_d.metric("Val Acc",    f"{latest['val_acc']:.1f}%")

        if phase_str:
            st.caption(f"Phase:{phase_str}")

        # ── Live charts ───────────────────────────────────────────────────────
        epochs_idx = [h['epoch'] + 1 for h in history]

        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("**Loss**")
            st.line_chart(pd.DataFrame({
                'Train Loss': [h['train_loss'] for h in history],
                'Val Loss':   [h['val_loss']   for h in history],
            }, index=epochs_idx))
        with lc2:
            st.markdown("**Accuracy (%)**")
            st.line_chart(pd.DataFrame({
                'Train Acc': [h['train_acc'] for h in history],
                'Val Acc':   [h['val_acc']   for h in history],
            }, index=epochs_idx))

        if any(h.get('train_lang_acc') for h in history):
            st.markdown("**Language Accuracy (%)**")
            st.line_chart(pd.DataFrame({
                'Train Lang Acc': [h.get('train_lang_acc', 0) for h in history],
                'Val Lang Acc':   [h.get('val_lang_acc',   0) for h in history],
            }, index=epochs_idx))

    # ── Check for completion ──────────────────────────────────────────────────
    final = None
    if result_q:
        try:
            final = result_q.get_nowait()
        except queue.Empty:
            pass

    if final is not None:
        if final['type'] == 'cancelled':
            st.session_state[_STATE] = "cancelled"
        elif final['type'] == 'error':
            st.session_state[_STATE] = "error"
            st.session_state[_ERROR] = final['error']
        else:
            st.session_state[_STATE]  = "done"
            st.session_state[_RESULT] = final
        st.rerun()
        return

    if thread and not thread.is_alive():
        # Thread died without posting to result_q
        st.session_state[_STATE] = "done"
        st.rerun()
        return

    # Poll every 0.4 s — main thread stays responsive for stop button
    _time.sleep(0.4)
    st.rerun()


# ── Done state — render results ───────────────────────────────────────────────

def _render_done(device: torch.device) -> None:
    from sklearn.metrics import confusion_matrix as sk_cm

    result  = st.session_state.get(_RESULT, {})
    config: TrainingConfig = st.session_state.get(_CONFIG)
    history = st.session_state.get(_HISTORY, [])

    if not result or not isinstance(result, dict) or result.get('type') == 'done' and not result.get('save_path'):
        st.warning("No results available.")
        _render_reset_button()
        return

    model_type = result.get('model_type', '')
    st.success(f"✅ Training complete — saved to `{result.get('save_path', '?')}`")
    st.markdown(f"Best val loss: `{result.get('best_val_loss', 0):.4f}`")

    # ── Training curves ───────────────────────────────────────────────────────
    if history:
        st.markdown("---")
        st.markdown("### Training Curves")
        epochs_idx = [h['epoch'] + 1 for h in history]

        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("**Loss**")
            st.line_chart(pd.DataFrame({
                'Train Loss': [h['train_loss'] for h in history],
                'Val Loss':   [h['val_loss']   for h in history],
            }, index=epochs_idx))
        with lc2:
            st.markdown("**Accuracy (%)**")
            st.line_chart(pd.DataFrame({
                'Train Acc': [h['train_acc'] for h in history],
                'Val Acc':   [h['val_acc']   for h in history],
            }, index=epochs_idx))

        if any(h.get('train_lang_acc') for h in history):
            st.markdown("**Language Accuracy (%)**")
            st.line_chart(pd.DataFrame({
                'Train Lang': [h.get('train_lang_acc', 0) for h in history],
                'Val Lang':   [h.get('val_lang_acc',   0) for h in history],
            }, index=epochs_idx))

    # ── Test metrics ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Test Set Results")

    if model_type == "multitask":
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Digit Accuracy", f"{result.get('test_digit_acc', 0):.2f}%")
        mc2.metric("Language Accuracy", f"{result.get('test_lang_acc', 0):.2f}%")
        mc3.metric("Test Loss", f"{result.get('test_loss', 0):.4f}")
    elif model_type == "separate":
        per = result.get('persian', {})
        eng = result.get('english', {})
        mc1, mc2 = st.columns(2)
        mc1.metric("Persian Test Acc", f"{per.get('test_acc', 0):.2f}%")
        mc2.metric("English Test Acc", f"{eng.get('test_acc', 0):.2f}%")
    else:
        mc1, mc2 = st.columns(2)
        mc1.metric("Test Accuracy", f"{result.get('test_acc', 0):.2f}%")
        mc2.metric("Test Loss", f"{result.get('test_loss', 0):.4f}", delta_color="inverse")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    def _draw_cm(y_true, y_pred, title="Confusion Matrix", unified=False):
        if not y_true or not y_pred:
            return
        st.markdown(f"#### {title}")
        all_lbl = sorted(set(y_true) | set(y_pred))
        if unified:
            def _n(c):
                return ("ENG_0" if c == 0 else f"ENG_{c}" if c < 10
                        else "PER_0" if c == 10 else f"PER_{c-10}")
            class_names = [_n(c) for c in all_lbl]
        else:
            class_names = ["Empty" if l == 0 else str(l) for l in all_lbl]

        cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.pyplot(cm_fig)
        plt.close(cm_fig)

        cm_arr = sk_cm(y_true, y_pred, labels=all_lbl)
        pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
        st.dataframe(pd.DataFrame({
            'Correct': cm_arr.diagonal(),
            'Total':   cm_arr.sum(axis=1),
            'Acc (%)': [f"{a:.1f}" for a in pca],
        }, index=class_names), use_container_width=True)

    st.markdown("---")
    st.markdown("### Confusion Matrix")

    y_true = result.get('y_true', [])
    y_pred = result.get('y_pred', [])
    _draw_cm(y_true, y_pred, "Digit Predictions", unified=(model_type == "unified"))

    if model_type == "multitask":
        st.markdown("---")
        l_true = result.get('l_true', [])
        l_pred = result.get('l_pred', [])
        if l_true:
            st.markdown("#### Language Classification")
            fig = plot_lang_confusion_matrix(l_true, l_pred)
            st.pyplot(fig)
            plt.close(fig)

    if model_type == "separate":
        per = result.get('persian', {})
        eng = result.get('english', {})
        pc1, pc2 = st.columns(2)
        with pc1:
            _draw_cm(per.get('y_true', []), per.get('y_pred', []), "Persian")
        with pc2:
            _draw_cm(eng.get('y_true', []), eng.get('y_pred', []), "English")

    # ── Download report ───────────────────────────────────────────────────────
    st.markdown("---")
    for rpt_key in ('report_path',):
        rpt = result.get(rpt_key)
        if rpt and os.path.exists(rpt):
            with open(rpt, 'r', encoding='utf-8') as f:
                st.download_button("⬇ Download Training Report (.txt)",
                                   f.read(), os.path.basename(rpt), 'text/plain')

    _render_reset_button()


def _render_reset_button():
    st.markdown("---")
    if st.button("Train another model"):
        _reset()
        st.rerun()


def _reset():
    for key in [_STATE, _THREAD, _HISTORY, _RESULT, _CONFIG, _ERROR, _DEVICE,
                _EPOCH_Q, _RESULT_Q]:
        st.session_state.pop(key, None)
    clear_cancel(SCOPE)
