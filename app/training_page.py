"""Mode 2 — Model Training Dashboard."""
import os
import time as _time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from src.model import (
    DigitCNN, FocalLoss,
    MultiTaskDigitCNN, MultiTaskFocalLoss,
    UnifiedDigitCNN, UNIFIED_NUM_CLASSES,
    decode_unified_class,
    EfficientNetDigitCNN,
)
from src.data_utils import (
    get_dataloaders, get_dataloaders_all, get_dataloaders_mnist_hoda,
    get_dataloaders_mnist_only, get_dataloaders_multitask,
    get_dataloaders_persian, get_dataloaders_english, get_dataloaders_unified20,
)
from src.train import (
    train_epoch, validate, collect_predictions,
    train_epoch_multitask, validate_multitask, collect_predictions_multitask,
)
from src.report_utils import save_training_report, save_training_report_multitask
from app.cancel import RunCancelled, clear_cancel, raise_if_cancelled, render_stop_button
from app.plot_utils import (
    plot_confusion_matrix, plot_lang_confusion_matrix,
    plot_multitask_training_history, plot_singletask_summary,
    plot_lr_history, plot_time_per_epoch,
    make_run_dir, save_fig, save_run_metadata,
)


def render_training_page(device: torch.device) -> None:
    st.markdown("### Model Training Dashboard")
    st.markdown("Configure hyperparameters and monitor the CNN training in real-time.")
    render_stop_button("training", "Stop training")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    with param_col2:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
    with param_col3:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        weight_decay = st.number_input(
            "AdamW Weight Decay", value=1e-4, format="%.6f",
            help="L2 regularisation. 1e-4 safe default.",
        )
    with opt_col2:
        lr_schedule = st.selectbox(
            "LR Schedule",
            ["ReduceLROnPlateau", "CosineAnnealing", "OneCycleLR"],
            help="ReduceLROnPlateau: halves LR on stagnation. CosineAnnealing: smooth decay. OneCycleLR: aggressive warmup.",
        )

    data_path = st.text_input("Dataset Directory Path", value="data")

    st.markdown("---")
    st.markdown("#### Training Configuration")

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        model_choice = st.selectbox(
            "Model",
            [
                "DigitCNN",
                "MultiTaskCNN",
                "UnifiedCNN — MobileNetV3",
                "UnifiedCNN — ShuffleNetV2",
                "EfficientNetDigit",
            ],
            help=(
                "DigitCNN: lightweight 3-block CNN (~250K params). "
                "MultiTaskCNN: shared backbone + digit/language heads. "
                "UnifiedCNN: 20-class unified digit+language. "
                "EfficientNetDigit: transfer learning from ImageNet, strongest baseline."
            ),
        )
    with sel_col2:
        purpose = st.selectbox(
            "Purpose",
            ["English", "Persian", "Multi"],
            help="English: MNIST/fonts. Persian: Hoda. Multi: both scripts together.",
        )

    sel_col3, sel_col4 = st.columns(2)
    with sel_col3:
        if purpose == "English":
            dataset_options = ["MNIST Only", "MNIST + Fonts", "MNIST + Fonts + Hoda (All)"]
            dataset_default = 1
        elif purpose == "Persian":
            dataset_options = ["Persian Only (Hoda)", "MNIST + Fonts + Hoda (All)"]
            dataset_default = 0
        else:  # Multi
            dataset_options = ["MNIST + Fonts + Hoda (All)", "MNIST + Hoda"]
            dataset_default = 0
        dataset_choice = st.selectbox("Dataset", dataset_options, index=dataset_default)

    with sel_col4:
        aug_choice = st.selectbox(
            "Augmentation",
            ["full", "light", "none"],
            index=0,
            help=(
                "none: no geometric distortion — best for 28×28 odd digits (1,3,5,7,9). "
                "light: affine only. "
                "full: all transforms (recommended for EfficientNet at 224×224)."
            ),
        )

    # Multi-script sub-option
    if purpose == "Multi" and model_choice not in ("MultiTaskCNN",):
        multi_mode = st.radio(
            "Multi-script mode",
            ["Unified model (20-class)", "Separate models (Persian + English)"],
            horizontal=True,
        )
    else:
        multi_mode = None

    # EfficientNet-specific controls
    if model_choice == "EfficientNetDigit":
        eff_pretrained = st.toggle(
            "Use ImageNet pretrained weights", value=True,
            help="Strongly recommended — core advantage of EfficientNet.",
        )
        eff_col1, eff_col2 = st.columns(2)
        with eff_col1:
            eff_phase1_epochs = st.number_input("Phase 1 epochs (head only)", min_value=1, max_value=30, value=6)
        with eff_col2:
            eff_phase2_epochs = st.number_input("Phase 2 epochs (fine-tune)", min_value=1, max_value=50, value=14)
    else:
        eff_pretrained = True
        eff_phase1_epochs = 6
        eff_phase2_epochs = 14

    # UnifiedCNN backbone selector
    if model_choice in ("UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2"):
        unified_backbone = "mobilenet_v3_small" if "MobileNet" in model_choice else "shufflenet_v2_x0_5"
        unified_pretrained = st.toggle("Use ImageNet pretrained weights", value=False)
    else:
        unified_backbone = "mobilenet_v3_small"
        unified_pretrained = False

    if not st.button("Start Training Sequence", use_container_width=True):
        return

    clear_cancel("training")
    if not os.path.exists(data_path):
        st.error(f"Dataset path `{data_path}` does not exist.")
        st.stop()

    st.info("Initialising DataLoaders…")
    os.makedirs('models', exist_ok=True)

    try:
        if model_choice == "EfficientNetDigit":
            _run_efficientnet(
                device, data_path, batch_size, epochs, learning_rate,
                purpose, dataset_choice, aug_choice,
                eff_pretrained, eff_phase1_epochs, eff_phase2_epochs,
                weight_decay=weight_decay,
            )

        elif model_choice == "MultiTaskCNN":
            _run_multitask(
                device, data_path, batch_size, epochs, learning_rate,
                weight_decay=weight_decay, lr_schedule=lr_schedule,
            )

        elif model_choice in ("UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2"):
            dataset_mode_str = _map_dataset_to_legacy(dataset_choice)
            _run_unified(
                device, data_path, batch_size, epochs, learning_rate,
                unified_backbone, unified_pretrained, dataset_mode_str,
                weight_decay=weight_decay, lr_schedule=lr_schedule,
            )

        elif purpose == "Persian":
            _run_lang_specific(
                device, data_path, batch_size, epochs, learning_rate,
                "Persian Only (Hoda — saves best_model_persian.pt)", is_persian_only=True,
                weight_decay=weight_decay, lr_schedule=lr_schedule, aug_preset=aug_choice,
            )

        elif purpose == "English":
            dataset_mode_str = _map_dataset_to_legacy(dataset_choice)
            _run_singletask(
                device, data_path, batch_size, epochs, learning_rate, dataset_mode_str,
                weight_decay=weight_decay, lr_schedule=lr_schedule, aug_preset=aug_choice,
            )

        else:  # Multi + DigitCNN
            if multi_mode and "Separate" in multi_mode:
                st.info("Running Persian training…")
                _run_lang_specific(
                    device, data_path, batch_size, epochs, learning_rate,
                    "Persian Only (Hoda — saves best_model_persian.pt)", is_persian_only=True,
                    weight_decay=weight_decay, lr_schedule=lr_schedule, aug_preset=aug_choice,
                )
                st.info("Running English training…")
                _run_lang_specific(
                    device, data_path, batch_size, epochs, learning_rate,
                    "English Only (MNIST + Fonts — saves best_model_english.pt)", is_persian_only=False,
                    weight_decay=weight_decay, lr_schedule=lr_schedule, aug_preset=aug_choice,
                )
            else:
                _run_singletask(
                    device, data_path, batch_size, epochs, learning_rate,
                    "MNIST + Fonts + Hoda (All)",
                    weight_decay=weight_decay, lr_schedule=lr_schedule, aug_preset=aug_choice,
                )

    except RunCancelled as e:
        st.warning(str(e))
        clear_cancel("training")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        st.error(f"Training Error: {e}")
        st.code(traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _map_dataset_to_legacy(dataset_choice: str) -> str:
    """Map new UI dataset strings to legacy dataset_mode strings used by existing _run_* functions."""
    mapping = {
        "MNIST Only":               "MNIST Only",
        "MNIST + Fonts":            "MNIST + Fonts (Recommended for printed Sudoku)",
        "MNIST + Fonts + Hoda (All)": "MNIST + Fonts + Hoda (All)",
        "MNIST + Hoda":             "MNIST + Hoda",
        "Persian Only (Hoda)":      "Persian Only (Hoda — saves best_model_persian.pt)",
    }
    return mapping.get(dataset_choice, dataset_choice)


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser / scheduler factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_scheduler(optimizer, schedule: str, epochs: int, train_loader=None):
    if schedule == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if schedule == "OneCycleLR":
        steps = len(train_loader) if train_loader else 100
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'] * 10,
            steps_per_epoch=steps, epochs=epochs, pct_start=0.3,
        )
    # default: ReduceLROnPlateau
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


def _scheduler_step(scheduler, val_loss: float, schedule: str) -> None:
    """Step correctly depending on scheduler type."""
    if schedule == "OneCycleLR":
        pass   # OneCycleLR is stepped per batch inside the train loop
    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


# ──────────────────────────────────────────────────────────────────────────────
# Training paths
# ──────────────────────────────────────────────────────────────────────────────

def _run_unified(device, data_path, batch_size, epochs, lr, backbone, pretrained, dataset_mode,
                 weight_decay: float = 1e-4, lr_schedule: str = "ReduceLROnPlateau"):
    train_loader, val_loader, test_loader = get_dataloaders_unified20(data_path, batch_size=batch_size)
    model = UnifiedDigitCNN(backbone=backbone, pretrained=pretrained,
                             num_classes=UNIFIED_NUM_CLASSES).to(device)
    total_p, _ = model.param_count()
    st.info(f"Backbone: **{backbone}** | Params: **{total_p:,}** | Pretrained: {pretrained}")

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, lr_schedule, int(epochs))

    save_path    = 'models/best_model_unified20.pt'
    progress_bar = st.progress(0)
    status_text  = st.empty()
    lc1, lc2    = st.columns(2)
    with lc1:
        st.markdown("#### Loss Curve"); loss_ph = st.empty()
    with lc2:
        st.markdown("#### 20-Class Accuracy"); acc_ph = st.empty()
    metrics_tbl  = st.empty()
    best_val_loss = float('inf')
    history       = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}
    lr_history    = []
    time_history  = []

    for epoch in range(int(epochs)):
        status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")
        raise_if_cancelled("training")
        lr_history.append(optimizer.param_groups[0]['lr'])
        t0 = _time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                     should_stop=lambda: raise_if_cancelled("training"))
        v_loss, v_acc = validate(model, val_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        time_history.append(_time.time() - t0)
        _scheduler_step(scheduler, v_loss, lr_schedule)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
        history['Train Loss'].append(t_loss); history['Val Loss'].append(v_loss)
        history['Train Acc'].append(t_acc);   history['Val Acc'].append(v_acc)
        loss_ph.line_chart(pd.DataFrame({'Train': history['Train Loss'], 'Val': history['Val Loss']}))
        acc_ph.line_chart(pd.DataFrame({'Train': history['Train Acc'],  'Val': history['Val Acc']}))
        metrics_tbl.markdown(
            f"| Metric | Train | Val |\n|---|---|---|\n"
            f"| **Loss** | {t_loss:.4f} | {v_loss:.4f} |\n"
            f"| **20-class Acc** | {t_acc:.2f}% | {v_acc:.2f}% |  \n"
            f"| **LR** | {lr_history[-1]:.2e} | — |  \n"
            f"| **Epoch time** | {time_history[-1]:.1f}s | — |"
        )
        progress_bar.progress((epoch + 1) / int(epochs))

    status_text.success(f"Done! Saved to `{save_path}` (best val loss: {best_val_loss:.4f})")
    st.markdown("---"); st.markdown("### Test Set Evaluation")
    with st.spinner("Evaluating…"):
        model.load_state_dict(torch.load(save_path, map_location=device))
        t_loss, t_acc = validate(model, test_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        st.metric("20-class Test Accuracy", f"{t_acc:.2f}%",
                  delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

    st.markdown("---"); st.markdown("### Confusion Matrix (20 classes)")
    with st.spinner("Computing…"):
        y_true20, y_pred20 = collect_predictions(model, test_loader, device,
                                                   should_stop=lambda: raise_if_cancelled("training"))
        all_lbl = sorted(set(y_true20) | set(y_pred20))
        def _name(c):
            if c == 0:   return "ENG_0"
            if c < 10:   return f"ENG_{c}"
            if c == 10:  return "PER_0"
            return f"PER_{c - 10}"
        cn = [_name(c) for c in all_lbl]
        cm_fig = plot_confusion_matrix(y_true20, y_pred20, cn)
        st.pyplot(cm_fig); plt.close('all')
        digit_true = [decode_unified_class(c)[0] for c in y_true20]
        digit_pred = [decode_unified_class(c)[0] for c in y_pred20]
        d_acc = 100 * sum(t == p for t, p in zip(digit_true, digit_pred)) / max(len(digit_true), 1)
        st.metric("Digit Accuracy (folded to 0-9)", f"{d_acc:.2f}%")

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true20, y_pred=y_pred20,
        dataset_mode=dataset_mode + f" [{backbone}]",
        epochs=int(epochs), learning_rate=lr, batch_size=batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path='models/training_report_unified20.txt',
    )
    st.success(f"Report saved to `{rpt}`")
    with open(rpt, 'r', encoding='utf-8') as f:
        st.download_button("Download Unified Report (.txt)",
                           f.read(), 'training_report_unified20.txt', 'text/plain')

    # ── Save all plots ────────────────────────────────────────────────
    run_dir = make_run_dir(f"Unified20_{backbone}", int(epochs), batch_size, lr, weight_decay)
    save_fig(plot_singletask_summary(history, lr_history, time_history), run_dir, "01_summary.png")
    save_fig(plot_lr_history(lr_history),    run_dir, "02_lr_schedule.png")
    save_fig(plot_time_per_epoch(time_history), run_dir, "03_time_per_epoch.png")
    save_fig(cm_fig if plt.fignum_exists(cm_fig.number) else plot_confusion_matrix(y_true20, y_pred20, cn),
             run_dir, "04_confusion_matrix.png")
    save_run_metadata(run_dir, {
        "model": f"Unified20_{backbone}", "epochs": int(epochs),
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "lr_schedule": lr_schedule, "best_val_loss": best_val_loss,
        "test_acc": t_acc, "total_time_s": sum(time_history),
    })
    st.success(f"All plots saved to `{run_dir}/`")
    st.sidebar.success(f"Plots → `{run_dir}/`")


def _run_lang_specific(device, data_path, batch_size, epochs, lr, dataset_mode, is_persian_only,
                       weight_decay: float = 1e-4, lr_schedule: str = "ReduceLROnPlateau",
                       aug_preset: str = "full"):
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

    from src.data_utils import build_train_transform_preset
    train_loader.dataset.transform = build_train_transform_preset(aug_preset)

    model     = DigitCNN(num_classes=10).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, lr_schedule, int(epochs))

    progress_bar = st.progress(0); status_text = st.empty(); metrics_tbl = st.empty()
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f"#### Loss Curve ({lang_label})"); loss_ph = st.empty()
    with lc2:
        st.markdown(f"#### Accuracy Curve ({lang_label})"); acc_ph = st.empty()

    best_val_loss = float('inf')
    history      = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}
    lr_history   = []
    time_history = []

    for epoch in range(int(epochs)):
        status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")
        raise_if_cancelled("training")
        lr_history.append(optimizer.param_groups[0]['lr'])
        t0 = _time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                     should_stop=lambda: raise_if_cancelled("training"))
        v_loss, v_acc = validate(model, val_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        time_history.append(_time.time() - t0)
        _scheduler_step(scheduler, v_loss, lr_schedule)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
        history['Train Loss'].append(t_loss); history['Val Loss'].append(v_loss)
        history['Train Acc'].append(t_acc);   history['Val Acc'].append(v_acc)
        loss_ph.line_chart(pd.DataFrame({'Train': history['Train Loss'], 'Val': history['Val Loss']}))
        acc_ph.line_chart(pd.DataFrame({'Train': history['Train Acc'],  'Val': history['Val Acc']}))
        metrics_tbl.markdown(
            f"| Metric | Train | Val |\n|---|---|---|\n"
            f"| **Loss** | {t_loss:.4f} | {v_loss:.4f} |\n"
            f"| **Accuracy** | {t_acc:.2f}% | {v_acc:.2f}% |  \n"
            f"| **LR** | {lr_history[-1]:.2e} | — |  \n"
            f"| **Epoch time** | {time_history[-1]:.1f}s | — |"
        )
        progress_bar.progress((epoch + 1) / int(epochs))

    status_text.success(f"Training complete! Saved to `{save_path}`")
    st.markdown("---"); st.markdown("### Test Set Evaluation")
    with st.spinner("Evaluating…"):
        model.load_state_dict(torch.load(save_path, map_location=device))
        t_loss, t_acc = validate(model, test_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        st.metric(f"{lang_label} Test Accuracy", f"{t_acc:.2f}%",
                  delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

    st.markdown("---"); st.markdown("### Confusion Matrix")
    with st.spinner("Computing…"):
        y_true, y_pred = collect_predictions(model, test_loader, device,
                                              should_stop=lambda: raise_if_cancelled("training"))
        all_lbl     = sorted(set(y_true) | set(y_pred))
        class_names = ["Empty" if l == 0 else str(l) for l in all_lbl]
        cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.pyplot(cm_fig); plt.close('all')
        cm_arr = confusion_matrix(y_true, y_pred, labels=all_lbl)
        pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
        st.dataframe(pd.DataFrame({
            'Class': class_names, 'Correct': cm_arr.diagonal(),
            'Total': cm_arr.sum(axis=1), 'Accuracy (%)': [f"{a:.1f}" for a in pca],
        }).set_index('Class'), width='stretch')

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred, dataset_mode=dataset_mode,
        epochs=int(epochs), learning_rate=lr, batch_size=batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path=f'models/training_report_{lang_label.lower()}.txt',
    )
    st.success(f"Report saved to `{rpt}`")
    with open(rpt, 'r', encoding='utf-8') as f:
        st.download_button(f"Download {lang_label} Report (.txt)",
                           f.read(), f'training_report_{lang_label.lower()}.txt', 'text/plain')

    # ── Save all plots ────────────────────────────────────────────────
    run_dir = make_run_dir(f"DigitCNN_{lang_label}", int(epochs), batch_size, lr, weight_decay)
    save_fig(plot_singletask_summary(history, lr_history, time_history), run_dir, "01_summary.png")
    save_fig(plot_lr_history(lr_history),       run_dir, "02_lr_schedule.png")
    save_fig(plot_time_per_epoch(time_history),  run_dir, "03_time_per_epoch.png")
    save_fig(cm_fig, run_dir, "04_confusion_matrix.png")
    save_run_metadata(run_dir, {
        "model": f"DigitCNN_{lang_label}", "epochs": int(epochs),
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "lr_schedule": lr_schedule, "best_val_loss": best_val_loss,
        "test_acc": t_acc, "total_time_s": sum(time_history),
    })
    st.success(f"All plots saved to `{run_dir}/`")
    st.sidebar.success(f"Plots → `{run_dir}/`")


def _run_multitask(device, data_path, batch_size, epochs, lr,
                   weight_decay: float = 1e-4, lr_schedule: str = "ReduceLROnPlateau"):
    train_loader, val_loader, test_loader, balance_info, lang_weights = \
        get_dataloaders_multitask(data_path, batch_size=batch_size)

    for split_name, b in balance_info.items():
        if b.get('imbalanced', False):
            st.warning(
                f"**Language imbalance in {split_name}:** "
                f"Persian {b['ratio_persian']:.1f}% vs English {b['ratio_english']:.1f}%. "
                f"Class-weighted loss applied."
            )

    model     = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2).to(device)
    criterion = MultiTaskFocalLoss(
        digit_weight=0.7, lang_weight=0.3,
        lang_class_weights=lang_weights.to(device),
    )
    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, lr_schedule, int(epochs))

    progress_bar = st.progress(0); status_text = st.empty()
    st.markdown("#### Live Training Curves")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("**Loss**"); loss_ph = st.empty()
    with lc2:
        st.markdown("**Digit Accuracy (%)**"); dacc_ph = st.empty()
    with lc3:
        st.markdown("**Language Accuracy (%)**"); lacc_ph = st.empty()
    metrics_tbl = st.empty()

    best_val_loss = float('inf')
    history = {
        'Train Loss': [], 'Val Loss': [],
        'Train Digit Acc': [], 'Val Digit Acc': [],
        'Train Lang Acc':  [], 'Val Lang Acc':  [],
        'Train Digit Loss': [], 'Train Lang Loss': [],
        'Val Digit Loss':   [], 'Val Lang Loss':   [],
    }
    lr_history   = []
    time_history = []

    for epoch in range(int(epochs)):
        status_text.markdown(f"**Epoch {epoch + 1}/{epochs}…**")
        raise_if_cancelled("training")
        lr_history.append(optimizer.param_groups[0]['lr'])
        t0 = _time.time()
        t_loss, t_d_acc, t_l_acc, t_d_loss, t_l_loss = train_epoch_multitask(
            model, train_loader, criterion, optimizer, device,
            should_stop=lambda: raise_if_cancelled("training"))
        v_loss, v_d_acc, v_l_acc, v_d_loss, v_l_loss = validate_multitask(
            model, val_loader, criterion, device,
            should_stop=lambda: raise_if_cancelled("training"))
        time_history.append(_time.time() - t0)
        _scheduler_step(scheduler, v_loss, lr_schedule)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'models/best_model_multitask.pt')

        history['Train Loss'].append(t_loss);        history['Val Loss'].append(v_loss)
        history['Train Digit Acc'].append(t_d_acc);  history['Val Digit Acc'].append(v_d_acc)
        history['Train Lang Acc'].append(t_l_acc);   history['Val Lang Acc'].append(v_l_acc)
        history['Train Digit Loss'].append(t_d_loss); history['Val Digit Loss'].append(v_d_loss)
        history['Train Lang Loss'].append(t_l_loss);  history['Val Lang Loss'].append(v_l_loss)

        loss_ph.line_chart(pd.DataFrame({
            'Train Total': history['Train Loss'], 'Val Total': history['Val Loss'],
            'Train Digit': history['Train Digit Loss'], 'Val Digit': history['Val Digit Loss'],
            'Train Lang':  history['Train Lang Loss'],  'Val Lang':  history['Val Lang Loss'],
        }))
        dacc_ph.line_chart(pd.DataFrame({'Train': history['Train Digit Acc'], 'Val': history['Val Digit Acc']}))
        lacc_ph.line_chart(pd.DataFrame({'Train': history['Train Lang Acc'],  'Val': history['Val Lang Acc']}))
        metrics_tbl.markdown(
            f"| Metric | Train | Val |\n|---|---|---|\n"
            f"| **Total Loss** | {t_loss:.4f} | {v_loss:.4f} |\n"
            f"| **Digit Acc** | {t_d_acc:.2f}% | {v_d_acc:.2f}% |\n"
            f"| **Lang Acc** | {t_l_acc:.2f}% | {v_l_acc:.2f}% |  \n"
            f"| **LR** | {lr_history[-1]:.2e} | — |  \n"
            f"| **Epoch time** | {time_history[-1]:.1f}s | — |"
        )
        progress_bar.progress((epoch + 1) / int(epochs))

    status_text.success(f"Done! Saved to `models/best_model_multitask.pt`")
    st.markdown("---"); st.markdown("### Training Summary Dashboard")
    dash_fig = plot_multitask_training_history(history)
    st.pyplot(dash_fig); plt.close(dash_fig)

    st.markdown("---"); st.markdown("### Test Set Evaluation")
    with st.spinner("Evaluating…"):
        model.load_state_dict(torch.load('models/best_model_multitask.pt', map_location=device))
        t_loss, t_d_acc, t_l_acc, _, _ = validate_multitask(
            model, test_loader, criterion, device,
            should_stop=lambda: raise_if_cancelled("training"))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Digit Accuracy", f"{t_d_acc:.2f}%")
        mc2.metric("Language Accuracy", f"{t_l_acc:.2f}%")
        mc3.metric("Total Loss", f"{t_loss:.4f}")

    st.markdown("---"); st.markdown("### Confusion Matrices")
    with st.spinner("Computing…"):
        d_true, d_pred_lst, l_true, l_pred_lst = collect_predictions_multitask(
            model, test_loader, device, should_stop=lambda: raise_if_cancelled("training"))
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("#### Digit Classification")
            all_lbl = sorted(set(d_true) | set(d_pred_lst))
            cn = ["Empty" if l == 0 else str(l) for l in all_lbl]
            st.pyplot(plot_confusion_matrix(d_true, d_pred_lst, cn)); plt.close('all')
        with cm_col2:
            st.markdown("#### Language Classification")
            st.pyplot(plot_lang_confusion_matrix(l_true, l_pred_lst)); plt.close('all')
        cm_arr = confusion_matrix(d_true, d_pred_lst, labels=all_lbl)
        pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
        st.markdown("#### Per-Class Digit Accuracy")
        st.dataframe(pd.DataFrame({
            'Class': cn, 'Correct': cm_arr.diagonal(),
            'Total': cm_arr.sum(axis=1), 'Accuracy (%)': [f"{a:.1f}" for a in pca],
        }).set_index('Class'), width='stretch')

    rpt_path = save_training_report_multitask(
        history=history, test_digit_loss=t_loss,
        test_digit_acc=t_d_acc, test_lang_acc=t_l_acc,
        digit_true=d_true, digit_pred=d_pred_lst,
        lang_true=l_true,  lang_pred=l_pred_lst,
        lang_balance_info=balance_info,
        dataset_mode="MNIST+Fonts+Hoda [multi-task]",
        epochs=int(epochs), learning_rate=lr, batch_size=batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path='models/training_report_multitask.txt',
    )
    st.success(f"Report saved to `{rpt_path}`")
    with open(rpt_path, 'r', encoding='utf-8') as f:
        st.download_button("Download Multi-Task Report (.txt)",
                           f.read(), 'training_report_multitask.txt', 'text/plain')

    # ── Save all plots ────────────────────────────────────────────────
    run_dir = make_run_dir("MultiTaskCNN", int(epochs), batch_size, lr, weight_decay)
    save_fig(plot_multitask_training_history(history), run_dir, "01_multitask_dashboard.png")
    save_fig(plot_lr_history(lr_history),              run_dir, "02_lr_schedule.png")
    save_fig(plot_time_per_epoch(time_history),        run_dir, "03_time_per_epoch.png")
    save_fig(plot_confusion_matrix(d_true, d_pred_lst,
             ["Empty" if l == 0 else str(l) for l in sorted(set(d_true))]),
             run_dir, "04_confusion_digit.png")
    save_fig(plot_lang_confusion_matrix(l_true, l_pred_lst), run_dir, "05_confusion_lang.png")
    save_run_metadata(run_dir, {
        "model": "MultiTaskCNN", "epochs": int(epochs),
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "lr_schedule": lr_schedule, "best_val_loss": best_val_loss,
        "test_digit_acc": t_d_acc, "test_lang_acc": t_l_acc,
        "total_time_s": sum(time_history),
    })
    st.success(f"All plots saved to `{run_dir}/`")
    st.sidebar.success(f"Plots → `{run_dir}/`")


def _run_singletask(device, data_path, batch_size, epochs, lr, dataset_mode,
                    weight_decay: float = 1e-4, lr_schedule: str = "ReduceLROnPlateau",
                    aug_preset: str = "full"):
    if dataset_mode.startswith("MNIST + Fonts + Hoda"):
        train_loader, val_loader, test_loader = get_dataloaders_all(data_path, batch_size=batch_size)
    elif dataset_mode.startswith("MNIST + Hoda"):
        train_loader, val_loader, test_loader = get_dataloaders_mnist_hoda(data_path, batch_size=batch_size)
    elif dataset_mode.startswith("MNIST Only"):
        train_loader, val_loader, test_loader = get_dataloaders_mnist_only(batch_size=batch_size)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(data_path, batch_size=batch_size)

    from src.data_utils import build_train_transform_preset
    train_loader.dataset.transform = build_train_transform_preset(aug_preset)

    model     = DigitCNN(num_classes=10).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, lr_schedule, int(epochs))

    progress_bar = st.progress(0); status_text = st.empty(); metrics_tbl = st.empty()
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("#### Loss Curve"); loss_ph = st.empty()
    with lc2:
        st.markdown("#### Accuracy Curve"); acc_ph = st.empty()

    best_val_loss = float('inf')
    history      = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}
    lr_history   = []
    time_history = []

    for epoch in range(int(epochs)):
        status_text.markdown(f"**Running Epoch {epoch + 1}/{epochs}…**")
        raise_if_cancelled("training")
        lr_history.append(optimizer.param_groups[0]['lr'])
        t0 = _time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                     should_stop=lambda: raise_if_cancelled("training"))
        v_loss, v_acc = validate(model, val_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        time_history.append(_time.time() - t0)
        _scheduler_step(scheduler, v_loss, lr_schedule)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
        history['Train Loss'].append(t_loss); history['Val Loss'].append(v_loss)
        history['Train Acc'].append(t_acc);   history['Val Acc'].append(v_acc)
        loss_ph.line_chart(pd.DataFrame({'Train Loss': history['Train Loss'], 'Val Loss': history['Val Loss']}))
        acc_ph.line_chart(pd.DataFrame({'Train Acc': history['Train Acc'],  'Val Acc': history['Val Acc']}))
        metrics_tbl.markdown(
            f"| Metric | Train | Val |\n|---|---|---|\n"
            f"| **Loss** | {t_loss:.4f} | {v_loss:.4f} |\n"
            f"| **Accuracy** | {t_acc:.2f}% | {v_acc:.2f}% |  \n"
            f"| **LR** | {lr_history[-1]:.2e} | — |  \n"
            f"| **Epoch time** | {time_history[-1]:.1f}s | — |"
        )
        progress_bar.progress((epoch + 1) / int(epochs))

    status_text.success(f"Training Complete! Saved to `models/best_model.pt`")
    st.markdown("---"); st.markdown("### Test Set Evaluation")
    with st.spinner("Evaluating…"):
        model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
        t_loss, t_acc = validate(model, test_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        st.metric("Final Test Accuracy", f"{t_acc:.2f}%",
                  delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

    st.markdown("---"); st.markdown("### Confusion Matrix")
    with st.spinner("Computing confusion matrix…"):
        y_true, y_pred = collect_predictions(model, test_loader, device,
                                              should_stop=lambda: raise_if_cancelled("training"))
        all_lbl     = sorted(set(y_true) | set(y_pred))
        class_names = ["Empty" if l == 0 else str(l) for l in all_lbl]
        cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.pyplot(cm_fig); plt.close('all')
        cm_arr = confusion_matrix(y_true, y_pred, labels=all_lbl)
        pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
        st.markdown("#### Per-Class Accuracy")
        st.dataframe(pd.DataFrame({
            'Class': class_names, 'Correct': cm_arr.diagonal(),
            'Total': cm_arr.sum(axis=1), 'Accuracy (%)': [f"{a:.1f}" for a in pca],
        }).set_index('Class'), width='stretch')

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred, dataset_mode=dataset_mode,
        epochs=int(epochs), learning_rate=lr, batch_size=batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path='models/training_report.txt',
    )
    st.success(f"Training report saved to `{rpt}`")
    with open(rpt, 'r', encoding='utf-8') as f:
        st.download_button("Download Training Report (.txt)",
                           f.read(), 'training_report.txt', 'text/plain')

    # ── Save all plots ────────────────────────────────────────────────
    run_dir = make_run_dir("DigitCNN", int(epochs), batch_size, lr, weight_decay)
    save_fig(plot_singletask_summary(history, lr_history, time_history), run_dir, "01_summary.png")
    save_fig(plot_lr_history(lr_history),       run_dir, "02_lr_schedule.png")
    save_fig(plot_time_per_epoch(time_history),  run_dir, "03_time_per_epoch.png")
    save_fig(cm_fig, run_dir, "04_confusion_matrix.png")
    save_run_metadata(run_dir, {
        "model": "DigitCNN", "dataset_mode": dataset_mode,
        "epochs": int(epochs), "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay, "lr_schedule": lr_schedule,
        "best_val_loss": best_val_loss, "test_acc": t_acc,
        "total_time_s": sum(time_history),
    })
    st.success(f"All plots saved to `{run_dir}/`")
    st.sidebar.success(f"Plots → `{run_dir}/`")


# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet two-phase training path
# ──────────────────────────────────────────────────────────────────────────────

def _run_efficientnet(
    device, data_path, batch_size, epochs, lr,
    purpose, dataset_choice, aug_preset,
    pretrained, phase1_epochs, phase2_epochs,
    weight_decay: float = 1e-4,
):
    from src.data_utils import get_dataloaders_efficientnet
    from src.train import run_twophase_training

    dataset_mode_map = {
        "MNIST Only":                 "mnist_only",
        "MNIST + Fonts":              "mnist_fonts",
        "MNIST + Fonts + Hoda (All)": "all",
        "MNIST + Hoda":               "mnist_hoda",
        "Persian Only (Hoda)":        "persian",
    }
    dataset_mode = dataset_mode_map.get(dataset_choice, "all")

    train_loader, val_loader, test_loader = get_dataloaders_efficientnet(
        data_path, dataset_mode=dataset_mode, batch_size=batch_size, aug_preset=aug_preset,
    )

    model = EfficientNetDigitCNN(num_classes=10, pretrained=pretrained).to(device)
    total_p, _ = model.param_count()
    st.info(
        f"EfficientNet-B0 | Params: **{total_p:,}** | Pretrained: {pretrained} | "
        f"Dataset: {dataset_choice} | Aug: {aug_preset}"
    )

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    progress_bar = st.progress(0)
    status_text  = st.empty()
    lc1, lc2    = st.columns(2)
    with lc1:
        st.markdown("#### Loss"); loss_ph = st.empty()
    with lc2:
        st.markdown("#### Accuracy"); acc_ph = st.empty()
    metrics_tbl = st.empty()

    total_epochs  = phase1_epochs + phase2_epochs
    epoch_counter = [0]

    def on_epoch_end(epoch_idx, phase, t_loss, t_acc, v_loss, v_acc):
        epoch_counter[0] += 1
        status_text.markdown(f"**Epoch {epoch_counter[0]}/{total_epochs} — Phase: {phase}**")
        progress_bar.progress(epoch_counter[0] / total_epochs)
        metrics_tbl.markdown(
            f"| Metric | Train | Val |\n|---|---|---|\n"
            f"| **Loss** | {t_loss:.4f} | {v_loss:.4f} |\n"
            f"| **Accuracy** | {t_acc:.2f}% | {v_acc:.2f}% |\n"
            f"| **Phase** | {phase} | — |"
        )

    save_path = f'models/best_model_efficientnet_{purpose.lower()}.pt'

    history, best_val_loss = run_twophase_training(
        model, train_loader, val_loader, criterion, device,
        epochs_phase1=phase1_epochs,
        epochs_phase2=phase2_epochs,
        lr_phase1=lr,
        lr_phase2=lr * 0.01,
        weight_decay=weight_decay,
        unfreeze_blocks=3,
        should_stop=lambda: raise_if_cancelled("training"),
        on_epoch_end=on_epoch_end,
    )

    loss_ph.line_chart(pd.DataFrame({'Train': history['Train Loss'], 'Val': history['Val Loss']}))
    acc_ph.line_chart(pd.DataFrame({'Train': history['Train Acc'],  'Val': history['Val Acc']}))

    torch.save(model.state_dict(), save_path)
    status_text.success(f"Done! Saved to `{save_path}` (best val loss: {best_val_loss:.4f})")

    st.markdown("---"); st.markdown("### Test Set Evaluation")
    with st.spinner("Evaluating…"):
        t_loss, t_acc = validate(model, test_loader, criterion, device,
                                  should_stop=lambda: raise_if_cancelled("training"))
        st.metric("Test Accuracy", f"{t_acc:.2f}%", delta=f"Loss: {t_loss:.4f}", delta_color="inverse")

    st.markdown("---"); st.markdown("### Confusion Matrix")
    with st.spinner("Computing…"):
        y_true, y_pred = collect_predictions(model, test_loader, device,
                                              should_stop=lambda: raise_if_cancelled("training"))
        all_lbl     = sorted(set(y_true) | set(y_pred))
        class_names = ["Empty" if l == 0 else str(l) for l in all_lbl]
        cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
        st.pyplot(cm_fig); plt.close('all')
        cm_arr = confusion_matrix(y_true, y_pred, labels=all_lbl)
        pca    = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
        st.dataframe(pd.DataFrame({
            'Class': class_names, 'Correct': cm_arr.diagonal(),
            'Total': cm_arr.sum(axis=1), 'Accuracy (%)': [f"{a:.1f}" for a in pca],
        }).set_index('Class'), use_container_width=True)

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred,
        dataset_mode=f"EfficientNet-B0 [{dataset_choice}] aug={aug_preset}",
        epochs=total_epochs, learning_rate=lr, batch_size=batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path=f'models/training_report_efficientnet_{purpose.lower()}.txt',
    )
    st.success(f"Report saved to `{rpt}`")
    with open(rpt, 'r', encoding='utf-8') as f:
        st.download_button("Download EfficientNet Report (.txt)",
                           f.read(), f'training_report_efficientnet_{purpose.lower()}.txt', 'text/plain')

    run_dir = make_run_dir(f"EfficientNet_{purpose}", total_epochs, batch_size, lr, weight_decay)
    save_fig(plot_singletask_summary(history, [], []), run_dir, "01_summary.png")
    save_fig(cm_fig, run_dir, "04_confusion_matrix.png")
    save_run_metadata(run_dir, {
        "model": "EfficientNetDigit", "purpose": purpose, "dataset": dataset_choice,
        "aug_preset": aug_preset, "pretrained": pretrained,
        "epochs_phase1": phase1_epochs, "epochs_phase2": phase2_epochs,
        "best_val_loss": best_val_loss, "test_acc": t_acc,
    })
    st.success(f"All plots saved to `{run_dir}/`")
    st.sidebar.success(f"Plots → `{run_dir}/`")
