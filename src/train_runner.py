"""Pure-Python training runner — no Streamlit dependency.

Can be called from the Streamlit UI (via thread) or from the CLI directly.

Usage (CLI):
    python run_training_cli.py --model DigitCNN --purpose English ...

Usage (Streamlit UI):
    import threading, queue
    from src.train_runner import TrainingConfig, run_training

    progress_q = queue.Queue()
    stop_ev    = threading.Event()
    config     = TrainingConfig(...)

    def on_epoch(update: dict):
        progress_q.put(update)

    t = threading.Thread(
        target=run_training,
        args=(config, device, on_epoch, stop_ev),
        daemon=True,
    )
    t.start()
"""

from __future__ import annotations

import copy
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ── Sentinel exception ────────────────────────────────────────────────────────

class RunCancelled(Exception):
    pass


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_choice:       str   = "DigitCNN"
    purpose:            str   = "English"
    dataset_choice:     str   = "MNIST + Fonts"
    aug_preset:         str   = "full"
    epochs:             int   = 20
    learning_rate:      float = 1e-3
    batch_size:         int   = 128
    weight_decay:       float = 1e-4
    lr_schedule:        str   = "ReduceLROnPlateau"
    data_path:          str   = "data"
    # EfficientNet
    eff_pretrained:     bool  = True
    eff_phase1_epochs:  int   = 6
    eff_phase2_epochs:  int   = 14
    # UnifiedCNN
    unified_backbone:   str   = "mobilenet_v3_small"
    unified_pretrained: bool  = False
    # Multi-script
    multi_mode:         str   = "Unified model (20-class)"
    # Cosine scheduler params
    cosine_t0:          int   = 10    # CosineAnnealingWarmRestarts first cycle length
    cosine_t_mult:      int   = 1     # cycle length multiplier after each restart
    cosine_eta_min:     float = 1e-6  # minimum LR floor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_stop(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RunCancelled("Training stopped by user.")


def _make_optimizer(model, lr: float, wd: float) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _make_scheduler(optimizer, schedule: str, epochs: int, train_loader=None,
                    cosine_t0: int = 10, cosine_t_mult: int = 1, cosine_eta_min: float = 1e-6):
    if schedule == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, cosine_t0), T_mult=max(1, cosine_t_mult),
            eta_min=cosine_eta_min,
        )
    if schedule == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=cosine_eta_min,
        )
    if schedule == "OneCycleLR":
        steps = len(train_loader) if train_loader else 100
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]['lr'] * 10,
            steps_per_epoch=steps, epochs=epochs, pct_start=0.3,
        )
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


def _scheduler_step(scheduler, val_loss: float, schedule: str) -> None:
    if schedule == "OneCycleLR":
        return
    if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
        scheduler.step()
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _map_dataset_to_loader_key(dataset_choice: str) -> str:
    mapping = {
        "MNIST Only":                 "mnist_only",
        "MNIST + Fonts":              "mnist_fonts",
        "MNIST + Fonts + Hoda (All)": "all",
        "MNIST + Hoda":               "mnist_hoda",
        "Persian Only (Hoda)":        "persian",
        # legacy strings from old radio
        "MNIST + Fonts (Recommended for printed Sudoku)": "mnist_fonts",
        "MNIST + Fonts + Hoda (All)":                     "all",
        "Persian Only (Hoda — saves best_model_persian.pt)": "persian",
        "English Only (MNIST + Fonts — saves best_model_english.pt)": "mnist_fonts",
    }
    return mapping.get(dataset_choice, "all")


# ── Epoch loop helpers ────────────────────────────────────────────────────────

def _train_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs, device, on_epoch, stop_event, lr_schedule, save_path, phase="train",
):
    """Generic single-task epoch loop. Returns (history, best_val_loss)."""
    best_val_loss = float('inf')
    best_state    = None
    history = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}

    from src.train import train_epoch, validate

    for epoch in range(epochs):
        _check_stop(stop_event)
        lr_now = optimizer.param_groups[0]['lr']
        t0 = time.time()

        t_loss, t_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            should_stop=lambda: _check_stop(stop_event),
        )
        v_loss, v_acc = validate(
            model, val_loader, criterion, device,
            should_stop=lambda: _check_stop(stop_event),
        )
        elapsed = time.time() - t0

        _scheduler_step(scheduler, v_loss, lr_schedule)

        history['Train Loss'].append(t_loss)
        history['Val Loss'].append(v_loss)
        history['Train Acc'].append(t_acc)
        history['Val Acc'].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = copy.deepcopy(model.state_dict())

        if on_epoch:
            on_epoch({
                'type':        'epoch',
                'epoch':       epoch,
                'total':       epochs,
                'phase':       phase,
                'train_loss':  t_loss,
                'train_acc':   t_acc,
                'val_loss':    v_loss,
                'val_acc':     v_acc,
                'lr':          lr_now,
                'elapsed_s':   elapsed,
            })

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)

    return history, best_val_loss


# ── Trainers ──────────────────────────────────────────────────────────────────

def _run_singletask(cfg: TrainingConfig, device, on_epoch, stop_event) -> dict:
    from src.model import DigitCNN, FocalLoss
    from src.data_utils import (
        get_dataloaders, get_dataloaders_all,
        get_dataloaders_mnist_hoda, get_dataloaders_mnist_only,
        build_train_transform_preset,
    )
    from src.train import validate, collect_predictions
    from src.report_utils import save_training_report

    key = _map_dataset_to_loader_key(cfg.dataset_choice)
    if key == "all":
        train_loader, val_loader, test_loader = get_dataloaders_all(cfg.data_path, batch_size=cfg.batch_size)
    elif key == "mnist_hoda":
        train_loader, val_loader, test_loader = get_dataloaders_mnist_hoda(cfg.data_path, batch_size=cfg.batch_size)
    elif key == "mnist_only":
        train_loader, val_loader, test_loader = get_dataloaders_mnist_only(batch_size=cfg.batch_size)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(cfg.data_path, batch_size=cfg.batch_size)

    train_loader.dataset.transform = build_train_transform_preset(cfg.aug_preset)

    label = "Persian" if cfg.purpose == "Persian" else "English"
    save_path = f"models/best_model_{label.lower()}.pt" if cfg.purpose in ("Persian", "English") else "models/best_model.pt"
    report_path = f"models/training_report_{label.lower()}.txt" if cfg.purpose in ("Persian", "English") else "models/training_report.txt"

    model     = DigitCNN(num_classes=10).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = _make_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.lr_schedule, cfg.epochs, train_loader,
                                cfg.cosine_t0, cfg.cosine_t_mult, cfg.cosine_eta_min)

    history, best_val_loss = _train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg.epochs, device, on_epoch, stop_event, cfg.lr_schedule, save_path,
    )

    model.load_state_dict(torch.load(save_path, map_location=device))
    t_loss, t_acc = validate(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred,
        dataset_mode=cfg.dataset_choice, epochs=cfg.epochs,
        learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path=report_path,
    )

    return {
        'type': 'done', 'model_type': 'singletask', 'label': label,
        'save_path': save_path, 'report_path': rpt,
        'test_loss': t_loss, 'test_acc': t_acc,
        'y_true': y_true, 'y_pred': y_pred,
        'history': history, 'best_val_loss': best_val_loss,
        'model': model, 'test_loader': test_loader, 'criterion': criterion,
    }


def _run_lang_specific(cfg: TrainingConfig, device, on_epoch, stop_event, is_persian: bool) -> dict:
    from src.model import DigitCNN, FocalLoss
    from src.data_utils import (
        get_dataloaders_persian, get_dataloaders_english,
        build_train_transform_preset,
    )
    from src.train import validate, collect_predictions
    from src.report_utils import save_training_report

    if is_persian:
        train_loader, val_loader, test_loader = get_dataloaders_persian(cfg.data_path, batch_size=cfg.batch_size)
        if train_loader is None:
            raise FileNotFoundError("Hoda dataset not found. Check data/DigitDB/ exists.")
        save_path   = "models/best_model_persian.pt"
        report_path = "models/training_report_persian.txt"
        lang_label  = "Persian"
    else:
        train_loader, val_loader, test_loader = get_dataloaders_english(cfg.data_path, batch_size=cfg.batch_size)
        save_path   = "models/best_model_english.pt"
        report_path = "models/training_report_english.txt"
        lang_label  = "English"

    train_loader.dataset.transform = build_train_transform_preset(cfg.aug_preset)

    model     = DigitCNN(num_classes=10).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = _make_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.lr_schedule, cfg.epochs, train_loader,
                                cfg.cosine_t0, cfg.cosine_t_mult, cfg.cosine_eta_min)

    history, best_val_loss = _train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg.epochs, device, on_epoch, stop_event, cfg.lr_schedule, save_path,
        phase=lang_label,
    )

    model.load_state_dict(torch.load(save_path, map_location=device))
    t_loss, t_acc = validate(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred,
        dataset_mode=lang_label, epochs=cfg.epochs,
        learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path=report_path,
    )

    return {
        'type': 'done', 'model_type': 'lang_specific', 'label': lang_label,
        'save_path': save_path, 'report_path': rpt,
        'test_loss': t_loss, 'test_acc': t_acc,
        'y_true': y_true, 'y_pred': y_pred,
        'history': history, 'best_val_loss': best_val_loss,
        'model': model, 'test_loader': test_loader, 'criterion': criterion,
    }


def _run_multitask(cfg: TrainingConfig, device, on_epoch, stop_event) -> dict:
    from src.model import MultiTaskDigitCNN, MultiTaskFocalLoss
    from src.data_utils import get_dataloaders_multitask
    from src.train import (
        train_epoch_multitask, validate_multitask, collect_predictions_multitask,
    )
    from src.report_utils import save_training_report_multitask

    train_loader, val_loader, test_loader, balance_info, lang_weights = \
        get_dataloaders_multitask(cfg.data_path, batch_size=cfg.batch_size)

    model     = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2).to(device)
    criterion = MultiTaskFocalLoss(
        digit_weight=0.7, lang_weight=0.3,
        lang_class_weights=lang_weights.to(device),
    )
    optimizer = _make_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.lr_schedule, cfg.epochs, train_loader,
                                cfg.cosine_t0, cfg.cosine_t_mult, cfg.cosine_eta_min)
    save_path = "models/best_model_multitask.pt"

    best_val_loss = float('inf')
    best_state    = None
    history = {
        'Train Loss': [], 'Val Loss': [],
        'Train Digit Acc': [], 'Val Digit Acc': [],
        'Train Lang Acc':  [], 'Val Lang Acc':  [],
        'Train Digit Loss': [], 'Train Lang Loss': [],
        'Val Digit Loss':   [], 'Val Lang Loss':   [],
    }

    for epoch in range(cfg.epochs):
        _check_stop(stop_event)
        lr_now = optimizer.param_groups[0]['lr']
        t0 = time.time()

        t_loss, t_d_acc, t_l_acc, t_d_loss, t_l_loss = train_epoch_multitask(
            model, train_loader, criterion, optimizer, device,
            should_stop=lambda: _check_stop(stop_event),
        )
        v_loss, v_d_acc, v_l_acc, v_d_loss, v_l_loss = validate_multitask(
            model, val_loader, criterion, device,
            should_stop=lambda: _check_stop(stop_event),
        )
        elapsed = time.time() - t0
        _scheduler_step(scheduler, v_loss, cfg.lr_schedule)

        for k, v in [
            ('Train Loss', t_loss), ('Val Loss', v_loss),
            ('Train Digit Acc', t_d_acc), ('Val Digit Acc', v_d_acc),
            ('Train Lang Acc', t_l_acc), ('Val Lang Acc', v_l_acc),
            ('Train Digit Loss', t_d_loss), ('Val Digit Loss', v_d_loss),
            ('Train Lang Loss', t_l_loss), ('Val Lang Loss', v_l_loss),
        ]:
            history[k].append(v)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = copy.deepcopy(model.state_dict())

        if on_epoch:
            on_epoch({
                'type':          'epoch',
                'epoch':         epoch,
                'total':         cfg.epochs,
                'phase':         'multitask',
                'train_loss':    t_loss,
                'train_acc':     t_d_acc,
                'val_loss':      v_loss,
                'val_acc':       v_d_acc,
                'train_lang_acc': t_l_acc,
                'val_lang_acc':  v_l_acc,
                'lr':            lr_now,
                'elapsed_s':     elapsed,
            })

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    t_loss, t_d_acc, t_l_acc, _, _ = validate_multitask(model, test_loader, criterion, device)
    d_true, d_pred, l_true, l_pred = collect_predictions_multitask(model, test_loader, device)

    rpt = save_training_report_multitask(
        history=history, test_digit_loss=t_loss,
        test_digit_acc=t_d_acc, test_lang_acc=t_l_acc,
        digit_true=d_true, digit_pred=d_pred,
        lang_true=l_true, lang_pred=l_pred,
        lang_balance_info=balance_info,
        dataset_mode="MNIST+Fonts+Hoda [multi-task]",
        epochs=cfg.epochs, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path="models/training_report_multitask.txt",
    )

    return {
        'type': 'done', 'model_type': 'multitask', 'label': 'MultiTask',
        'save_path': save_path, 'report_path': rpt,
        'test_loss': t_loss, 'test_digit_acc': t_d_acc, 'test_lang_acc': t_l_acc,
        'd_true': d_true, 'd_pred': d_pred, 'l_true': l_true, 'l_pred': l_pred,
        'history': history, 'best_val_loss': best_val_loss,
        'model': model, 'test_loader': test_loader, 'criterion': criterion,
        'balance_info': balance_info,
    }


def _run_unified(cfg: TrainingConfig, device, on_epoch, stop_event) -> dict:
    from src.model import UnifiedDigitCNN, UNIFIED_NUM_CLASSES, decode_unified_class
    from src.data_utils import get_dataloaders_unified20
    from src.train import validate, collect_predictions
    from src.report_utils import save_training_report

    train_loader, val_loader, test_loader = get_dataloaders_unified20(
        cfg.data_path, batch_size=cfg.batch_size)

    model     = UnifiedDigitCNN(
        backbone=cfg.unified_backbone,
        pretrained=cfg.unified_pretrained,
        num_classes=UNIFIED_NUM_CLASSES,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.lr_schedule, cfg.epochs, train_loader,
                                cfg.cosine_t0, cfg.cosine_t_mult, cfg.cosine_eta_min)
    save_path = "models/best_model_unified20.pt"

    history, best_val_loss = _train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        cfg.epochs, device, on_epoch, stop_event, cfg.lr_schedule, save_path,
        phase="unified",
    )

    model.load_state_dict(torch.load(save_path, map_location=device))
    t_loss, t_acc = validate(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred,
        dataset_mode=f"Unified20 [{cfg.unified_backbone}]",
        epochs=cfg.epochs, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path="models/training_report_unified20.txt",
    )

    return {
        'type': 'done', 'model_type': 'unified', 'label': 'Unified20',
        'save_path': save_path, 'report_path': rpt,
        'test_loss': t_loss, 'test_acc': t_acc,
        'y_true': y_true, 'y_pred': y_pred,
        'history': history, 'best_val_loss': best_val_loss,
        'model': model, 'test_loader': test_loader, 'criterion': criterion,
        'decode_fn': decode_unified_class,
    }


def _run_efficientnet(cfg: TrainingConfig, device, on_epoch, stop_event) -> dict:
    from src.model import EfficientNetDigitCNN, FocalLoss
    from src.data_utils import get_dataloaders_efficientnet
    from src.train import run_twophase_training, validate, collect_predictions
    from src.report_utils import save_training_report

    dataset_mode_map = {
        "MNIST Only": "mnist_only", "MNIST + Fonts": "mnist_fonts",
        "MNIST + Fonts + Hoda (All)": "all", "MNIST + Hoda": "mnist_hoda",
        "Persian Only (Hoda)": "persian",
    }
    dataset_mode = dataset_mode_map.get(cfg.dataset_choice, "all")

    train_loader, val_loader, test_loader = get_dataloaders_efficientnet(
        cfg.data_path, dataset_mode=dataset_mode,
        batch_size=cfg.batch_size, aug_preset=cfg.aug_preset,
    )

    model     = EfficientNetDigitCNN(num_classes=10, pretrained=cfg.eff_pretrained).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    save_path = f"models/best_model_efficientnet_{cfg.purpose.lower()}.pt"

    total_epochs  = cfg.eff_phase1_epochs + cfg.eff_phase2_epochs
    epoch_counter = [0]

    def _on_epoch_eff(epoch_idx, phase, t_loss, t_acc, v_loss, v_acc):
        epoch_counter[0] += 1
        if on_epoch:
            on_epoch({
                'type':       'epoch',
                'epoch':      epoch_counter[0] - 1,
                'total':      total_epochs,
                'phase':      phase,
                'train_loss': t_loss,
                'train_acc':  t_acc,
                'val_loss':   v_loss,
                'val_acc':    v_acc,
                'lr':         0.0,
                'elapsed_s':  0.0,
            })

    history, best_val_loss = run_twophase_training(
        model, train_loader, val_loader, criterion, device,
        epochs_phase1=cfg.eff_phase1_epochs,
        epochs_phase2=cfg.eff_phase2_epochs,
        lr_phase1=cfg.learning_rate,
        lr_phase2=cfg.learning_rate * 0.01,
        weight_decay=cfg.weight_decay,
        unfreeze_blocks=3,
        should_stop=lambda: _check_stop(stop_event),
        on_epoch_end=_on_epoch_eff,
    )

    torch.save(model.state_dict(), save_path)
    t_loss, t_acc = validate(model, test_loader, criterion, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    rpt = save_training_report(
        history=history, test_loss=t_loss, test_acc=t_acc,
        y_true=y_true, y_pred=y_pred,
        dataset_mode=f"EfficientNet-B0 [{cfg.dataset_choice}] aug={cfg.aug_preset}",
        epochs=total_epochs, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        best_val_loss=best_val_loss, model=model,
        output_path=f"models/training_report_efficientnet_{cfg.purpose.lower()}.txt",
    )

    return {
        'type': 'done', 'model_type': 'efficientnet', 'label': f'EfficientNet-{cfg.purpose}',
        'save_path': save_path, 'report_path': rpt,
        'test_loss': t_loss, 'test_acc': t_acc,
        'y_true': y_true, 'y_pred': y_pred,
        'history': history, 'best_val_loss': best_val_loss,
        'model': model, 'test_loader': test_loader, 'criterion': criterion,
    }


# ── Public router ─────────────────────────────────────────────────────────────

def run_training(
    config: TrainingConfig,
    device: torch.device,
    on_epoch: Optional[Callable[[dict], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Run training. Raises RunCancelled if stop_event fires. Returns result dict.

    on_epoch(update_dict) is called after each epoch with keys:
        type, epoch, total, phase, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s
    """
    os.makedirs("models", exist_ok=True)

    m = config.model_choice
    p = config.purpose

    if m == "EfficientNetDigit":
        return _run_efficientnet(config, device, on_epoch, stop_event)

    if m == "MultiTaskCNN":
        return _run_multitask(config, device, on_epoch, stop_event)

    if m in ("UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2"):
        return _run_unified(config, device, on_epoch, stop_event)

    if p == "Persian":
        return _run_lang_specific(config, device, on_epoch, stop_event, is_persian=True)

    if p == "English":
        return _run_singletask(config, device, on_epoch, stop_event)

    # Multi + DigitCNN
    if config.multi_mode and "Separate" in config.multi_mode:
        r_per = _run_lang_specific(config, device, on_epoch, stop_event, is_persian=True)
        r_eng = _run_lang_specific(config, device, on_epoch, stop_event, is_persian=False)
        return {'type': 'done', 'model_type': 'separate', 'persian': r_per, 'english': r_eng}

    return _run_singletask(config, device, on_epoch, stop_event)
