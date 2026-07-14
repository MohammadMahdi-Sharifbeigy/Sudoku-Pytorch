import sys
import time

import torch
from tqdm.auto import tqdm


def _check_stop(should_stop):
    if should_stop is not None:
        should_stop()


def _is_tty() -> bool:
    """True when stdout is an interactive terminal (CLI), False in Streamlit/notebook."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def train_epoch(model, train_loader, criterion, optimizer, device, should_stop=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loader = tqdm(train_loader, desc="  train", leave=False,
                  disable=not _is_tty(), ncols=80, unit="batch")

    for data, target in loader:
        _check_stop(should_stop)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total   += target.size(0)
        correct += (predicted == target).sum().item()

        loader.set_postfix(loss=f"{loss.item():.4f}",
                           acc=f"{100*correct/max(total,1):.1f}%",
                           refresh=False)
        _check_stop(should_stop)

    return total_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device, should_stop=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            _check_stop(should_stop)
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            _check_stop(should_stop)
            
    return total_loss / len(val_loader), 100 * correct / total

def collect_predictions(model, loader, device, should_stop=None):
    """Run model over loader and return all true labels and predicted labels."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            _check_stop(should_stop)
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(target.numpy().tolist())
            _check_stop(should_stop)
    return all_targets, all_preds


# ──────────────────────────────────────────────
# Multi-task training functions
# ──────────────────────────────────────────────

def train_epoch_multitask(model, train_loader, criterion, optimizer, device, should_stop=None):
    """One epoch for MultiTaskDigitCNN.
    train_loader yields (data, digit_target, lang_target).
    criterion must be MultiTaskFocalLoss.
    Returns (total_loss, digit_acc_pct, lang_acc_pct, digit_loss, lang_loss).
    """
    model.train()
    total_loss = d_loss_sum = l_loss_sum = 0.0
    d_correct = d_total = l_correct = l_total = 0

    for data, d_target, l_target in train_loader:
        _check_stop(should_stop)
        data, d_target, l_target = data.to(device), d_target.to(device), l_target.to(device)
        optimizer.zero_grad()
        digit_logits, lang_logits = model(data)
        loss, d_loss, l_loss = criterion(digit_logits, lang_logits, d_target, l_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        d_loss_sum += d_loss
        l_loss_sum += l_loss

        _, d_pred = torch.max(digit_logits.data, 1)
        d_total   += d_target.size(0)
        d_correct += (d_pred == d_target).sum().item()

        valid = l_target != -1
        if valid.any():
            _, l_pred = torch.max(lang_logits.data, 1)
            l_correct += (l_pred[valid] == l_target[valid]).sum().item()
            l_total   += valid.sum().item()
        _check_stop(should_stop)

    n = len(train_loader)
    return (total_loss / n,
            100.0 * d_correct / max(d_total, 1),
            100.0 * l_correct / max(l_total, 1),
            d_loss_sum / n,
            l_loss_sum / n)


def validate_multitask(model, val_loader, criterion, device, should_stop=None):
    """Validation for MultiTaskDigitCNN.
    Returns (total_loss, digit_acc_pct, lang_acc_pct, digit_loss, lang_loss).
    """
    model.eval()
    total_loss = d_loss_sum = l_loss_sum = 0.0
    d_correct = d_total = l_correct = l_total = 0

    with torch.no_grad():
        for data, d_target, l_target in val_loader:
            _check_stop(should_stop)
            data, d_target, l_target = data.to(device), d_target.to(device), l_target.to(device)
            digit_logits, lang_logits = model(data)
            loss, d_loss, l_loss = criterion(digit_logits, lang_logits, d_target, l_target)

            total_loss += loss.item()
            d_loss_sum += d_loss
            l_loss_sum += l_loss

            _, d_pred = torch.max(digit_logits.data, 1)
            d_total   += d_target.size(0)
            d_correct += (d_pred == d_target).sum().item()

            valid = l_target != -1
            if valid.any():
                _, l_pred = torch.max(lang_logits.data, 1)
                l_correct += (l_pred[valid] == l_target[valid]).sum().item()
                l_total   += valid.sum().item()
            _check_stop(should_stop)

    n = len(val_loader)
    return (total_loss / n,
            100.0 * d_correct / max(d_total, 1),
            100.0 * l_correct / max(l_total, 1),
            d_loss_sum / n,
            l_loss_sum / n)


def collect_predictions_multitask(model, loader, device, should_stop=None):
    """Run MultiTaskDigitCNN over loader.
    Returns (digit_true, digit_pred, lang_true, lang_pred).
    lang_* contain only non-empty-cell samples (lang_target != -1).
    """
    model.eval()
    d_preds, d_targets = [], []
    l_preds, l_targets = [], []

    with torch.no_grad():
        for data, d_target, l_target in loader:
            _check_stop(should_stop)
            data = data.to(device)
            digit_logits, lang_logits = model(data)

            d_pred = torch.argmax(digit_logits, dim=1).cpu()
            l_pred = torch.argmax(lang_logits, dim=1).cpu()

            d_preds.extend(d_pred.numpy().tolist())
            d_targets.extend(d_target.numpy().tolist())

            valid = l_target != -1
            l_preds.extend(l_pred[valid].numpy().tolist())
            l_targets.extend(l_target[valid].numpy().tolist())
            _check_stop(should_stop)

    return d_targets, d_preds, l_targets, l_preds


def run_twophase_training(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs_phase1: int = 6,
    epochs_phase2: int = 14,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-5,
    weight_decay: float = 1e-4,
    unfreeze_blocks: int = 3,
    should_stop=None,
    on_epoch_end=None,
):
    """Two-phase training for EfficientNetDigitCNN.

    Phase 1: freeze backbone, train head only (lr_phase1, epochs_phase1 epochs).
    Phase 2: unfreeze last unfreeze_blocks feature blocks (lr_phase2, epochs_phase2 epochs).
    BN remains frozen in both phases (correct for ImageNet-pretrained transfer).

    on_epoch_end(epoch_idx, phase, train_loss, train_acc, val_loss, val_acc) called each epoch.
    Returns: (history dict, best_val_loss, phase_boundaries).
    phase_boundaries is a list of 1-based epoch indices marking phase
    transitions (used to draw dashed lines on the saved LR-schedule plot).
    """
    import copy
    import torch.optim as optim

    history = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': [],
               'Phase': [], 'LR': [], 'Epoch Time': []}
    best_val_loss = float('inf')
    best_state = None
    phase_boundaries = []  # 1-based epoch indices where a phase transition occurs

    def _run_phase(phase_label, n_epochs, lr):
        nonlocal best_val_loss, best_state
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs))
        for _ in range(n_epochs):
            if should_stop:
                should_stop()
            # Re-freeze BN each epoch — model.train() would re-enable running stats otherwise
            model.train()
            model._freeze_batchnorm()

            t0 = time.time()
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                        should_stop=should_stop)
            v_loss, v_acc = validate(model, val_loader, criterion, device,
                                     should_stop=should_stop)
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]['lr']
            scheduler.step()

            history['Train Loss'].append(t_loss)
            history['Val Loss'].append(v_loss)
            history['Train Acc'].append(t_acc)
            history['Val Acc'].append(v_acc)
            history['Phase'].append(phase_label)
            history['LR'].append(lr_now)
            history['Epoch Time'].append(elapsed)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = copy.deepcopy(model.state_dict())

            if on_epoch_end:
                on_epoch_end(
                    len(history['Train Loss']) - 1,
                    phase_label, t_loss, t_acc, v_loss, v_acc,
                )

    model.freeze_backbone()
    _run_phase("head-only", epochs_phase1, lr_phase1)
    phase_boundaries.append(len(history['Train Loss']))

    model.unfreeze_last_blocks(n=unfreeze_blocks)
    _run_phase("fine-tune", epochs_phase2, lr_phase2)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_loss, phase_boundaries
