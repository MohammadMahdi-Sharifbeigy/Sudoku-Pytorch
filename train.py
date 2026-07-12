import torch


def _check_stop(should_stop):
    if should_stop is not None:
        should_stop()


def train_epoch(model, train_loader, criterion, optimizer, device, should_stop=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        _check_stop(should_stop)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
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
