import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
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

    return total_loss / len(train_loader), 100 * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return total_loss / len(val_loader), 100 * correct / total


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run model over loader and return all true labels and predicted labels."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(target.numpy().tolist())
    return all_targets, all_preds
