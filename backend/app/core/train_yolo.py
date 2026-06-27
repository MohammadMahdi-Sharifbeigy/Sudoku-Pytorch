"""Ultralytics YOLOv8-detect fine-tuning — shared by the CLI script and SSE endpoint."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Callable, Optional
import torch


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo(
    data_yaml: str, seed_weights: str, epochs: int, imgsz: int, batch: int,
    device: str, models_dir: str, on_epoch: Optional[Callable[[dict], None]] = None,
) -> str:
    from ultralytics import YOLO
    resolved_device = pick_device(device)
    model = YOLO(str(seed_weights))   # task=detect inferred from the seed

    if on_epoch is not None:
        def _cb(trainer):
            epoch_num = int(getattr(trainer, "epoch", 0)) + 1
            if epoch_num > epochs:
                # ultralytics fires this callback once more for the post-train
                # final validation pass — don't report a phantom epoch.
                return
            metrics = getattr(trainer, "metrics", {}) or {}
            losses = {}
            li = getattr(trainer, "label_loss_items", None)
            if li and getattr(trainer, "loss_items", None) is not None:
                try:
                    losses = li(trainer.loss_items)
                except Exception:
                    losses = {}
            on_epoch({
                "epoch": epoch_num,
                "epochs": int(epochs),
                "box_loss": float(losses.get("train/box_loss", 0.0)),
                "cls_loss": float(losses.get("train/cls_loss", 0.0)),
                "dfl_loss": float(losses.get("train/dfl_loss", 0.0)),
                "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
            })
        model.add_callback("on_fit_epoch_end", _cb)

    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        device=resolved_device, project=str(Path(models_dir) / "yolo" / "runs"),
        name="train", exist_ok=True, verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out_dir = Path(models_dir) / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"sudoku_detect_ep{epochs}_imgsz{imgsz}.pt"
    shutil.copyfile(best, dest)
    from app.core.registry.yolo_models import write_yolo_pointer
    write_yolo_pointer(models_dir, str(dest))
    return str(dest)
