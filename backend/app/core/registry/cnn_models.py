"""List + cached-load digit-CNN checkpoints (models/*.pt, excluding YOLO seeds)."""
from __future__ import annotations
import datetime
from pathlib import Path
import torch
from app.core.model import DigitCNN


def _is_yolo_seed(name: str) -> bool:
    return name.lower().startswith("yolov8")


def _entry(path: Path, base: Path, default_id: str | None) -> dict:
    stat = path.stat()
    rel = path.relative_to(base).as_posix()
    return {
        "id": rel, "filename": path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "created_at": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
        "is_default": rel == default_id,
    }


def list_cnn_models(models_dir: str, default_id: str | None = None) -> list[dict]:
    base = Path(models_dir)
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if _is_yolo_seed(p.name):
            continue
        out.append(_entry(p, base, default_id))
    return out


def resolve_cnn_path(models_dir: str, model_id: str) -> Path:
    base = Path(models_dir).resolve()
    candidate = (base / model_id).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"Invalid CNN model id: {model_id!r}")
    if _is_yolo_seed(candidate.name):
        raise ValueError("YOLO weights are not a CNN checkpoint")
    return candidate


def load_cnn_model(models_dir: str, model_id: str, device: torch.device, cache: dict) -> torch.nn.Module:
    path = resolve_cnn_path(models_dir, model_id)
    key = str(path)
    if key in cache:
        return cache[key]
    try:
        model = DigitCNN(num_classes=10)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to load CNN model {model_id!r}: {exc}")
    cache[key] = model
    return model
