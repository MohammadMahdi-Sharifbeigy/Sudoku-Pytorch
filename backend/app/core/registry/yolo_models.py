"""List + cached-load YOLOv8-pose weights (models/yolo/*.pt + seed yolov8s-pose.pt)."""
from __future__ import annotations
import datetime
from pathlib import Path
import torch

YOLO_SUBDIR = "yolo"
POSE_SEED_NAME = "yolov8s-pose.pt"
YOLO_POINTER = "yolo/latest_yolo.txt"


def _entry(path: Path, base: Path, default_id: str | None) -> dict:
    stat = path.stat()
    rel = path.relative_to(base).as_posix()
    return {
        "id": rel, "filename": path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "created_at": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
        "is_default": rel == default_id,
    }


def list_yolo_models(models_dir: str, default_id: str | None = None) -> list[dict]:
    base = Path(models_dir)
    if not base.is_dir():
        return []
    paths = []
    seed = base / POSE_SEED_NAME
    if seed.is_file():
        paths.append(seed)
    sub = base / YOLO_SUBDIR
    if sub.is_dir():
        paths.extend(sorted(sub.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    return [_entry(p, base, default_id) for p in paths]


def resolve_yolo_path(models_dir: str, model_id: str) -> Path:
    base = Path(models_dir).resolve()
    candidate = (base / model_id).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"Invalid YOLO model id: {model_id!r}")
    return candidate


def default_yolo_id(models_dir: str) -> str | None:
    base = Path(models_dir)
    pointer = base / YOLO_POINTER
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value and (base / value).is_file():
            return value
    sub = base / YOLO_SUBDIR
    if sub.is_dir():
        weights = sorted(sub.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if weights:
            return weights[-1].relative_to(base).as_posix()
    return None


def load_yolo_weights(models_dir: str, model_id: str, device: torch.device, cache: dict):
    from ultralytics import YOLO
    path = resolve_yolo_path(models_dir, model_id)
    key = str(path)
    if key in cache:
        return cache[key]
    model = YOLO(str(path))
    model.to(device)
    cache[key] = model
    return model


def write_yolo_pointer(models_dir: str, weights_path: str) -> None:
    base = Path(models_dir).resolve()
    rel = Path(weights_path).resolve().relative_to(base).as_posix()
    pointer = base / YOLO_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(rel, encoding="utf-8")
