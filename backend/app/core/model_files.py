import os
import re
from pathlib import Path

LATEST_MODEL_POINTER = "latest_model.txt"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "dataset"


def _format_learning_rate(value: float) -> str:
    text = f"{value:.8g}".replace(".", "p")
    return text.replace("-", "m")


def build_training_model_filename(
    dataset_mode: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> str:
    dataset = _slug(dataset_mode.replace("_", "-"))
    lr = _format_learning_rate(learning_rate)
    return f"sudoku_{dataset}_lr{lr}_bs{batch_size}_ep{epochs}.pt"


def build_training_model_path(
    models_dir: str,
    dataset_mode: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> str:
    return str(
        Path(models_dir)
        / build_training_model_filename(dataset_mode, learning_rate, batch_size, epochs)
    )


def pointer_path(models_dir: str) -> Path:
    return Path(models_dir) / LATEST_MODEL_POINTER


def write_latest_model_pointer(models_dir: str, model_path: str) -> None:
    models_base = Path(models_dir)
    models_base.mkdir(parents=True, exist_ok=True)
    resolved_model = Path(model_path).resolve()

    try:
        pointer_value = resolved_model.relative_to(models_base.resolve())
    except ValueError:
        pointer_value = resolved_model

    pointer_path(models_dir).write_text(str(pointer_value), encoding="utf-8")


def latest_model_path(models_dir: str, fallback_model_path: str) -> str:
    base = Path(models_dir).resolve()
    pointer = pointer_path(models_dir)
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            candidate = Path(value)
            if candidate.is_absolute():
                try:
                    candidate.relative_to(base)
                except ValueError:
                    candidate = Path()
            else:
                candidate = Path(models_dir) / candidate
            if candidate.is_file():
                return str(candidate)

    if os.path.exists(fallback_model_path):
        return fallback_model_path

    model_files = sorted(Path(models_dir).glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if model_files:
        return str(model_files[-1])

    return fallback_model_path


def artifact_path_for_model(model_path: str, extension: str) -> str:
    return str(Path(model_path).with_suffix(extension))
