import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Any

import torch
from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.core.model import DigitCNN, FocalLoss
from app.core.train import train_epoch, validate, collect_predictions
from app.core.data_utils import get_dataloaders, get_dataloaders_mnist_hoda, get_dataloaders_all
from app.core.registry.model_files import build_training_model_path, write_latest_model_pointer

router = APIRouter()

_is_training: bool = False
_train_lock: asyncio.Lock | None = None  # initialised lazily (event loop must exist)
_train_executor = ThreadPoolExecutor(max_workers=1)

DATASET_LOADERS = {
    "mnist_fonts": get_dataloaders,
    "mnist_hoda": get_dataloaders_mnist_hoda,
    "all": get_dataloaders_all,
}


def _get_lock() -> asyncio.Lock:
    global _train_lock
    if _train_lock is None:
        _train_lock = asyncio.Lock()
    return _train_lock


def _release_training_flag(loop: asyncio.AbstractEventLoop) -> None:
    """Called from the training thread to release the concurrency guard."""
    global _is_training

    def _clear() -> None:
        global _is_training
        _is_training = False

    loop.call_soon_threadsafe(_clear)


def _run_training_thread(
    model: DigitCNN,
    device: torch.device,
    model_path: str,
    models_dir: str,
    data_path: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    dataset_mode: str,
    queue: asyncio.Queue,  # type: ignore[type-arg]
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Blocking training loop — runs in thread pool. Never calls asyncio directly."""

    def emit(event: dict) -> None:  # type: ignore[type-arg]
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        loader_fn = DATASET_LOADERS[dataset_mode]
        train_loader, val_loader, test_loader = loader_fn(data_path, batch_size)

        criterion = FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            emit({
                "type": "epoch",
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_dir = os.path.dirname(os.path.abspath(model_path))
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                write_latest_model_pointer(models_dir, model_path)
                emit({"type": "best_model", "val_loss": round(val_loss, 6), "epoch": epoch})

        test_loss, test_acc = validate(model, test_loader, criterion, device)
        y_true, y_pred = collect_predictions(model, test_loader, device)
        emit({
            "type": "test_results",
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 4),
            "y_true": y_true,
            "y_pred": y_pred,
        })
        emit({"type": "complete", "model_path": model_path})

    except Exception as exc:
        emit({"type": "error", "message": str(exc)})

    finally:
        # Sentinel tells the SSE generator to stop.
        loop.call_soon_threadsafe(queue.put_nowait, None)
        # Release the concurrency guard AFTER sentinel is queued.
        _release_training_flag(loop)


async def _reload_model(app_state: Any, model_path: str) -> None:
    """Reload best weights into the live inference model after training."""
    if not os.path.exists(model_path):
        return
    try:
        state = torch.load(model_path, map_location=app_state.device, weights_only=True)
        app_state.model.load_state_dict(state)
        app_state.model.eval()
    except Exception:
        pass


async def _sse_generator(
    queue: asyncio.Queue,  # type: ignore[type-arg]
    app_state: Any,
) -> AsyncGenerator[str, None]:
    """Drain training events from queue and yield as SSE. Keepalive ping on timeout."""
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue

        if event is None:  # sentinel — thread finished
            await _reload_model(app_state, app_state.model_path)
            break

        yield f"data: {json.dumps(event)}\n\n"


@router.get("/train/stream")
async def train_stream(
    request: Request,
    epochs: int = Query(default=20, ge=1, le=100),
    learning_rate: float = Query(default=0.001, gt=0.0),
    batch_size: int = Query(default=128, ge=1),
    dataset_mode: str = Query(default="mnist_fonts"),
) -> StreamingResponse:
    global _is_training

    if dataset_mode not in DATASET_LOADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset_mode '{dataset_mode}'. Valid: {list(DATASET_LOADERS)}",
        )

    lock = _get_lock()
    async with lock:
        if _is_training:
            raise HTTPException(status_code=409, detail="Training already in progress.")
        _is_training = True

    device: torch.device = request.app.state.device
    models_dir: str = request.app.state.models_dir
    model_path = build_training_model_path(
        models_dir,
        dataset_mode,
        learning_rate,
        batch_size,
        epochs,
    )
    request.app.state.model_path = model_path
    data_path: str = request.app.state.data_path

    # Fresh model for training — don't mutate the live inference model.
    train_model = DigitCNN(num_classes=10).to(device)

    queue: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
    loop = asyncio.get_running_loop()

    # Fire-and-forget — training thread posts to queue; SSE generator drains it.
    loop.run_in_executor(
        _train_executor,
        _run_training_thread,
        train_model,
        device,
        model_path,
        models_dir,
        data_path,
        epochs,
        learning_rate,
        batch_size,
        dataset_mode,
        queue,
        loop,
    )

    return StreamingResponse(
        _sse_generator(queue, request.app.state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/train/status")
async def train_status() -> dict:  # type: ignore[type-arg]
    return {"is_training": _is_training}
