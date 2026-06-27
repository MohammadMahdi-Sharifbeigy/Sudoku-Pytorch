import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.core.train_yolo import train_yolo

logger = logging.getLogger(__name__)
router = APIRouter()
_is_yolo_training = False
_yolo_lock: asyncio.Lock | None = None
_yolo_executor = ThreadPoolExecutor(max_workers=1)


def _get_lock() -> asyncio.Lock:
    global _yolo_lock
    if _yolo_lock is None:
        _yolo_lock = asyncio.Lock()
    return _yolo_lock


def _worker(data_yaml, seed, epochs, imgsz, batch, device, models_dir, queue, loop):
    def emit(e): loop.call_soon_threadsafe(queue.put_nowait, e)
    try:
        out = train_yolo(data_yaml=data_yaml, seed_weights=seed, epochs=epochs,
                         imgsz=imgsz, batch=batch, device=device, models_dir=models_dir,
                          on_epoch=lambda m: emit({"type": "epoch", **m}))
        emit({"type": "complete", "model_path": out})
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)
        def _clear():
            global _is_yolo_training
            _is_yolo_training = False
        loop.call_soon_threadsafe(_clear)


async def _reload_yolo(app_state: Any, model_path: str) -> None:
    """Load freshly-trained pose weights into app.state without blocking the loop."""
    if not os.path.exists(model_path):
        return
    try:
        from app.core.vision import load_yolo_model
        loop = asyncio.get_running_loop()
        # YOLO(...) + .to(device) is synchronous and slow — run off the event loop.
        model = await loop.run_in_executor(
            _yolo_executor, load_yolo_model, model_path, app_state.device
        )
        app_state.yolo_model = model
        app_state.yolo_model_id = os.path.relpath(model_path, app_state.models_dir).replace("\\", "/")
    except Exception:
        logger.warning("Failed to reload YOLO model after training", exc_info=True)


async def _sse(queue: asyncio.Queue, app_state: Any) -> AsyncGenerator[str, None]:
    last_path = None
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue
        if event is None:
            if last_path:
                await _reload_yolo(app_state, last_path)
            break
        if event.get("type") == "complete":
            last_path = event.get("model_path")
        yield f"data: {json.dumps(event)}\n\n"


@router.get("/train/yolo/stream")
async def train_yolo_stream(
    request: Request,
    epochs: int = Query(default=100, ge=1, le=500),
    imgsz: int = Query(default=640, ge=64, le=1280),
    batch: int = Query(default=8, ge=1, le=128),
    model_seed: str = Query(default="yolov8s.pt"),
) -> StreamingResponse:
    global _is_yolo_training
    async with _get_lock():
        if _is_yolo_training:
            raise HTTPException(status_code=409, detail="YOLO training already in progress.")
        _is_yolo_training = True

    models_dir = request.app.state.models_dir
    data_root = request.app.state.data_path
    data_yaml = os.path.join(data_root, "sudoku_detect", "data.yaml")
    if not os.path.exists(data_yaml):
        async def _err():
            yield f"data: {json.dumps({'type':'error','message':'Detect dataset missing. Run scripts/convert_labels_to_detect.py first.'})}\n\n"
        _is_yolo_training = False
        return StreamingResponse(_err(), media_type="text/event-stream")

    seed_path = os.path.join(models_dir, model_seed)
    if not os.path.exists(seed_path):
        seed_path = model_seed
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_yolo_executor, _worker, data_yaml, seed_path, epochs,
                         imgsz, batch, "auto", models_dir, queue, loop)
    return StreamingResponse(_sse(queue, request.app.state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/train/yolo/status")
async def yolo_status() -> dict:
    return {"is_training": _is_yolo_training}
