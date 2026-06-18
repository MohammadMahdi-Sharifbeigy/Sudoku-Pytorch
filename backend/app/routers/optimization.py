import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import APIRouter, Request, HTTPException

from app.core.model import DigitCNN
from app.core.model_files import latest_model_path
from app.core.optimize_model import run_optimization_and_benchmark
from app.schemas import BenchmarkResult, BenchmarkEntry, ModelInfo

router = APIRouter()
_opt_executor = ThreadPoolExecutor(max_workers=1)


def _run_benchmark(model_path: str, models_dir: str) -> list[dict]:
    """Load fresh CPU model and run benchmark — blocking, runs in thread pool."""
    device = torch.device("cpu")
    model = DigitCNN(num_classes=10)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return run_optimization_and_benchmark(model, model_path, device, models_dir)


@router.post("/optimize", response_model=BenchmarkResult)
async def optimize(request: Request) -> BenchmarkResult:
    models_dir: str = request.app.state.models_dir
    model_path = latest_model_path(models_dir, request.app.state.model_path)
    request.app.state.model_path = model_path

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found at '{model_path}'. Train the model first.",
        )

    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(
            _opt_executor,
            _run_benchmark,
            model_path,
            models_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")

    entries = [BenchmarkEntry(**r) for r in raw]
    return BenchmarkResult(success=True, results=entries)


@router.get("/model/info", response_model=ModelInfo)
async def model_info(request: Request) -> ModelInfo:
    models_dir: str = request.app.state.models_dir
    model_path = latest_model_path(models_dir, request.app.state.model_path)
    request.app.state.model_path = model_path

    if not os.path.exists(model_path):
        return ModelInfo(
            exists=False,
            size_mb=None,
            created_at=None,
            total_params=None,
            trainable_params=None,
        )

    stat = os.stat(model_path)
    size_mb = round(stat.st_size / (1024 * 1024), 3)
    created_at = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat()

    model: torch.nn.Module = request.app.state.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ModelInfo(
        exists=True,
        size_mb=size_mb,
        created_at=created_at,
        total_params=total_params,
        trainable_params=trainable_params,
    )
