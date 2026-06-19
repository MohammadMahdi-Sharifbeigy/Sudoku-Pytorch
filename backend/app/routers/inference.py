import asyncio
import base64
import copy
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException

from app.core.vision import (
    resize_and_maintain_aspect_ratio,
    apply_grayscale_blur_and_threshold,
    get_cells_classical,
    get_cells_yolo,
    get_per_cell_predictions,
    get_predicted_sudoku_grid_torch,
    plot_cell_images_in_grid,
    generate_solution_image,
)
from app.core.solver import SudokuSolver
from app.schemas import (
    CellPrediction,
    ConfidenceStats,
    SolveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _encode_image_b64(img: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_cell_b64(cell_img: np.ndarray) -> str:
    display = cv2.resize(cell_img, (56, 56), interpolation=cv2.INTER_NEAREST)
    success, buf = cv2.imencode(".png", display)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _run_pipeline(
    img_bytes: bytes,
    model: torch.nn.Module,
    device: torch.device,
    detector: str,
    yolo_model: object | None,
) -> dict:
    """Full inference pipeline — runs in thread pool to avoid blocking event loop."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid JPEG or PNG.")

    img = resize_and_maintain_aspect_ratio(img, new_width=1000)
    threshold_image = apply_grayscale_blur_and_threshold(img)

    if detector == "yolo" and yolo_model is not None:
        try:
            cells, M, board_image = get_cells_yolo(img, yolo_model)
            detector_used = "yolo"
        except Exception:
            logger.warning("YOLO detection failed, falling back to classical", exc_info=True)
            cells, M, board_image = get_cells_classical(img)
            detector_used = "classical (yolo fallback)"
    else:
        cells, M, board_image = get_cells_classical(img)
        detector_used = "classical"
    cell_grid_image = plot_cell_images_in_grid(cells)

    per_cell_info = get_per_cell_predictions(model, cells, device)
    grid_array = get_predicted_sudoku_grid_torch(model, cells, device)
    original_grid = grid_array.tolist()

    board_copy = copy.deepcopy(original_grid)
    solver = SudokuSolver(board_copy)
    solved = solver.solve()

    solved_board = solver.board if solved else None

    solution_image_b64 = None
    if solved and solved_board is not None:
        sol_img = generate_solution_image(img, board_image, cells, solved_board, M)
        solution_image_b64 = _encode_image_b64(sol_img)

    board_image_b64 = _encode_image_b64(board_image)

    per_cell_out = []
    for idx, (cell, info) in enumerate(zip(cells, per_cell_info)):
        per_cell_out.append(CellPrediction(
            index=idx,
            row=idx // 9,
            col=idx % 9,
            has_digit=info["has_digit"],
            label=info["label"],
            confidence=info["confidence"],
            cell_image_b64=_encode_cell_b64(cell["img"]),
        ))

    digit_cells = [i for i in per_cell_info if i["has_digit"]]
    confs = [i["confidence"] for i in digit_cells]
    stats = ConfidenceStats(
        digit_count=len(digit_cells),
        empty_count=81 - len(digit_cells),
        avg_confidence=float(np.mean(confs)) if confs else 0.0,
        min_confidence=float(np.min(confs)) if confs else 0.0,
        low_confidence_count=sum(1 for c in confs if c < 0.5),
    )

    return dict(
        solved=solved,
        original_grid=original_grid,
        solved_board=solved_board,
        per_cell=per_cell_out,
        confidence_stats=stats,
        solution_image_b64=solution_image_b64,
        board_image_b64=board_image_b64,
        threshold_image_b64=_encode_image_b64(threshold_image),
        cell_grid_image_b64=_encode_image_b64(cell_grid_image),
        detector_used=detector_used,
    )


@router.post("/solve", response_model=SolveResponse)
async def solve(
    request: Request,
    image: UploadFile = File(...),
    detector: str | None = Form(default=None),
    cnn_model: str | None = Form(default=None),
    yolo_model: str | None = Form(default=None),
) -> SolveResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{image.content_type}'. Allowed: jpeg, png.",
        )

    img_bytes = await image.read()
    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(img_bytes) / 1024 / 1024:.1f} MB). Max 10 MB.",
        )

    device = request.app.state.device
    models_dir = request.app.state.models_dir

    model = request.app.state.model
    if cnn_model:
        from app.core.registry.cnn_models import load_cnn_model
        try:
            model = load_cnn_model(models_dir, cnn_model, device, request.app.state.cnn_cache)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    yolo = request.app.state.yolo_model
    if yolo_model:
        from app.core.registry.yolo_models import load_yolo_weights
        try:
            yolo = load_yolo_weights(models_dir, yolo_model, device, request.app.state.yolo_cache)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    if detector not in (None, "classical", "yolo"):
        raise HTTPException(status_code=400, detail=f"Invalid detector '{detector}'.")
    chosen = detector or ("yolo" if yolo is not None else "classical")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_pipeline, img_bytes, model, device, chosen, yolo
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return SolveResponse(success=True, **result)
