import os
from pathlib import Path

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import FileResponse

from app.core.registry.model_files import artifact_path_for_model, latest_model_path

router = APIRouter()

_FORMAT_MEDIA_TYPES: dict[str, str] = {
    "pt": "application/octet-stream",
    "ts": "application/octet-stream",
    "onnx": "application/octet-stream",
}

# Only .txt reports are served; filename must be a bare name with no path separators.
_ALLOWED_REPORT_EXT = ".txt"


def _safe_filename(name: str) -> bool:
    """Reject path traversal, separators, null bytes, and absolute paths."""
    return (
        len(name) > 0
        and "\x00" not in name
        and ".." not in name
        and "/" not in name
        and "\\" not in name
        and not os.path.isabs(name)
    )


@router.get("/models/download")
async def download_model(
    request: Request,
    format: str = Query(..., pattern="^(pt|ts|onnx)$"),
) -> FileResponse:
    models_dir: str = request.app.state.models_dir
    model_path = latest_model_path(models_dir, request.app.state.model_path)
    request.app.state.model_path = model_path

    file_path = model_path if format == "pt" else artifact_path_for_model(model_path, f".{format}")
    filename = os.path.basename(file_path)
    resolved = Path(file_path).resolve()
    base = Path(models_dir).resolve()

    if not resolved.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid model path.")

    if not resolved.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file '{filename}' not found. Run /api/optimize to generate .ts and .onnx formats.",
        )

    return FileResponse(
        path=str(resolved),
        media_type=_FORMAT_MEDIA_TYPES[format],
        filename=filename,
    )


@router.get("/reports/{filename}")
async def download_report(filename: str, request: Request) -> FileResponse:
    if not _safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not filename.endswith(_ALLOWED_REPORT_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"Only {_ALLOWED_REPORT_EXT} reports are served.",
        )

    models_dir: str = request.app.state.models_dir
    resolved = Path(models_dir).resolve() / filename
    base = Path(models_dir).resolve()

    # pathlib.is_relative_to avoids the startswith substring collision bug.
    if not resolved.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")

    return FileResponse(
        path=str(resolved),
        media_type="text/plain",
        filename=filename,
    )
