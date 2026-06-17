import os

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# Map query param → fixed filename within models_dir (no user-controlled paths).
_FORMAT_TO_FILENAME: dict[str, str] = {
    "pt": "best_model.pt",
    "ts": "best_model.ts",
    "onnx": "best_model.onnx",
}

_FORMAT_MEDIA_TYPES: dict[str, str] = {
    "pt": "application/octet-stream",
    "ts": "application/octet-stream",
    "onnx": "application/octet-stream",
}

# Only .txt reports are served; filename must be a bare name with no path separators.
_ALLOWED_REPORT_EXT = ".txt"


def _safe_filename(name: str) -> bool:
    """Reject any filename containing path traversal components or separators."""
    return (
        len(name) > 0
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
    filename = _FORMAT_TO_FILENAME[format]
    file_path = os.path.join(models_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file '{filename}' not found. Run /api/optimize to generate .ts and .onnx formats.",
        )

    return FileResponse(
        path=file_path,
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
    file_path = os.path.realpath(os.path.join(models_dir, filename))
    base_dir = os.path.realpath(models_dir)

    # Double-check resolved path stays inside models_dir.
    if not file_path.startswith(base_dir + os.sep) and file_path != base_dir:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=filename,
    )
