from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/train/status")
async def train_status() -> dict:
    # Full implementation in Section 3
    return {"is_training": False}


@router.get("/train/stream")
async def train_stream() -> JSONResponse:
    # Full SSE implementation in Section 3
    return JSONResponse(
        status_code=501,
        content={"success": False, "error": "Training SSE endpoint not yet implemented."},
    )
