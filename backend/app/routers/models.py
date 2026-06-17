from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/models/download")
async def download_model() -> JSONResponse:
    # Full implementation in Section 4
    return JSONResponse(
        status_code=501,
        content={"success": False, "error": "Download endpoint not yet implemented."},
    )


@router.get("/reports/{filename}")
async def download_report(filename: str) -> JSONResponse:
    # Full implementation in Section 4
    return JSONResponse(
        status_code=501,
        content={"success": False, "error": "Reports endpoint not yet implemented."},
    )
