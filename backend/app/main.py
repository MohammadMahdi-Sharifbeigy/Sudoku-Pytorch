import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model import DigitCNN
from app.core.registry.model_files import latest_model_path
from app.routers import inference, training, optimization, models as models_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    models_dir = os.environ.get("MODELS_DIR", "models")
    configured_model_path = os.environ.get("MODEL_PATH", "models/best_model.pt")
    model_path = latest_model_path(models_dir, configured_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitCNN(num_classes=10)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    app.state.model = model
    app.state.device = device
    app.state.model_path = model_path
    app.state.models_dir = models_dir
    app.state.data_path = os.environ.get("DATA_PATH", "data")

    yield

    # cleanup (nothing to release here)


default_cors_origins = ",".join(
    [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8050",
        "http://localhost:8080",
    ]
)
cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", default_cors_origins).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Sudoku Solver API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(optimization.router, prefix="/api")
app.include_router(models_router.router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
