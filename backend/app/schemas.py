from pydantic import BaseModel, Field
from typing import Optional


class CellPrediction(BaseModel):
    index: int
    row: int
    col: int
    has_digit: bool
    label: int
    confidence: float
    cell_image_b64: str  # base64-encoded 28x28 PNG


class ConfidenceStats(BaseModel):
    digit_count: int
    empty_count: int
    avg_confidence: float
    min_confidence: float
    low_confidence_count: int  # cells with confidence < 0.5


class SolveResponse(BaseModel):
    success: bool
    detector_used: str = "classical"
    solved: bool
    original_grid: list[list[int]]
    solved_board: Optional[list[list[int]]]
    per_cell: list[CellPrediction]
    confidence_stats: ConfidenceStats
    solution_image_b64: Optional[str]  # base64-encoded annotated PNG
    board_image_b64: str               # base64-encoded warped grid PNG
    threshold_image_b64: str           # base64-encoded adaptive threshold PNG
    cell_grid_image_b64: str           # base64-encoded extracted cells grid PNG
    error: Optional[str] = None
    detail: Optional[str] = None


class TrainingConfig(BaseModel):
    epochs: int = Field(default=20, ge=1, le=100)
    learning_rate: float = Field(default=0.001, gt=0.0)
    batch_size: int = Field(default=128, ge=1)
    dataset_mode: str = Field(default="mnist_fonts")  # mnist_fonts | mnist_hoda | all


class BenchmarkEntry(BaseModel):
    format: str
    size_mb: float
    latency_ms: Optional[float]
    path: str


class BenchmarkResult(BaseModel):
    success: bool
    results: list[BenchmarkEntry]
    error: Optional[str] = None


class ModelInfo(BaseModel):
    exists: bool
    size_mb: Optional[float]
    created_at: Optional[str]
    total_params: Optional[int]
    trainable_params: Optional[int]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class ModelEntry(BaseModel):
    id: str
    filename: str
    size_mb: float
    created_at: str
    is_default: bool


class ModelList(BaseModel):
    models: list[ModelEntry]


class YoloTrainingConfig(BaseModel):
    epochs: int = Field(default=100, ge=1, le=500)
    imgsz: int = Field(default=640, ge=64, le=1280)
    batch: int = Field(default=8, ge=1, le=128)
    model_seed: str = Field(default="yolov8s-pose.pt")
