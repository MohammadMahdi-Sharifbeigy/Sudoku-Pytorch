"""CLI: fine-tune YOLOv8s-detect on the Sudoku bounding-box dataset.

    python scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 16

Run scripts/convert_labels_to_detect.py first to create data/sudoku_detect/.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.train_yolo import train_yolo  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8s-detect on the Sudoku dataset.")
    p.add_argument("--data", default=str(repo_root / "data" / "sudoku_detect" / "data.yaml"))
    p.add_argument("--model", default=str(repo_root / "models" / "yolov8s.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--models-dir", default=str(repo_root / "models"))
    a = p.parse_args()
    out = train_yolo(
        data_yaml=a.data, seed_weights=a.model, epochs=a.epochs, imgsz=a.imgsz,
        batch=a.batch, device=a.device, models_dir=a.models_dir,
        on_epoch=lambda m: print(
            f"epoch {m['epoch']}/{m['epochs']} box={m['box_loss']:.4f} "
            f"cls={m['cls_loss']:.4f} dfl={m['dfl_loss']:.4f} mAP50={m['map50']:.4f}", flush=True),
    )
    print(f"Best weights -> {out}")


if __name__ == "__main__":
    main()
