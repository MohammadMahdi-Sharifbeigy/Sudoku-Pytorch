from pathlib import Path
from app.core.registry.cnn_models import list_cnn_models
from app.core.registry.yolo_models import list_yolo_models


def _touch(p: Path, size: int = 1024) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)


def test_list_cnn_excludes_yolo_seeds(tmp_path):
    _touch(tmp_path / "best_model.pt")
    _touch(tmp_path / "sudoku_all_lr0p001_bs256_ep20.pt")
    _touch(tmp_path / "yolov8s.pt")
    _touch(tmp_path / "yolov8s-pose.pt")
    _touch(tmp_path / "yolo" / "sudoku_pose_ep100.pt")
    ids = {m["id"] for m in list_cnn_models(str(tmp_path))}
    assert ids == {"best_model.pt", "sudoku_all_lr0p001_bs256_ep20.pt"}


def test_list_yolo_includes_seed_and_subdir(tmp_path):
    _touch(tmp_path / "yolov8s-pose.pt")
    _touch(tmp_path / "yolo" / "sudoku_pose_ep100.pt")
    ids = {m["id"] for m in list_yolo_models(str(tmp_path))}
    assert "yolov8s-pose.pt" in ids and "yolo/sudoku_pose_ep100.pt" in ids
