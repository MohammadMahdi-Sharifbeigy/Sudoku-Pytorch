"""Convert the 4-point polygon dataset into a YOLOv8-pose dataset.

    python scripts/convert_labels_to_pose.py
Writes data/sudoku_pose/ with pose labels + data.yaml. Idempotent.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.scripts_lib_pose import polygon_line_to_pose  # noqa: E402

DATA_YAML = """path: .
train: train/images
val: valid/images
test: test/images
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
names:
  0: sudoku
"""


def convert(src_root: Path, dst_root: Path) -> None:
    for split in ("train", "valid", "test"):
        src_img = src_root / split / "images"
        src_lbl = src_root / split / "labels"
        dst_img = dst_root / split / "images"
        dst_lbl = dst_root / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        n = 0
        if not src_lbl.is_dir():
            print(f"[{split}] no labels dir, skipping")
            continue
        for lbl in src_lbl.glob("*.txt"):
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            pose_lines = [polygon_line_to_pose(l) for l in lines if len(l.split()) == 9]
            (dst_lbl / lbl.name).write_text("\n".join(pose_lines) + "\n")
            # copy matching image (any extension)
            for img in src_img.glob(lbl.stem + ".*"):
                shutil.copyfile(img, dst_img / img.name)
            n += 1
        print(f"[{split}] converted {n} label files")
    (dst_root / "data.yaml").write_text(DATA_YAML)
    print(f"Wrote {dst_root / 'data.yaml'}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "data" / "Sudoku-Detector.yolov8"
    dst = repo_root / "data" / "sudoku_pose"
    convert(src, dst)


if __name__ == "__main__":
    main()
