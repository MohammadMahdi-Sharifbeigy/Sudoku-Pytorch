"""Convert the 4-point polygon dataset into a standard YOLOv8 detect dataset.

    python scripts/convert_labels_to_detect.py

Reads  data/Sudoku-Detector.yolov8/ (polygon labels: class x1 y1 x2 y2 x3 y3 x4 y4)
Writes data/sudoku_detect/          (detect labels:  class cx cy w h)
Idempotent — safe to re-run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.scripts_lib_detect import polygon_line_to_detect  # noqa: E402

# Absolute path written into data.yaml so Ultralytics resolves it correctly
# regardless of the working directory training is launched from.
DATA_YAML_TEMPLATE = """path: {root}
train: train/images
val: valid/images
test: test/images
nc: 1
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
            detect_lines = [polygon_line_to_detect(l) for l in lines]
            detect_lines = [dl for dl in detect_lines if dl is not None]
            (dst_lbl / lbl.name).write_text("\n".join(detect_lines) + "\n")
            # copy matching image (any extension)
            for img in src_img.glob(lbl.stem + ".*"):
                shutil.copyfile(img, dst_img / img.name)
            n += 1
        print(f"[{split}] converted {n} label files")
    yaml_text = DATA_YAML_TEMPLATE.format(root=dst_root.resolve().as_posix())
    (dst_root / "data.yaml").write_text(yaml_text)
    print(f"Wrote {dst_root / 'data.yaml'}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "data" / "Sudoku-Detector.yolov8"
    dst = repo_root / "data" / "sudoku_detect"
    convert(src, dst)


if __name__ == "__main__":
    main()
