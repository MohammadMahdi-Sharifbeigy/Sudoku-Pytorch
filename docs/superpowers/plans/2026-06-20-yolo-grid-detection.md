# YOLOv8-Pose Grid Corner Detection + Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the YOLO grid step with a **YOLOv8-pose** model that predicts the grid's 4 corner keypoints directly (fed into `perform_four_point_transform`), with classical CV as a selectable fallback; add per-request CNN/pose model selection across backend + frontend; and train pose via a CLI script and an in-app SSE endpoint. The existing 4-point polygon labels are converted to pose format by a one-time script.

**Architecture:** Reorganize `vision.py` → `vision/` package (`common`, `classical`, `yolo`) and model-file helpers → `registry/` package (model files, CNN list, pose list). Both detectors return the same `(cells, M, board_image)` tuple, so the solve pipeline stays detector-agnostic. Pose inference reads `results[0].keypoints.xy`, orders the 4 corners TL,TR,BR,BL, and warps. Frontend gains a reusable `ModelSelect` dropdown, a detector toggle, and a pose training tab.

**Tech Stack:** FastAPI, PyTorch, Ultralytics YOLOv8-pose (already in `requirements.txt`, 8.4.71 installed; seed `models/yolov8s-pose.pt` present), OpenCV, Next.js 15 (App Router), TypeScript, Playwright.

---

## ⚠️ Commit policy

The user requires explicit approval before any `git commit`. **Do NOT commit during execution.** "Checkpoint" steps stage with `git add` and pause — only `git commit` when the user explicitly says so.

## Conventions

- Backend run dir `backend/`; interpreter `backend/.venv/Scripts/python.exe` (Windows). Tests in `backend/tests/`. `pytest>=8.0` added to `requirements.txt`.
- Frontend run dir `frontend/`: `npm run lint`, `npm run build`, `npx playwright test`.
- Repo root holds `data/`, `models/`, `docs/`.

---

## File Structure

**Backend — created:** `app/core/vision/{__init__,common,classical,yolo}.py`,
`app/core/registry/{__init__,model_files,cnn_models,yolo_models}.py`,
`app/core/train_yolo.py`, `app/routers/yolo_training.py`,
`scripts/train_yolo.py`, `scripts/convert_labels_to_pose.py`,
`tests/{__init__,conftest,test_vision_compat,test_yolo_corners,test_registry,test_label_conversion}.py`.

**Backend — modified:** `app/core/vision.py` (deleted after split),
`app/core/model_files.py` (moved), `app/main.py`, `app/schemas.py`,
`app/routers/{inference,models,optimization,training}.py`, `requirements.txt`.

**Data — created by script:** `data/sudoku_pose/{train,valid,test}/{images,labels}/`, `data/sudoku_pose/data.yaml`.

**Frontend — created/modified:** `lib/api.ts`, `components/ui/ModelSelect.tsx`,
`app/{solve,optimize,train}/page.tsx`, `tests/{solve,train}.spec.ts`.

**Docs:** root `README.md` (+ brief `backend/README.md`).

---

## Phase 0 — Deps

### Task 0.1: Add pytest

**Files:** Modify `backend/requirements.txt`

- [ ] **Step 1:** Append `pytest>=8.0.0` under the dotenv line.
- [ ] **Step 2:** Install — run (from `backend/`): `.venv/Scripts/python.exe -m pip install "pytest>=8.0.0"` → `Successfully installed pytest-...`.
- [ ] **Step 3: Checkpoint** — `git add backend/requirements.txt` (await user).

---

## Phase 1 — Label conversion (pose dataset)

### Task 1.1: Conversion logic (TDD)

**Files:**
- Create: `backend/scripts/convert_labels_to_pose.py`
- Create: `backend/tests/__init__.py` (empty), `backend/tests/conftest.py`
- Create: `backend/tests/test_label_conversion.py`

- [ ] **Step 1: conftest import root**

Create `backend/tests/conftest.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: Write failing test**

Create `backend/tests/test_label_conversion.py`:
```python
from app.scripts_lib_pose import polygon_line_to_pose, order_tl_tr_br_bl


def test_order_canonical():
    # deliberately shuffled: BR, TL, BL, TR
    pts = [(0.9, 0.9), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1)]
    tl, tr, br, bl = order_tl_tr_br_bl(pts)
    assert tl == (0.1, 0.1)
    assert tr == (0.9, 0.1)
    assert br == (0.9, 0.9)
    assert bl == (0.1, 0.9)


def test_polygon_line_to_pose_shape_and_bbox():
    line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"  # TL,TR,BR,BL already
    out = polygon_line_to_pose(line)
    cols = out.split()
    assert len(cols) == 17          # class + cx cy w h + 4*(x y v)
    assert cols[0] == "0"
    cx, cy, w, h = map(float, cols[1:5])
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6
    assert abs(w - 0.8) < 1e-6 and abs(h - 0.8) < 1e-6
    # every visibility flag == 2
    assert cols[7] == "2" and cols[10] == "2" and cols[13] == "2" and cols[16] == "2"
```

- [ ] **Step 3: Run — expect fail (ImportError)**

Run (from `backend/`): `.venv/Scripts/python.exe -m pytest tests/test_label_conversion.py -v` → FAIL (module missing).

- [ ] **Step 4: Implement the conversion library**

Create `backend/app/scripts_lib_pose.py` (importable helpers, so they're unit-testable):
```python
"""Pure helpers for polygon->pose label conversion (unit-tested)."""
from __future__ import annotations


def order_tl_tr_br_bl(pts: list[tuple[float, float]]) -> tuple:
    """Order 4 (x,y) points into TL, TR, BR, BL by coordinate geometry."""
    by_sum = sorted(pts, key=lambda p: p[0] + p[1])
    tl, br = by_sum[0], by_sum[-1]
    rest = [p for p in pts if p is not tl and p is not br]
    # of the remaining two, larger (x - y) is TR, smaller is BL
    rest.sort(key=lambda p: p[0] - p[1])
    bl, tr = rest[0], rest[-1]
    return tl, tr, br, bl


def polygon_line_to_pose(line: str) -> str:
    """`class x1 y1 x2 y2 x3 y3 x4 y4` -> pose `class cx cy w h x y v ...`."""
    parts = line.split()
    cls = parts[0]
    coords = list(map(float, parts[1:9]))
    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    ordered = order_tl_tr_br_bl(pts)

    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    cx = max(0.0, min(1.0, (xmin + xmax) / 2))
    cy = max(0.0, min(1.0, (ymin + ymax) / 2))
    w = max(0.0, min(1.0, xmax - xmin))
    h = max(0.0, min(1.0, ymax - ymin))

    kpts = " ".join(f"{p[0]:.6f} {p[1]:.6f} 2" for p in ordered)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts}"
```

- [ ] **Step 5: Run — expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_label_conversion.py -v` → PASS (2 tests).

### Task 1.2: Conversion CLI script

**Files:**
- Create: `backend/scripts/convert_labels_to_pose.py`

- [ ] **Step 1: Implement script**

Create `backend/scripts/convert_labels_to_pose.py`:
```python
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
```

- [ ] **Step 2: Run conversion for real**

Run (from `backend/`): `.venv/Scripts/python.exe scripts/convert_labels_to_pose.py`
Expected: prints `[train] converted 18 ...`, `[valid] converted 5 ...`, `[test] converted 2 ...`, `Wrote .../data/sudoku_pose/data.yaml`.

- [ ] **Step 3: Verify a converted label**

Run (from repo root): `head -1 data/sudoku_pose/train/labels/$(ls data/sudoku_pose/train/labels | head -1) | awk '{print NF}'`
Expected: `17`.

- [ ] **Step 4: Checkpoint** — `git add backend/app/scripts_lib_pose.py backend/scripts/convert_labels_to_pose.py backend/tests/` (await user). Note: `data/sudoku_pose/` is generated — leave for user to decide on tracking.

---

## Phase 2 — vision package (back-compat split)

### Task 2.1: Compat test (RED)

**Files:** Create `backend/tests/test_vision_compat.py`

- [ ] **Step 1: Write test**

Create `backend/tests/test_vision_compat.py`:
```python
PUBLIC_API = [
    "resize_and_maintain_aspect_ratio", "apply_grayscale_blur_and_threshold",
    "get_quadrilateral_points_in_order", "perform_four_point_transform",
    "center_and_resize_digit", "check_for_digit_in_cell_image",
    "slice_grid_into_cells", "sort_cells_into_grid", "build_grid_from_partial_cells",
    "get_valid_cells_from_image", "get_cells_classical",
    "get_predicted_sudoku_grid_torch", "get_per_cell_predictions",
    "plot_cell_images_in_grid", "generate_solution_image",
    "get_grid_corners_yolo", "get_cells_yolo", "load_yolo_model",
]


def test_public_api_importable():
    import app.core.vision as vision
    missing = [n for n in PUBLIC_API if not hasattr(vision, n)]
    assert not missing, f"missing exports: {missing}"
```

- [ ] **Step 2: Run — expect fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vision_compat.py -v` → FAIL.

### Task 2.2: Split vision.py → package (GREEN)

**Files:** Create `backend/app/core/vision/{common,classical,yolo,__init__}.py`; delete `backend/app/core/vision.py`.

- [ ] **Step 1: `common.py`** — move verbatim from current `vision.py`: `resize_and_maintain_aspect_ratio`, `apply_grayscale_blur_and_threshold`, `get_quadrilateral_points_in_order`, `perform_four_point_transform`, `center_and_resize_digit`, `check_for_digit_in_cell_image`, `_bbox_iou`, `_nms`, `_contour_to_quad`, `locate_cells_within_grid`, `sort_cells_into_grid`, `_clear_border_components`, `slice_grid_into_cells`, `build_grid_from_partial_cells`, `get_predicted_sudoku_grid_torch`, `get_per_cell_predictions`, `plot_cell_images_in_grid`, `generate_solution_image`. Keep imports (`cv2, numpy as np, torch, imutils, sklearn.cluster.KMeans, typing.Optional`).

- [ ] **Step 2: `classical.py`**
```python
"""Classical contour-based Sudoku grid detection."""
from typing import Optional
import cv2
import numpy as np
import imutils
from app.core.vision.common import (
    apply_grayscale_blur_and_threshold, perform_four_point_transform,
    _contour_to_quad, _nms, locate_cells_within_grid, sort_cells_into_grid,
    build_grid_from_partial_cells, slice_grid_into_cells,
)
```
Then move `find_grid_contour_candidates` and `get_valid_cells_from_image` verbatim; rename the latter to `get_cells_classical`; append `get_valid_cells_from_image = get_cells_classical`.

- [ ] **Step 3: `yolo.py` stub**
```python
"""YOLOv8-pose grid corner detection — implemented in Phase 4."""
from __future__ import annotations
import numpy as np
import torch


def load_yolo_model(weights_path: str, device: torch.device):
    raise NotImplementedError


def get_grid_corners_yolo(img: np.ndarray, yolo_model, conf: float = 0.25) -> np.ndarray:
    raise NotImplementedError


def get_cells_yolo(img: np.ndarray, yolo_model, grid_size: int = 576):
    raise NotImplementedError
```

- [ ] **Step 4: `__init__.py` re-export** (same as compat list)
```python
from app.core.vision.common import (
    resize_and_maintain_aspect_ratio, apply_grayscale_blur_and_threshold,
    get_quadrilateral_points_in_order, perform_four_point_transform,
    center_and_resize_digit, check_for_digit_in_cell_image, locate_cells_within_grid,
    sort_cells_into_grid, slice_grid_into_cells, build_grid_from_partial_cells,
    get_predicted_sudoku_grid_torch, get_per_cell_predictions,
    plot_cell_images_in_grid, generate_solution_image,
)
from app.core.vision.classical import (
    find_grid_contour_candidates, get_cells_classical, get_valid_cells_from_image,
)
from app.core.vision.yolo import load_yolo_model, get_grid_corners_yolo, get_cells_yolo

__all__ = [
    "resize_and_maintain_aspect_ratio", "apply_grayscale_blur_and_threshold",
    "get_quadrilateral_points_in_order", "perform_four_point_transform",
    "center_and_resize_digit", "check_for_digit_in_cell_image", "locate_cells_within_grid",
    "sort_cells_into_grid", "slice_grid_into_cells", "build_grid_from_partial_cells",
    "get_predicted_sudoku_grid_torch", "get_per_cell_predictions",
    "plot_cell_images_in_grid", "generate_solution_image", "find_grid_contour_candidates",
    "get_cells_classical", "get_valid_cells_from_image", "load_yolo_model",
    "get_grid_corners_yolo", "get_cells_yolo",
]
```

- [ ] **Step 5:** Delete old module — run (from `backend/`): `rm app/core/vision.py`.

- [ ] **Step 6:** Run compat test — `.venv/Scripts/python.exe -m pytest tests/test_vision_compat.py -v` → PASS.

- [ ] **Step 7:** Import-smoke — `.venv/Scripts/python.exe -c "import app.main; print('ok')"` → `ok`.

- [ ] **Step 8: Checkpoint** — `git add` package + deleted file + test (await user).

---

## Phase 3 — registry package

### Task 3.1: Move model_files.py

**Files:** Create `backend/app/core/registry/{__init__,model_files}.py`; delete `backend/app/core/model_files.py`; update 4 imports.

- [ ] **Step 1:** `mkdir -p app/core/registry && touch app/core/registry/__init__.py && git mv app/core/model_files.py app/core/registry/model_files.py` (plain `mv` if untracked).
- [ ] **Step 2:** Change `from app.core.model_files import` → `from app.core.registry.model_files import` in `main.py:10`, `routers/training.py:14`, `routers/models.py:7`, `routers/optimization.py:11`.
- [ ] **Step 3:** Import-smoke → `ok`.

### Task 3.2: CNN + pose registries (TDD)

**Files:** Create `backend/app/core/registry/{cnn_models,yolo_models}.py`; create `backend/tests/test_registry.py`.

- [ ] **Step 1: Failing test**

Create `backend/tests/test_registry.py`:
```python
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
```

- [ ] **Step 2:** Run → FAIL (ImportError).

- [ ] **Step 3: `cnn_models.py`**
```python
"""List + cached-load digit-CNN checkpoints (models/*.pt, excluding YOLO seeds)."""
from __future__ import annotations
import datetime
from pathlib import Path
import torch
from app.core.model import DigitCNN


def _is_yolo_seed(name: str) -> bool:
    return name.lower().startswith("yolov8")


def _entry(path: Path, base: Path, default_id: str | None) -> dict:
    stat = path.stat()
    rel = path.relative_to(base).as_posix()
    return {
        "id": rel, "filename": path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "created_at": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
        "is_default": rel == default_id,
    }


def list_cnn_models(models_dir: str, default_id: str | None = None) -> list[dict]:
    base = Path(models_dir)
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if _is_yolo_seed(p.name):
            continue
        out.append(_entry(p, base, default_id))
    return out


def resolve_cnn_path(models_dir: str, model_id: str) -> Path:
    base = Path(models_dir).resolve()
    candidate = (base / model_id).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"Invalid CNN model id: {model_id!r}")
    if _is_yolo_seed(candidate.name):
        raise ValueError("YOLO weights are not a CNN checkpoint")
    return candidate


def load_cnn_model(models_dir: str, model_id: str, device: torch.device, cache: dict) -> torch.nn.Module:
    path = resolve_cnn_path(models_dir, model_id)
    key = str(path)
    if key in cache:
        return cache[key]
    model = DigitCNN(num_classes=10)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    cache[key] = model
    return model
```

- [ ] **Step 4: `yolo_models.py`**
```python
"""List + cached-load YOLOv8-pose weights (models/yolo/*.pt + seed yolov8s-pose.pt)."""
from __future__ import annotations
import datetime
from pathlib import Path
import torch

YOLO_SUBDIR = "yolo"
POSE_SEED_NAME = "yolov8s-pose.pt"
YOLO_POINTER = "yolo/latest_yolo.txt"


def _entry(path: Path, base: Path, default_id: str | None) -> dict:
    stat = path.stat()
    rel = path.relative_to(base).as_posix()
    return {
        "id": rel, "filename": path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "created_at": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat(),
        "is_default": rel == default_id,
    }


def list_yolo_models(models_dir: str, default_id: str | None = None) -> list[dict]:
    base = Path(models_dir)
    if not base.is_dir():
        return []
    paths = []
    seed = base / POSE_SEED_NAME
    if seed.is_file():
        paths.append(seed)
    sub = base / YOLO_SUBDIR
    if sub.is_dir():
        paths.extend(sorted(sub.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    return [_entry(p, base, default_id) for p in paths]


def resolve_yolo_path(models_dir: str, model_id: str) -> Path:
    base = Path(models_dir).resolve()
    candidate = (base / model_id).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f"Invalid YOLO model id: {model_id!r}")
    return candidate


def default_yolo_id(models_dir: str) -> str | None:
    base = Path(models_dir)
    pointer = base / YOLO_POINTER
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value and (base / value).is_file():
            return value
    sub = base / YOLO_SUBDIR
    if sub.is_dir():
        weights = sorted(sub.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if weights:
            return weights[-1].relative_to(base).as_posix()
    return None


def load_yolo_weights(models_dir: str, model_id: str, device: torch.device, cache: dict):
    from ultralytics import YOLO
    path = resolve_yolo_path(models_dir, model_id)
    key = str(path)
    if key in cache:
        return cache[key]
    model = YOLO(str(path))
    model.to(device)
    cache[key] = model
    return model


def write_yolo_pointer(models_dir: str, weights_path: str) -> None:
    base = Path(models_dir).resolve()
    rel = Path(weights_path).resolve().relative_to(base).as_posix()
    pointer = base / YOLO_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(rel, encoding="utf-8")
```

- [ ] **Step 5:** Run registry tests → PASS (2).
- [ ] **Step 6: Checkpoint** — `git add` registry + test (await user).

---

## Phase 4 — Pose detection (vision/yolo.py)

### Task 4.1: Keypoint corner extraction (TDD)

**Files:** Modify `backend/app/core/vision/yolo.py`; create `backend/tests/test_yolo_corners.py`.

- [ ] **Step 1: Failing test with a fake pose result**

Create `backend/tests/test_yolo_corners.py`:
```python
import numpy as np
from app.core.vision.yolo import get_grid_corners_yolo


class _KP:
    def __init__(self, xy):
        self.xy = np.array([xy], dtype=np.float32)  # shape (1, 4, 2)


class _Boxes:
    def __init__(self, conf):
        self.conf = np.array(conf, dtype=np.float32)


class _Result:
    def __init__(self, kpts_xy, confs):
        self.keypoints = _KP(kpts_xy)
        self.boxes = _Boxes(confs)


class _FakePose:
    def __init__(self, result):
        self._result = result

    def predict(self, img, conf=0.25, verbose=False, **kw):
        return [self._result]


def test_returns_ordered_corners():
    # one instance, 4 corners shuffled (BR, TL, BL, TR)
    xy = [(340, 340), (60, 60), (60, 340), (340, 60)]
    fake = _FakePose(_Result(xy, [0.9]))
    img = np.zeros((400, 400, 3), np.uint8)
    corners = get_grid_corners_yolo(img, fake)
    assert corners.shape == (4, 2)
    tl, tr, br, bl = corners
    assert tuple(tl) == (60, 60)
    assert tuple(br) == (340, 340)


def test_empty_raises():
    class _Empty(_FakePose):
        def predict(self, img, conf=0.25, verbose=False, **kw):
            return []
    img = np.zeros((400, 400, 3), np.uint8)
    try:
        get_grid_corners_yolo(img, _Empty(None))
        assert False
    except ValueError:
        pass
```

- [ ] **Step 2:** Run → FAIL (stub raises NotImplementedError).

- [ ] **Step 3: Implement `vision/yolo.py`**
```python
"""YOLOv8-pose grid corner detection + cell extraction."""
from __future__ import annotations
import numpy as np
import torch

from app.core.vision.common import (
    get_quadrilateral_points_in_order, perform_four_point_transform,
    locate_cells_within_grid, sort_cells_into_grid, build_grid_from_partial_cells,
    slice_grid_into_cells,
)


def load_yolo_model(weights_path: str, device: torch.device):
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def _best_keypoints(result) -> np.ndarray:
    kpts = getattr(result, "keypoints", None)
    if kpts is None or getattr(kpts, "xy", None) is None:
        raise ValueError("YOLO-pose found no grid")
    xy = np.asarray(kpts.xy)               # (n_instances, n_kpts, 2)
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] < 4:
        raise ValueError("YOLO-pose found no grid")
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
        idx = int(np.argmax(np.asarray(boxes.conf)))
    else:
        idx = 0
    return xy[idx][:4].astype(np.float32)  # (4, 2)


def get_grid_corners_yolo(img: np.ndarray, yolo_model, conf: float = 0.25) -> np.ndarray:
    results = yolo_model.predict(img, conf=conf, verbose=False)
    if not results:
        raise ValueError("YOLO-pose found no grid")
    corners = _best_keypoints(results[0])
    return get_quadrilateral_points_in_order(corners)


def get_cells_yolo(img: np.ndarray, yolo_model, grid_size: int = 576):
    corners = get_grid_corners_yolo(img, yolo_model)
    M, grid_sq = perform_four_point_transform(img, corners, size=grid_size)
    cells = locate_cells_within_grid(grid_sq)
    if len(cells) == 81:
        return sort_cells_into_grid(cells), M, grid_sq
    if len(cells) >= 54:
        recovered = build_grid_from_partial_cells(cells)
        if recovered is not None:
            return recovered, M, grid_sq
    return slice_grid_into_cells(grid_sq), M, grid_sq
```

- [ ] **Step 4:** Run → PASS (2).
- [ ] **Step 5:** Full backend tests + import-smoke — `.venv/Scripts/python.exe -m pytest tests/ -v && .venv/Scripts/python.exe -c "import app.main; print('ok')"` → all pass, `ok`.
- [ ] **Step 6: Checkpoint** — `git add app/core/vision/yolo.py tests/test_yolo_corners.py` (await user).

---

## Phase 5 — Schemas & app state

### Task 5.1: Schemas

**Files:** Modify `backend/app/schemas.py`.

- [ ] **Step 1:** Append:
```python
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
```
And add to `SolveResponse` after `success: bool`: `detector_used: str = "classical"`.
- [ ] **Step 2:** Import-smoke → `ok`.

### Task 5.2: Lifespan loads default pose model

**Files:** Modify `backend/app/main.py`.

- [ ] **Step 1:** After the CNN block (`model.eval()`) add:
```python
    app.state.cnn_cache = {}
    app.state.yolo_cache = {}

    from app.core.registry.yolo_models import default_yolo_id, load_yolo_weights
    app.state.yolo_model = None
    app.state.yolo_model_id = None
    yolo_env = os.environ.get("YOLO_MODEL_PATH")
    if yolo_env and os.path.exists(yolo_env):
        try:
            from app.core.vision import load_yolo_model
            app.state.yolo_model = load_yolo_model(yolo_env, device)
            app.state.yolo_model_id = yolo_env
        except Exception:
            app.state.yolo_model = None
    else:
        yid = default_yolo_id(models_dir)
        if yid is not None:
            try:
                app.state.yolo_model = load_yolo_weights(models_dir, yid, device, app.state.yolo_cache)
                app.state.yolo_model_id = yid
            except Exception:
                app.state.yolo_model = None
```
> Note: `default_yolo_id` looks only in `models/yolo/` + pointer — the seed `yolov8s-pose.pt` is NOT auto-loaded as a detector (it's untrained), so until the user trains, the default detector stays classical.

- [ ] **Step 2:** Import-smoke → `ok`.
- [ ] **Step 3: Checkpoint** — `git add app/schemas.py app/main.py` (await user).

---

## Phase 6 — Inference router (detector + model selection)

### Task 6.1: Pipeline + /solve

**Files:** Modify `backend/app/routers/inference.py`.

- [ ] **Step 1: imports**
```python
from app.core.vision import (
    resize_and_maintain_aspect_ratio, apply_grayscale_blur_and_threshold,
    get_cells_classical, get_cells_yolo, get_per_cell_predictions,
    get_predicted_sudoku_grid_torch, plot_cell_images_in_grid, generate_solution_image,
)
from fastapi import Form
```

- [ ] **Step 2: `_run_pipeline` signature + detector block**
```python
def _run_pipeline(img_bytes, model, device, detector, yolo_model) -> dict:  # type: ignore[type-arg]
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the file is a valid JPEG or PNG.")
    img = resize_and_maintain_aspect_ratio(img, new_width=1000)
    threshold_image = apply_grayscale_blur_and_threshold(img)

    if detector == "yolo" and yolo_model is not None:
        try:
            cells, M, board_image = get_cells_yolo(img, yolo_model)
            detector_used = "yolo"
        except Exception:
            cells, M, board_image = get_cells_classical(img)
            detector_used = "classical (yolo fallback)"
    else:
        cells, M, board_image = get_cells_classical(img)
        detector_used = "classical"
```
Keep the remaining body identical; add `detector_used=detector_used,` to the returned `dict(...)`.

- [ ] **Step 3: `/solve` handler**
```python
@router.post("/solve", response_model=SolveResponse)
async def solve(
    request: Request,
    image: UploadFile = File(...),
    detector: str | None = Form(default=None),
    cnn_model: str | None = Form(default=None),
    yolo_model: str | None = Form(default=None),
) -> SolveResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image.content_type}'. Allowed: jpeg, png.")
    img_bytes = await image.read()
    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large ({len(img_bytes)/1024/1024:.1f} MB). Max 10 MB.")

    device = request.app.state.device
    models_dir = request.app.state.models_dir

    model = request.app.state.model
    if cnn_model:
        from app.core.registry.cnn_models import load_cnn_model
        try:
            model = load_cnn_model(models_dir, cnn_model, device, request.app.state.cnn_cache)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    yolo = request.app.state.yolo_model
    if yolo_model:
        from app.core.registry.yolo_models import load_yolo_weights
        try:
            yolo = load_yolo_weights(models_dir, yolo_model, device, request.app.state.yolo_cache)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    if detector not in (None, "classical", "yolo"):
        raise HTTPException(status_code=400, detail=f"Invalid detector '{detector}'.")
    chosen = detector or ("yolo" if yolo is not None else "classical")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, _run_pipeline, img_bytes, model, device, chosen, yolo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    return SolveResponse(success=True, **result)
```

- [ ] **Step 4:** Import-smoke + boot — `import app.main` → `ok`; boot uvicorn on 8011, `curl -s localhost:8011/health` → `{"status":"ok"}`; kill.
- [ ] **Step 5: Live classical solve smoke** — boot 8011, `curl -s -F "image=@../data/sudoku_images/fa1.jpg" -F "detector=classical" localhost:8011/api/solve | head -c 120` → JSON `{"success":true,"detector_used":"classical"...` (or 400 "No grid detected" — acceptable, proves wiring). Kill.
- [ ] **Step 6: Checkpoint** — `git add app/routers/inference.py` (await user).

---

## Phase 7 — Models listing + optimize selection

### Task 7.1: List endpoints

**Files:** Modify `backend/app/routers/models.py`.

- [ ] **Step 1:** Add:
```python
from app.schemas import ModelList, ModelEntry
from app.core.registry.cnn_models import list_cnn_models
from app.core.registry.yolo_models import list_yolo_models, default_yolo_id
from app.core.registry.model_files import latest_model_path
import os as _os


@router.get("/models/cnn", response_model=ModelList)
async def list_cnn(request: Request) -> ModelList:
    models_dir = request.app.state.models_dir
    default_path = latest_model_path(models_dir, request.app.state.model_path)
    default_id = _os.path.relpath(default_path, models_dir).replace("\\", "/")
    return ModelList(models=[ModelEntry(**m) for m in list_cnn_models(models_dir, default_id)])


@router.get("/models/yolo", response_model=ModelList)
async def list_yolo(request: Request) -> ModelList:
    models_dir = request.app.state.models_dir
    default_id = request.app.state.yolo_model_id or default_yolo_id(models_dir)
    return ModelList(models=[ModelEntry(**m) for m in list_yolo_models(models_dir, default_id)])
```
- [ ] **Step 2:** Boot 8011; `curl -s localhost:8011/api/models/cnn | head -c 300` (CNN list, no `yolov8*`); `curl -s localhost:8011/api/models/yolo | head -c 300` (includes `yolov8s-pose.pt`). Kill.

### Task 7.2: optimize cnn_model

**Files:** Modify `backend/app/routers/optimization.py`.

- [ ] **Step 1:** Change handler:
```python
from fastapi import Body

@router.post("/optimize", response_model=BenchmarkResult)
async def optimize(request: Request, cnn_model: str | None = Body(default=None, embed=True)) -> BenchmarkResult:
    models_dir: str = request.app.state.models_dir
    if cnn_model:
        from app.core.registry.cnn_models import resolve_cnn_path
        try:
            model_path = str(resolve_cnn_path(models_dir, cnn_model))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        model_path = latest_model_path(models_dir, request.app.state.model_path)
        request.app.state.model_path = model_path
    # ... rest unchanged (existence check, executor call, return)
```
- [ ] **Step 2:** Import-smoke → `ok`.
- [ ] **Step 3: Checkpoint** — `git add app/routers/models.py app/routers/optimization.py` (await user).

---

## Phase 8 — Pose training (core + CLI + SSE)

### Task 8.1: Core trainer

**Files:** Create `backend/app/core/train_yolo.py`.

- [ ] **Step 1:**
```python
"""Ultralytics YOLOv8-pose fine-tuning — shared by the CLI script and SSE endpoint."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Callable, Optional
import torch


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo(
    data_yaml: str, seed_weights: str, epochs: int, imgsz: int, batch: int,
    device: str, models_dir: str, on_epoch: Optional[Callable[[dict], None]] = None,
) -> str:
    from ultralytics import YOLO
    resolved_device = pick_device(device)
    model = YOLO(str(seed_weights))   # task=pose inferred from the seed

    if on_epoch is not None:
        def _cb(trainer):
            metrics = getattr(trainer, "metrics", {}) or {}
            losses = {}
            li = getattr(trainer, "label_loss_items", None)
            if li and getattr(trainer, "loss_items", None) is not None:
                try:
                    losses = li(trainer.loss_items)
                except Exception:
                    losses = {}
            on_epoch({
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "epochs": int(epochs),
                "box_loss": float(losses.get("train/box_loss", 0.0)),
                "pose_loss": float(losses.get("train/pose_loss", 0.0)),
                "map50": float(metrics.get("metrics/mAP50(P)", metrics.get("metrics/mAP50(B)", 0.0))),
            })
        model.add_callback("on_fit_epoch_end", _cb)

    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        device=resolved_device, project=str(Path(models_dir) / "yolo" / "runs"),
        name="train", exist_ok=True, verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out_dir = Path(models_dir) / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"sudoku_pose_ep{epochs}_imgsz{imgsz}.pt"
    shutil.copyfile(best, dest)
    from app.core.registry.yolo_models import write_yolo_pointer
    write_yolo_pointer(models_dir, str(dest))
    return str(dest)
```
- [ ] **Step 2:** Import-smoke — `.venv/Scripts/python.exe -c "import app.core.train_yolo; print('ok')"` → `ok`.

### Task 8.2: CLI script

**Files:** Create `backend/scripts/train_yolo.py`.

- [ ] **Step 1:**
```python
"""CLI: fine-tune YOLOv8-pose on the converted Sudoku corner dataset.

    python scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 8
(run scripts/convert_labels_to_pose.py first to create data/sudoku_pose/)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.train_yolo import train_yolo  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8-pose on Sudoku corners.")
    p.add_argument("--data", default=str(repo_root / "data" / "sudoku_pose" / "data.yaml"))
    p.add_argument("--model", default=str(repo_root / "models" / "yolov8s-pose.pt"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--models-dir", default=str(repo_root / "models"))
    a = p.parse_args()
    out = train_yolo(
        data_yaml=a.data, seed_weights=a.model, epochs=a.epochs, imgsz=a.imgsz,
        batch=a.batch, device=a.device, models_dir=a.models_dir,
        on_epoch=lambda m: print(
            f"epoch {m['epoch']}/{m['epochs']} box={m['box_loss']:.4f} "
            f"pose={m['pose_loss']:.4f} mAP50={m['map50']:.4f}", flush=True),
    )
    print(f"Best weights -> {out}")


if __name__ == "__main__":
    main()
```
- [ ] **Step 2: 1-epoch CPU smoke** (after conversion already run in Phase 1):
Run (from `backend/`): `.venv/Scripts/python.exe scripts/train_yolo.py --epochs 1 --imgsz 320 --batch 2 --device cpu`
Expected: prints `epoch 1/1 ...`, ends `Best weights -> .../models/yolo/sudoku_pose_ep1_imgsz320.pt`, and `models/yolo/latest_yolo.txt` created.
> If CPU training is too slow, interrupt after confirming a run dir + first epoch line appear; record that in the execution log.

### Task 8.3: SSE endpoint

**Files:** Create `backend/app/routers/yolo_training.py`; modify `backend/app/main.py`.

- [ ] **Step 1:** Create `backend/app/routers/yolo_training.py`:
```python
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.core.train_yolo import train_yolo

router = APIRouter()
_is_yolo_training = False
_yolo_lock: asyncio.Lock | None = None
_yolo_executor = ThreadPoolExecutor(max_workers=1)


def _get_lock() -> asyncio.Lock:
    global _yolo_lock
    if _yolo_lock is None:
        _yolo_lock = asyncio.Lock()
    return _yolo_lock


def _worker(data_yaml, seed, epochs, imgsz, batch, device, models_dir, queue, loop):
    def emit(e): loop.call_soon_threadsafe(queue.put_nowait, e)
    try:
        out = train_yolo(data_yaml=data_yaml, seed_weights=seed, epochs=epochs,
                         imgsz=imgsz, batch=batch, device=device, models_dir=models_dir,
                         on_epoch=lambda m: emit({"type": "epoch", **m}))
        emit({"type": "complete", "model_path": out})
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)
        def _clear():
            global _is_yolo_training
            _is_yolo_training = False
        loop.call_soon_threadsafe(_clear)


async def _reload_yolo(app_state: Any, model_path: str) -> None:
    if not os.path.exists(model_path):
        return
    try:
        from app.core.vision import load_yolo_model
        app_state.yolo_model = load_yolo_model(model_path, app_state.device)
        app_state.yolo_model_id = os.path.relpath(model_path, app_state.models_dir).replace("\\", "/")
    except Exception:
        pass


async def _sse(queue: asyncio.Queue, app_state: Any) -> AsyncGenerator[str, None]:
    last_path = None
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue
        if event is None:
            if last_path:
                await _reload_yolo(app_state, last_path)
            break
        if event.get("type") == "complete":
            last_path = event.get("model_path")
        yield f"data: {json.dumps(event)}\n\n"


@router.get("/train/yolo/stream")
async def train_yolo_stream(
    request: Request,
    epochs: int = Query(default=100, ge=1, le=500),
    imgsz: int = Query(default=640, ge=64, le=1280),
    batch: int = Query(default=8, ge=1, le=128),
    model_seed: str = Query(default="yolov8s-pose.pt"),
) -> StreamingResponse:
    global _is_yolo_training
    async with _get_lock():
        if _is_yolo_training:
            raise HTTPException(status_code=409, detail="YOLO training already in progress.")
        _is_yolo_training = True

    models_dir = request.app.state.models_dir
    data_root = request.app.state.data_path
    data_yaml = os.path.join(data_root, "sudoku_pose", "data.yaml")
    if not os.path.exists(data_yaml):
        async def _err():
            yield f"data: {json.dumps({'type':'error','message':'Pose dataset missing. Run scripts/convert_labels_to_pose.py first.'})}\n\n"
        global _is_yolo_training
        _is_yolo_training = False
        return StreamingResponse(_err(), media_type="text/event-stream")

    seed_path = os.path.join(models_dir, model_seed)
    if not os.path.exists(seed_path):
        seed_path = model_seed
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_yolo_executor, _worker, data_yaml, seed_path, epochs,
                         imgsz, batch, "auto", models_dir, queue, loop)
    return StreamingResponse(_sse(queue, request.app.state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/train/yolo/status")
async def yolo_status() -> dict:
    return {"is_training": _is_yolo_training}
```
> Fix the double `global _is_yolo_training` during implementation: declare it once at function top; the early-error branch just sets it False before returning.

- [ ] **Step 2:** Register in `main.py` — add `yolo_training` to the routers import and `app.include_router(yolo_training.router, prefix="/api")`.
- [ ] **Step 3:** Import-smoke → `ok`.
- [ ] **Step 4: Checkpoint** — `git add app/core/train_yolo.py scripts/train_yolo.py app/routers/yolo_training.py app/main.py` (await user).

---

## Phase 9 — Frontend

> Per `frontend/AGENTS.md`: skim `node_modules/next/dist/docs/` for App-Router changes before page edits. Use **frontend-design** + **ui-ux-pro-max** for new components; match the dark/cyan theme.

### Task 9.1: API client

**Files:** Modify `frontend/lib/api.ts`.

- [ ] **Step 1:** Add types + functions:
```typescript
export type Detector = "classical" | "yolo";

export interface ModelEntry {
  id: string; filename: string; size_mb: number; created_at: string; is_default: boolean;
}

export async function listCnnModels(signal?: AbortSignal): Promise<ModelEntry[]> {
  const d = await requestJson<{ models: ModelEntry[] }>(`${API_BASE}/api/models/cnn`, { signal });
  return d.models;
}
export async function listYoloModels(signal?: AbortSignal): Promise<ModelEntry[]> {
  const d = await requestJson<{ models: ModelEntry[] }>(`${API_BASE}/api/models/yolo`, { signal });
  return d.models;
}

export interface SolveOptions { detector?: Detector; cnnModel?: string; yoloModel?: string; }

export function yoloTrainingStreamUrl(c: { epochs: number; imgsz: number; batch: number; modelSeed?: string }): string {
  const url = new URL(`${API_BASE}/api/train/yolo/stream`);
  url.searchParams.set("epochs", String(c.epochs));
  url.searchParams.set("imgsz", String(c.imgsz));
  url.searchParams.set("batch", String(c.batch));
  if (c.modelSeed) url.searchParams.set("model_seed", c.modelSeed);
  return url.toString();
}
```
Add `detector_used?: string;` to both `SolveResponse` and `BackendSolveResponse`; set it in `normalizeSolveResponse` (`detector_used: data.detector_used`). Change `solveSudoku` to `(file, opts: SolveOptions = {}, signal?)` appending `detector`/`cnn_model`/`yolo_model` form fields when set; update `solveImage` to `solveSudoku(file, {}, signal)`. Change `runOptimization` to `(cnnModel?: string, signal?)` POSTing JSON `{ cnn_model }` with `Content-Type: application/json`.

- [ ] **Step 2:** `npm run lint` → clean.

### Task 9.2: ModelSelect (frontend-design + ui-ux-pro-max)

**Files:** Create `frontend/components/ui/ModelSelect.tsx`.

- [ ] **Step 1:** Controlled themed `<select>`; props `{ label, models, value: string|null, onChange:(id|null)=>void, loading?, disabled? }`. First option `Default (server)` → `null`; each model option shows `filename · {size_mb} MB` + `(default)` when `is_default`. Use `var(--surface)`, `var(--border-col)`, `var(--cyan)`, Space-Mono for filenames, min-height 44px, focus-visible cyan ring. ≤120 lines.
- [ ] **Step 2:** `npm run lint` → clean.

### Task 9.3: Solve page

**Files:** Modify `frontend/app/solve/page.tsx`.

- [ ] **Step 1:** State: `detector` (default `"yolo"`), `cnnModel`/`yoloModel` (`null`), `cnnModels`/`yoloModels`. On mount call `listCnnModels()`/`listYoloModels()` (errors → empty lists, non-fatal).
- [ ] **Step 2:** Render above Solve controls: Classical│YOLO segmented toggle + `<ModelSelect label="Digit model (CNN)" .../>` + (when `detector==="yolo"`) `<ModelSelect label="Grid model (pose)" .../>`.
- [ ] **Step 3:** Solve call → `solveSudoku(file, { detector, cnnModel: cnnModel ?? undefined, yoloModel: yoloModel ?? undefined }, signal)`. Render `result.detector_used` mono badge in the result header.
- [ ] **Step 4:** `npm run lint && npm run build` → succeeds.

### Task 9.4: Optimize page

**Files:** Modify `frontend/app/optimize/page.tsx`.

- [ ] **Step 1:** Add `cnnModels`/`cnnModel` state + `listCnnModels()` on mount; render `<ModelSelect label="Model to optimize" .../>`; call `runOptimization(cnnModel ?? undefined, signal)`.
- [ ] **Step 2:** `npm run lint && npm run build` → succeeds.

### Task 9.5: Train page pose tab

**Files:** Modify `frontend/app/train/page.tsx`.

- [ ] **Step 1:** Top toggle `Digit CNN │ Grid Pose`. CNN branch unchanged. Pose branch: `epochs / imgsz / batch` form + "Start pose training" → `EventSource(yoloTrainingStreamUrl({epochs, imgsz, batch}))`.
- [ ] **Step 2:** On `message` parse JSON: `type:"epoch"` → push `{epoch, box_loss, pose_loss, map50}` into a series + render a minimal line/list; `type:"complete"` → toast + close; `type:"error"` → toast message + close. Button disabled while streaming (single-flight).
- [ ] **Step 3:** `npm run lint && npm run build` → succeeds.
- [ ] **Step 4: Checkpoint** — `git add frontend/lib/api.ts frontend/components/ui/ModelSelect.tsx frontend/app/solve/page.tsx frontend/app/optimize/page.tsx frontend/app/train/page.tsx` (await user).

---

## Phase 10 — Playwright + docs

### Task 10.1: e2e specs

**Files:** Modify `frontend/tests/solve.spec.ts`, `frontend/tests/train.spec.ts`.

- [ ] **Step 1:** Solve spec — `page.route` mock `/api/models/cnn` + `/api/models/yolo` returning a small fixture; assert detector toggle + CNN ModelSelect visible; switching to YOLO reveals the pose ModelSelect.
- [ ] **Step 2:** Train spec — assert `Grid Pose` sub-tab selectable and reveals imgsz/batch fields.
- [ ] **Step 3:** Run — `npx playwright test tests/solve.spec.ts tests/train.spec.ts` → pass.

### Task 10.2: README

**Files:** Modify root `README.md` (+ brief `backend/README.md`).

- [ ] **Step 1:** Add "## Grid detection: YOLOv8-pose" covering:
  - the dataset has 4-corner polygon labels; convert to pose:
    ```bash
    cd backend
    .venv/Scripts/python.exe scripts/convert_labels_to_pose.py
    ```
  - train (CLI):
    ```bash
    .venv/Scripts/python.exe scripts/train_yolo.py --epochs 100 --imgsz 640 --batch 8 --device auto
    ```
  - in-app training (Train page → Grid Pose tab),
  - seed `models/yolov8s-pose.pt`; outputs `models/yolo/sudoku_pose_*.pt` + pointer `models/yolo/latest_yolo.txt`,
  - detector selection (pose default once trained weights exist; Solve-page toggle; `YOLO_MODEL_PATH` override),
  - per-request CNN + pose model dropdowns,
  - note the tiny dataset → add more annotated images for better accuracy.
- [ ] **Step 2: Checkpoint** — `git add` specs + README (await user).

---

## Phase 11 — Final verification & review

### Task 11.1: Verify

- [ ] **Step 1:** Backend tests — `.venv/Scripts/python.exe -m pytest tests/ -v` → all pass.
- [ ] **Step 2:** Boot uvicorn 8011; sweep `/health`, `/api/models/cnn`, `/api/models/yolo`, `/api/solve` (classical; yolo if trained weights exist). Kill.
- [ ] **Step 3:** Frontend — `npm run lint && npm run build && npx playwright test` → green.

### Task 11.2: caveman-review

- [ ] **Step 1:** Run `caveman:caveman-review` (or `caveman:cavecrew-reviewer` agent) over the full diff; fix high/medium findings inline.
- [ ] **Step 2:** Stage everything, summarize for the user, and **wait for explicit approval before committing** (commit policy). On approval, commit in Conventional-Commit groups (label conversion, vision/registry reorg, pose detection, model selection, training, frontend, docs).

---

## Self-Review notes

- **Spec coverage:** label conversion + canonical order (Ph1), vision reorg (Ph2), registry/model select (Ph3,6,7), pose corner detection (Ph4), app-state pose load (Ph5), detector-aware solve (Ph6), list endpoints + optimize select (Ph7), pose trainer + CLI + SSE (Ph8), frontend toggle/dropdowns/pose-train tab (Ph9), README (Ph10), verify + review (Ph11). All spec sections mapped.
- **Type consistency:** `get_cells_classical`/`get_cells_yolo` both return `(cells, M, board_image)`. `_best_keypoints` returns `(4,2)`; ordered by `get_quadrilateral_points_in_order`. `ModelEntry` fields identical backend↔TS (`id, filename, size_mb, created_at, is_default`). `_run_pipeline(img_bytes, model, device, detector, yolo_model)` matches its executor call. `detector_used` added to dict + schema + TS + normalizer. Pose seed exclusion uses `startswith("yolov8")` so both `yolov8s.pt` and `yolov8s-pose.pt` drop from the CNN list.
- **Known fix-up:** `yolo_training.py` early-error branch must declare `global _is_yolo_training` once at function top (flagged inline in Task 8.3 Step 1).
- **Commit policy:** Checkpoints stage only; no commits without explicit user approval.
```
