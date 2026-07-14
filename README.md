# Intelligent Sudoku Solver

An end-to-end computer vision and deep learning pipeline built with
**PyTorch** and **OpenCV** to detect, extract, recognize, and solve Sudoku
puzzles from real-world photos — supporting both **English** and **Persian
(Hoda)** handwritten/printed digits. The project ships a full Streamlit
application with three modes (Inference, Training, Optimization) plus a
Streamlit-free CLI training runner.

## Features

- **Grid extraction** — multi-threshold adaptive thresholding, contour
  detection with Non-Maximum Suppression, and perspective warping locate and
  flatten the Sudoku grid from an angled/skewed photo.
- **Robust cell extraction (3-tier strategy)** — a contour-based detector
  finds all 81 cells when possible; a KMeans-based partial-grid recovery
  reconstructs the full 9×9 layout when only some cells are detected
  (printed grids with slight skew); a deterministic 9×9 slice fallback
  guarantees output even on faint, wavy, or hand-drawn grids.
- **Digit recognition — six model architectures**:
  - `DigitCNN` — compact 3-block CNN (default, ~250K params).
  - `LegacyDigitCNN` — original 2-conv architecture, kept for backward
    compatibility with older checkpoints.
  - `MultiTaskCNN` — shared backbone with two heads: digit (0–9) and
    language (Persian/English), for automatic script detection.
  - `UnifiedCNN` — single 20-class head (English 0–9 + Persian 0–9) on a
    MobileNetV3-Small or ShuffleNetV2 backbone.
  - `EfficientNetDigit` — EfficientNet-B0 with two-phase transfer learning
    (frozen-backbone warmup, then partial fine-tuning).
- **Sudoku solving** — bitmask-constraint backtracking with a
  Minimum-Remaining-Values heuristic; detects and fails fast on
  self-conflicting givens (e.g. a vision misread).
- **Solution overlay** — the solved digits are rendered back onto the
  original angled photo via the inverse perspective transform.
- **Model optimization** — exports any trained model to **TorchScript** and
  **ONNX**, verifies numerical parity for multi-head models, and benchmarks
  CPU inference latency across all three formats.
- **Full Streamlit dashboard** — three interactive modes covering
  inference, training (with live progress, cancellation, and confusion
  matrices), and optimization/benchmarking.
- **CLI training runner** — trains any model/dataset/schedule combination
  without Streamlit, with a `tqdm` progress bar and clean `Ctrl+C` handling.

---

## Application Modes

Run the Streamlit app and switch modes from the sidebar:

```bash
streamlit run main.py
```

### 1. Inference (Solve)
Upload a photo of a Sudoku puzzle. Configure preprocessing (sharpening,
thresholding, erosion, digit scale) in the sidebar, pick a model (English,
Persian, auto-detecting Multi-Task/Unified, or a specific ONNX file), and
get back the solved puzzle overlaid on your original photo — plus a
downloadable per-cell inference report. A "Preprocessing Debug" tab
visualizes every intermediate CV step without running the classifier.

### 2. Model Training
Configure architecture, dataset mix (MNIST / printed fonts / Hoda Persian /
empty cells, in any combination), augmentation preset, and hyperparameters
(learning rate, batch size, weight decay, LR schedule), then train in the
background with live loss/accuracy charts and a **Stop** button. On
completion, view confusion matrices, per-class accuracy, and download the
full text report.

### 3. Model Optimization
Pick any trained checkpoint, export it to TorchScript and ONNX, and
benchmark CPU inference speed across PyTorch / TorchScript / ONNX. For
dual-head models (Multi-Task, Unified), the ONNX export is numerically
verified against the PyTorch output before benchmarking.

---

## CLI Training (no Streamlit)

```bash
python run_training_cli.py --model DigitCNN --purpose English --dataset "MNIST + Fonts" --epochs 20
python run_training_cli.py --model MultiTaskCNN --purpose Multi --epochs 30 --lr 0.001
python run_training_cli.py --model EfficientNetDigit --purpose Persian --aug none --phase1 6 --phase2 14
```

Run `python run_training_cli.py --help` for the full list of flags
(model/purpose/dataset choice, LR schedule, cosine-schedule parameters,
EfficientNet two-phase epoch counts, backbone choice, etc.).

For the full meaning and effect of every parameter, see
[`vision-report.md`](vision-report.md).

---

## Model Weights

Trained weights live under `models/`, one triplet (`.pt` / `.ts` / `.onnx`)
per variant once exported via the Optimization page:

| Variant | PyTorch checkpoint |
|---|---|
| English-only | `models/best_model_english.pt` |
| Persian-only | `models/best_model_persian.pt` |
| Multi-Task (digit + language) | `models/best_model_multitask.pt` |
| Unified 20-class | `models/best_model_unified20.pt` |
| EfficientNet-B0 (Persian) | `models/best_model_efficientnet_persian.pt` |

The Inference page auto-detects which checkpoints exist and offers the
matching toggles/selectors in the sidebar. TorchScript (`.ts`) and ONNX
(`.onnx`) exports are produced by the Model Optimization page and saved
alongside each `.pt` file.

---

## Installation & Setup

1. **Clone the repository** (if you haven't already).
2. **Install the dependencies**:

```bash
pip install -r requirements.txt
```

### Dependencies

- `torch`, `torchvision` — deep learning (all CNN architectures)
- `opencv-python` — computer vision pipeline
- `streamlit` — user interface
- `onnx`, `onnxruntime` — model optimization & CPU-accelerated inference
- `numpy`, `pandas`, `matplotlib`, `scikit-learn` — data handling, plotting,
  metrics, and KMeans-based partial-grid recovery

### Datasets

Place source data under `data/`:
- `data/DigitDB/` — Hoda Persian handwritten digit database (`.cdb` files).
- `data/digit_images/` — printed-font digit images (folders `1`–`9`).
- MNIST is downloaded automatically on first use via `torchvision.datasets`.

---

## Documentation

[`vision-report.md`](vision-report.md) is the complete technical reference:
every class and function in `src/` and `app/`, the full extraction and
training pipelines, and a parameter-by-parameter table covering what each
tunable value does and the effect of setting it higher or lower.
