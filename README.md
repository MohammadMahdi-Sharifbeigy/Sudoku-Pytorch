# Intelligent Sudoku Solver 🧩

This project is an end-to-end computer vision and deep learning pipeline built with **PyTorch** and **OpenCV** to automatically detect, extract, recognize, and solve Sudoku puzzles from real-world images.

## Features & Implemented Phases
- **Phase 1: Grid Extraction**: Uses OpenCV for adaptive thresholding, contour detection, and perspective warping to extract the grid.
- **Phase 2: Digit Recognition**: A custom CNN model trained on MNIST and Hoda datasets to recognize both English and Persian digits.
- **Phase 3: Sudoku Solving**: Utilizes a recursive backtracking algorithm to solve the extracted 9x9 grid.
- **Phase 4: Final System**: Re-projects the solved digits perfectly back onto the original angled image.
- **Bonus Option 2 (Model Optimization)**: Converts the PyTorch model to ONNX and TorchScript, offering a benchmarking tool to compare CPU inference speeds.
- **Bonus Option 3 (UI)**: A fully functional Streamlit dashboard to interact with all modes.

---

## 🚀 Execution Command (ارائه دستور اجرای پروژه)

To run the project and access the User Interface (Inference, Training, Optimization, and Robustness Analysis), run the following command in your terminal:

```bash
streamlit run main.py
```

---

## 📦 Model Weights (ارائه وزن مدل نهایی)

The final trained weights of the Convolutional Neural Network are saved locally in this repository at:
`models/best_model.pt`

When you run the application, the system automatically loads this model for inference. If you convert the model using the "Model Optimization" panel, the optimized versions will be saved as:
- `models/best_model.ts` (TorchScript)
- `models/best_model.onnx` (ONNX)

---

## Installation & Setup

1. **Clone the repository** (if you haven't already).
2. **Install the dependencies**. The project explicitly uses a `requirements.txt` file for full reproducibility:

```bash
pip install -r requirements.txt
```

### Dependencies Included:
- `torch`, `torchvision` (Deep Learning)
- `opencv-python` (Computer Vision)
- `streamlit` (User Interface)
- `onnx`, `onnxruntime` (Model Optimization & Deployment)
- `pandas`, `matplotlib`, `scikit-learn` (Data & Metrics)
