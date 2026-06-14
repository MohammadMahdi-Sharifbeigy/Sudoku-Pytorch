# Project: Intelligent Sudoku Recognition and Solver System
**Course:** Fundamentals of Computer Vision
**Course Instructor:** Dr. Mohammadreza Mohammadi
**Teaching Assistant:** Mehrshad Fallah
**Deadline:** 1404/04/19 (approx. July 10, 2025)

---

## 1) Introduction and Project Goal

The goal of this project is to design a computer vision system for the automatic recognition and solving of Sudoku from an image. The system should be able to extract the Sudoku grid from the image, recognize the digits, and display the final answer.

Main topics of the project include:
- Image processing with OpenCV
- Convolutional Neural Networks (CNN)
- Digit recognition
- Designing a complete computer vision Pipeline

The main focus of the project is on the computer vision part, correct extraction of the grid, and recognition of digits. For the Sudoku solving part, the use of standard Backtracking algorithm implementations is allowed. Students can use existing open-source and educational implementations.

## 2) Dataset and Data Sources

- Use of standard datasets such as:
  - MNIST
  - Chars74K
- Use of Persian digit datasets is allowed. A suitable example:
  - Hoda Dataset
- Generating synthetic data using:
  - Font change
  - Rotation
  - Noise
  - Brightness change
- Using real Sudoku images for the final test is mandatory.

## 3) Project Phases

### Phase 1 — Sudoku Grid Extraction
- Convert image to Grayscale
- Noise removal and edge extraction
- Grid detection with:
  - `findContours`
  - or Hough Transform
- Apply Perspective Transform
- Divide the grid into 81 cells
- Remove grid lines and detect empty cells
- **Deliverables:**
  - Images of processing steps
  - Sample output of cells
  - Analysis of failure samples

### Phase 2 — Digit Recognition
- Design and train a digit recognition model
- Digits can be:
  - Persian
  - English
- Define the model as 10-class:
  - Numbers 1 to 9
  - Empty cell class
- Use of:
  - CNN
  - or lightweight architectures like MobileNet
- Use of:
  - Data Augmentation
  - Normalization
  - CrossEntropy Loss
- Evaluation with:
  - Accuracy
  - Confusion Matrix
- **Deliverables:**
  - Training chart
  - Evaluation results
  - Error analysis

### Phase 3 — Sudoku Solving
- Convert model output to a 9x9 matrix
- Represent empty cells with a value of zero
- Use Backtracking algorithm
- Validate numbers in:
  - Row
  - Column
  - 3x3 block
- Manage inconsistent or unsolvable grids
- Use of open-source and educational implementations for the Solver is allowed.
- Suggested sample resources:
  - https://github.com/LiorSinai/SudokuSolver-Python
  - https://github.com/aimacode/aima-python

### Phase 4 — Final System
- Complete integration of project steps:
  - Grid extraction
  - Digit recognition
  - Sudoku solving
- Display final answer on the original image
- System performance analysis:
  - Accuracy
  - Execution time
  - Error analysis
- Prepare final report

## 4) Delivery Standard

**The project must be executable and reproducible.**
- Provide `environment.yml` or `requirements.txt` file
- Provide final model weights
- Provide project execution command
- All parts of the project must be delivered as `.py` files.
- Delivering the project solely in Jupyter Notebook format is not acceptable.

## 5) Project Evaluation

- Quality of Sudoku grid extraction
- Digit recognition accuracy
- Correct performance of the final system
- Error analysis and failure samples
- Quality of report and project structure

## 6) Bonus Section

### Option 1 — System Robustness in Real Conditions
- Correct system performance in difficult image conditions such as:
  - Low light
  - Noise
  - Image blur
  - Rotation
  - Extreme viewing angle
  - Shadow
- System performance analysis under these conditions
- Use of appropriate techniques to increase Robustness

### Option 2 — Model Optimization and Deployment
- Convert model to:
  - ONNX
  - TorchScript
- CPU execution speed analysis
- Comparison of execution time and model size before and after optimization

### Option 3 — User Interface (UI)
- Design a simple user interface for running the system
- Display:
  - Input image
  - Extracted grid
  - Final answer
- Suggested tools:
  - Streamlit
  - Gradio
  - PyQt

### Option 4 — Display Answer on Original Image
- Display the final Sudoku answer on the input image
- Maintain original grid perspective
- Correct alignment of output text and image

## 7) Project Rules

- The project is done in pairs (groups of two).
- Both members must be proficient in all parts.
- Use of educational resources and open-source projects is allowed.
- In the presentation session, questions may be asked about all parts of the project. (Except for the Sudoku solving algorithm)
