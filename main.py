import os
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# --- Local Module Imports ---
from model import DigitCNN, FocalLoss
from solver import SudokuSolver
from vision import (
    resize_and_maintain_aspect_ratio,
    apply_grayscale_blur_and_threshold,
    get_valid_cells_from_image,
    get_predicted_sudoku_grid_torch,
    generate_solution_image,
    plot_cell_images_in_grid
)
from train import train_epoch, validate, collect_predictions
from data_utils import get_dataloaders, get_dataloaders_mnist_hoda, get_dataloaders_all
from report_utils import save_training_report, save_inference_report

# --- UI Configuration ---
st.set_page_config(
    page_title="AI Sudoku Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stProgress .st-bo { background-color: #4CAF50; }
    .cell-box { border: 1px solid #555; border-radius: 4px; padding: 4px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("Intelligent Sudoku Solver (PyTorch + OpenCV)")
st.markdown("An end-to-end computer vision and deep learning pipeline for detecting and solving Sudoku puzzles.")

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["Inference (Solve)", "Model Training"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Compute Device:** `{device}`")


# ==========================================
# HELPER: Per-cell prediction details
# ==========================================
def get_per_cell_predictions(model, cells, device):
    """Returns predicted label and confidence for every cell."""
    results = []
    for cell in cells:
        if not cell['contains_digit']:
            results.append({'label': 0, 'confidence': 1.0, 'has_digit': False})
            continue
        img = cell['img'].astype('float32') / 255.0
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
        results.append({'label': pred, 'confidence': conf, 'has_digit': True})
    return results


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Draws and returns a matplotlib confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color='white', fontsize=11)
    ax.set_yticklabels(class_names, color='white', fontsize=11)
    ax.set_xlabel('Predicted Label', color='white', fontsize=13, labelpad=10)
    ax.set_ylabel('True Label', color='white', fontsize=13, labelpad=10)
    ax.set_title('Confusion Matrix (Test Set)', color='white', fontsize=15, pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if count == 0:
                continue
            ax.text(j, i, str(count),
                    ha='center', va='center', fontsize=9,
                    color='black' if count > thresh else 'white',
                    fontweight='bold')

    plt.tight_layout()
    return fig


# ==========================================
# MODE 1: INFERENCE (SOLVE SUDOKU)
# ==========================================
if app_mode == "Inference (Solve)":
    st.markdown("### Upload Sudoku Image")
    uploaded_file = st.file_uploader("Drag and drop your Sudoku image here", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

        # Load Model
        model = DigitCNN(num_classes=10).to(device)
        model_path = 'models/best_model.pt'

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            st.error(f"Trained model not found at `{model_path}`. Please train the model first.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original Image")
            st.image(img, use_container_width=True)

        with st.spinner('Processing image and extracting grid...'):
            try:
                # --- Step-by-Step Vision Pipeline ---
                with st.expander("View Intermediate Processing Steps", expanded=False):
                    step_col1, step_col2, step_col3 = st.columns(3)

                    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)
                    with step_col1:
                        st.markdown("**1. Adaptive Thresholding**")
                        st.image(thresh, use_container_width=True, channels="GRAY")

                    cells, M, board_image = get_valid_cells_from_image(img)
                    with step_col2:
                        st.markdown("**2. Perspective Transform**")
                        st.image(board_image, use_container_width=True, channels="GRAY")

                    with step_col3:
                        st.markdown("**3. Cell Extraction**")
                        fig = plot_cell_images_in_grid(cells)
                        st.pyplot(fig)

                # --- Per-cell Prediction Detail ---
                per_cell = get_per_cell_predictions(model, cells, device)

                with st.expander("🔍 Cell-by-Cell Extraction & Prediction Details", expanded=True):
                    st.markdown(
                        "Each cell shows its **extracted image**, whether a digit was **detected**, "
                        "the **predicted digit**, and the model's **confidence score**. "
                        "🟢 = high confidence (≥80%), 🟡 = medium (50–80%), 🔴 = low (<50%)."
                    )
                    st.markdown("---")

                    COLS = 9
                    grid_cols = st.columns(COLS)

                    # Column headers
                    for c in range(COLS):
                        with grid_cols[c]:
                            st.markdown(f"<div style='text-align:center;color:#888;font-size:11px;'>Col {c}</div>",
                                        unsafe_allow_html=True)

                    for row in range(9):
                        grid_cols = st.columns(COLS)
                        for col in range(COLS):
                            idx = row * 9 + col
                            cell = cells[idx]
                            info = per_cell[idx]

                            with grid_cols[col]:
                                # Display cell image
                                cell_img_display = cell['img']
                                st.image(cell_img_display, width=60, channels="GRAY",
                                         caption=None)

                                if not info['has_digit']:
                                    st.markdown(
                                        "<div style='text-align:center;font-size:11px;color:#888;'>Empty</div>",
                                        unsafe_allow_html=True)
                                else:
                                    label = info['label']
                                    conf = info['confidence']
                                    conf_pct = conf * 100

                                    if conf_pct >= 80:
                                        dot = "🟢"
                                    elif conf_pct >= 50:
                                        dot = "🟡"
                                    else:
                                        dot = "🔴"

                                    st.markdown(
                                        f"<div style='text-align:center;font-size:13px;font-weight:bold;'>{dot} {label}</div>"
                                        f"<div style='text-align:center;font-size:10px;color:#aaa;'>{conf_pct:.1f}%</div>",
                                        unsafe_allow_html=True)

                        # Row separator every 3 rows
                        if row in (2, 5):
                            st.markdown("<hr style='border-color:#444;margin:4px 0;'>", unsafe_allow_html=True)

                # --- Prediction & Solving ---
                grid_array = get_predicted_sudoku_grid_torch(model, cells, device)
                solver = SudokuSolver(board=copy.deepcopy(grid_array))
                solved_board = solver.board if solver.solve() else None

                # --- Save inference text report ---
                os.makedirs('models', exist_ok=True)
                inf_report_path = save_inference_report(
                    image_name=uploaded_file.name,
                    cells=cells,
                    per_cell_info=per_cell,
                    grid_array=grid_array,
                    solved_board=solved_board,
                    output_path='models/inference_report.txt',
                )
                st.sidebar.success(f"📄 Inference report saved to `{inf_report_path}`")

                if solved_board is not None:
                    final_image = generate_solution_image(
                        full_image=img, board_image=board_image,
                        cells_list=cells, solved_board_arr=solved_board, M_matrix=M
                    )

                    with col2:
                        st.markdown("#### Solved Sudoku")
                        st.image(final_image, use_container_width=True)
                        st.success("Sudoku solved successfully!")

                    st.markdown("### Digital Representation")
                    matrix_df = pd.DataFrame(solved_board)
                    st.dataframe(
                        matrix_df.style.set_properties(**{'text-align': 'center', 'font-weight': 'bold'}),
                        use_container_width=True
                    )

                    # Inline download button for inference report
                    with open(inf_report_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="⬇️ Download Inference Report (.txt)",
                            data=f.read(),
                            file_name='inference_report.txt',
                            mime='text/plain',
                        )
                else:
                    st.error("The extracted grid is invalid or unsolvable. Please ensure the image is clear and well-lit.")
                    st.markdown("**Extracted Grid (before solving):**")
                    st.dataframe(pd.DataFrame(grid_array), use_container_width=True)

            except Exception as e:
                import traceback
                st.error(f"Computer Vision Pipeline Error: {e}")
                st.code(traceback.format_exc())


# ==========================================
# MODE 2: MODEL TRAINING (CNN)
# ==========================================
elif app_mode == "Model Training":
    st.markdown("### Model Training Dashboard")
    st.markdown("Configure hyperparameters and monitor the CNN training process in real-time.")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    with param_col2:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
    with param_col3:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)

    data_path = st.text_input("Dataset Directory Path", value="data")
    dataset_mode = st.radio(
        "Training Dataset",
        ["MNIST + Fonts (Recommended for printed Sudoku)", "MNIST + Hoda", "MNIST + Fonts + Hoda (All)"],
        index=0,
        help="Font images from data/digit_images are critical for recognizing printed/typed Sudoku digits."
    )

    if st.button("Start Training Sequence", use_container_width=True):
        if not os.path.exists(data_path):
            st.error(f"Dataset path `{data_path}` does not exist. Please verify the path.")
            st.stop()

        st.info("Initializing DataLoaders. This may take a moment...")

        try:
            # 1. Load Data
            if dataset_mode.startswith("MNIST + Fonts + Hoda"):
                train_loader, val_loader, test_loader = get_dataloaders_all(data_path, batch_size=batch_size)
            elif dataset_mode.startswith("MNIST + Hoda"):
                train_loader, val_loader, test_loader = get_dataloaders_mnist_hoda(data_path, batch_size=batch_size)
            else:
                train_loader, val_loader, test_loader = get_dataloaders(data_path, batch_size=batch_size)

            # 2. Initialize Model, Loss, Optimizer
            model = DigitCNN(num_classes=10).to(device)
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            # 3. Setup UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_table = st.empty()

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Loss Curve")
                loss_placeholder = st.empty()
            with chart_col2:
                st.markdown("#### Accuracy Curve")
                acc_placeholder = st.empty()

            best_val_loss = float('inf')
            os.makedirs('models', exist_ok=True)

            history = {'Train Loss': [], 'Val Loss': [], 'Train Acc': [], 'Val Acc': []}

            # 4. Training Loop
            for epoch in range(int(epochs)):
                status_text.markdown(f"**Running Epoch {epoch + 1}/{epochs}...**")

                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'models/best_model.pt')

                history['Train Loss'].append(train_loss)
                history['Val Loss'].append(val_loss)
                history['Train Acc'].append(train_acc)
                history['Val Acc'].append(val_acc)

                loss_df = pd.DataFrame({'Train Loss': history['Train Loss'], 'Val Loss': history['Val Loss']})
                acc_df = pd.DataFrame({'Train Acc': history['Train Acc'], 'Val Acc': history['Val Acc']})
                loss_placeholder.line_chart(loss_df)
                acc_placeholder.line_chart(acc_df)

                metrics_table.markdown(f"""
                | Metric | Training | Validation |
                |---|---|---|
                | **Loss** | {train_loss:.4f} | {val_loss:.4f} |
                | **Accuracy** | {train_acc:.2f}% | {val_acc:.2f}% |
                """)

                progress_bar.progress((epoch + 1) / int(epochs))

            status_text.success(
                f"Training Complete! Best model saved to `models/best_model.pt` (Best Val Loss: {best_val_loss:.4f})"
            )

            # 5. Test Set Evaluation
            st.markdown("---")
            st.markdown("### Test Set Evaluation")
            with st.spinner("Evaluating on test set..."):
                model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
                test_loss, test_acc = validate(model, test_loader, criterion, device)
                st.metric(
                    label="Final Test Accuracy",
                    value=f"{test_acc:.2f}%",
                    delta=f"Loss: {test_loss:.4f}",
                    delta_color="inverse"
                )

            # 6. Confusion Matrix
            st.markdown("---")
            st.markdown("### Confusion Matrix")
            st.markdown(
                "Shows how often each true class was predicted as each other class. "
                "The diagonal (top-left to bottom-right) represents correct predictions. "
                "Off-diagonal cells reveal which digit pairs the model confuses."
            )
            with st.spinner("Computing confusion matrix on test set..."):
                y_true, y_pred = collect_predictions(model, test_loader, device)

                # Build class names for all labels present
                all_labels = sorted(set(y_true) | set(y_pred))
                class_names = []
                for lbl in all_labels:
                    class_names.append("Empty" if lbl == 0 else str(lbl))

                cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
                st.pyplot(cm_fig)

                # Per-class accuracy table
                cm_arr = confusion_matrix(y_true, y_pred, labels=all_labels)
                per_class_acc = cm_arr.diagonal() / cm_arr.sum(axis=1).clip(min=1) * 100
                acc_df = pd.DataFrame({
                    'Class': class_names,
                    'Correct': cm_arr.diagonal(),
                    'Total': cm_arr.sum(axis=1),
                    'Accuracy (%)': [f"{a:.1f}" for a in per_class_acc]
                })
                st.markdown("#### Per-Class Accuracy")
                st.dataframe(acc_df.set_index('Class'), use_container_width=True)

            # --- Save training text report ---
            train_report_path = save_training_report(
                history=history,
                test_loss=test_loss,
                test_acc=test_acc,
                y_true=y_true,
                y_pred=y_pred,
                dataset_mode=dataset_mode,
                epochs=int(epochs),
                learning_rate=learning_rate,
                batch_size=batch_size,
                best_val_loss=best_val_loss,
                model=model,
                output_path='models/training_report.txt',
            )
            st.success(f"📄 Training report saved to `{train_report_path}`")

            # Inline download button
            with open(train_report_path, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="⬇️ Download Training Report (.txt)",
                    data=f.read(),
                    file_name='training_report.txt',
                    mime='text/plain',
                )

        except Exception as e:
            import traceback
            st.error(f"Training Error: {e}")
            st.code(traceback.format_exc())