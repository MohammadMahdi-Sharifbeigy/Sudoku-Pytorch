import os
import copy
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

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
from train import train_epoch, validate
from data_utils import get_dataloaders, get_dataloaders_mnist_hoda, get_dataloaders_all

# --- UI Configuration ---
st.set_page_config(
    page_title="AI Sudoku Solver", 
    page_icon="🧩", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stProgress .st-bo { background-color: #4CAF50; }
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
# MODE 1: INFERENCE (SOLVE SUDOKU)
# ==========================================
if app_mode == "Inference (Solve)":
    st.markdown("### Upload Sudoku Image")
    uploaded_file = st.file_uploader("Drag and drop your Sudoku image here", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and preprocess image
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
            
        # UI Layout: Two main columns for Input and Final Output
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(img, use_container_width=True)
            
        with st.spinner('Processing image and extracting grid...'):
            try:
                # --- Step-by-Step Vision Pipeline ---
                with st.expander("View Intermediate Processing Steps", expanded=False):
                    step_col1, step_col2, step_col3 = st.columns(3)
                    
                    # Step 1: Thresholding
                    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)
                    with step_col1:
                        st.markdown("**1. Adaptive Thresholding**")
                        st.image(thresh, use_container_width=True, channels="GRAY")
                    
                    # Step 2: Grid Extraction & Warp
                    cells, M, board_image = get_valid_cells_from_image(img)
                    with step_col2:
                        st.markdown("**2. Perspective Transform**")
                        st.image(board_image, use_container_width=True, channels="GRAY")
                    
                    # Step 3: Cell Extraction (Display a few valid cells as representation)
                    with step_col3:
                        st.markdown("**3. Cell Extraction**")
                        if 'plot_cell_images_in_grid' in globals():
                            fig = plot_cell_images_in_grid(cells)
                            st.pyplot(fig)
                        else:
                            st.info("81 individual cells extracted successfully.")

                # --- Prediction & Solving ---
                grid_array = get_predicted_sudoku_grid_torch(model, cells, device)
                solver = SudokuSolver(board=copy.deepcopy(grid_array))
                
                if solver.solve():
                    # Generate Final Annotated Image
                    final_image = generate_solution_image(
                        full_image=img, board_image=board_image, 
                        cells_list=cells, solved_board_arr=solver.board, M_matrix=M
                    )
                    
                    with col2:
                        st.markdown("#### Solved Sudoku")
                        st.image(final_image, use_container_width=True)
                        st.success("Sudoku solved successfully!")
                        
                    # Display Digital Matrix
                    st.markdown("### Digital Representation")
                    matrix_df = pd.DataFrame(solver.board)
                    # Styling the dataframe to look like a grid
                    st.dataframe(matrix_df.style.set_properties(**{'text-align': 'center', 'font-weight': 'bold'}), use_container_width=True)
                else:
                    st.error("The extracted grid is invalid or unsolvable. Please ensure the image is clear and well-lit.")
                    st.write("Extracted Grid Matrix:")
                    st.write(grid_array)
                    
            except Exception as e:
                st.error(f"Computer Vision Pipeline Error: {e}")

# ==========================================
# MODE 2: MODEL TRAINING (CNN)
# ==========================================
elif app_mode == "Model Training":
    st.markdown("### Model Training Dashboard")
    st.markdown("Configure hyperparameters and monitor the Convolutional Neural Network training process in real-time.")
    
    # Organize hyperparameters using columns
    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    with param_col2:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
    with param_col3:
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
        
    data_path = st.text_input("Dataset Directory Path", value="data")
    
    if st.button("Start Training Sequence", use_container_width=True):
        if not os.path.exists(data_path):
            st.error(f"Dataset path `{data_path}` does not exist. Please verify the path.")
            st.stop()
            
        st.info("Initializing DataLoaders. This may take a moment...")
        
        try:
            # 1. Load Data
            train_loader, val_loader, test_loader = get_dataloaders_mnist_hoda(data_path, batch_size=batch_size)
            
            # 2. Initialize Model, Loss, Optimizer
            model = DigitCNN(num_classes=10).to(device)
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            # 3. Setup UI placeholders for optimized live updating
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_table = st.empty() # Placeholder for a clean metrics display
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Loss Curve")
                loss_placeholder = st.empty() # Placeholder instead of direct chart
            with chart_col2:
                st.markdown("#### Accuracy Curve")
                acc_placeholder = st.empty() # Placeholder instead of direct chart
                
            best_val_loss = float('inf')
            os.makedirs('models', exist_ok=True)
            
            # Dictionary to track history locally
            history = {
                'Train Loss': [], 'Val Loss': [],
                'Train Acc': [], 'Val Acc': []
            }
            
            # 4. Training Loop
            for epoch in range(int(epochs)):
                status_text.markdown(f"**Running Epoch {epoch + 1}/{epochs}...**")
                
                # Execute training and validation
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'models/best_model.pt')
                
                # Store metrics in history
                history['Train Loss'].append(train_loss)
                history['Val Loss'].append(val_loss)
                history['Train Acc'].append(train_acc)
                history['Val Acc'].append(val_acc)
                
                # OPTIMIZED UI UPDATE: Redraw the chart using the history dictionary
                loss_df = pd.DataFrame({'Train Loss': history['Train Loss'], 'Val Loss': history['Val Loss']})
                acc_df = pd.DataFrame({'Train Acc': history['Train Acc'], 'Val Acc': history['Val Acc']})
                
                loss_placeholder.line_chart(loss_df)
                acc_placeholder.line_chart(acc_df)
                
                # Update text metrics cleanly
                metrics_table.markdown(f"""
                | Metric | Training | Validation |
                |---|---|---|
                | **Loss** | {train_loss:.4f} | {val_loss:.4f} |
                | **Accuracy** | {train_acc:.2f}% | {val_acc:.2f}% |
                """)
                
                # Update Progress
                progress_bar.progress((epoch + 1) / int(epochs))
                
            status_text.success(f"Training Complete! Best model saved to `models/best_model.pt` (Best Val Loss: {best_val_loss:.4f})")
            
            # Final Test Evaluation
            st.markdown("---")
            st.markdown("### Test Set Evaluation")
            with st.spinner("Evaluating on test set..."):
                model.load_state_dict(torch.load('models/best_model.pt'))
                test_loss, test_acc = validate(model, test_loader, criterion, device)
                st.metric(label="Final Test Accuracy", value=f"{test_acc:.2f}%", delta=f"Loss: {test_loss:.4f}", delta_color="inverse")
            
        except Exception as e:
            st.error(f"Training Error: {e}")