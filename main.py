import streamlit as st
import torch

from app.inference_page import render_inference_page
from app.training_page import render_training_page
from app.optimization_page import render_optimization_page

st.set_page_config(
    page_title="AI Sudoku Solver",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    .stProgress .st-bo { background-color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

st.title("Intelligent Sudoku Solver (PyTorch + OpenCV)")
st.markdown(
    "End-to-end computer vision and deep learning pipeline "
    "for detecting and solving Sudoku puzzles."
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["Inference (Solve)", "Model Training", "Model Optimization"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Compute Device:** `{device}`")

if app_mode == "Inference (Solve)":
    render_inference_page(device)
elif app_mode == "Model Training":
    render_training_page(device)
elif app_mode == "Model Optimization":
    render_optimization_page(device)
