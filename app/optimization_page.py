"""Mode 3 — Model Optimization & Benchmarking."""
import os

import pandas as pd
import streamlit as st
import torch

from src.model import (
    DigitCNN, MultiTaskDigitCNN, UnifiedDigitCNN, UNIFIED_NUM_CLASSES,
)
from src.optimize_model import run_optimization_and_benchmark, run_optimization_and_benchmark_multitask
from app.cancel import RunCancelled, clear_cancel, raise_if_cancelled, render_stop_button


def render_optimization_page(device: torch.device) -> None:
    st.markdown("### Model Optimization & Benchmarking")
    st.markdown("Export a trained model to TorchScript + ONNX.")
    render_stop_button("optimization", "Stop optimization")

    cpu_device = torch.device('cpu')

    candidates = {
        "English / Default (DigitCNN)":        ("models/best_model.pt",           "single"),
        "English Only (DigitCNN)":              ("models/best_model_english.pt",   "single"),
        "Persian Only (DigitCNN)":              ("models/best_model_persian.pt",   "single"),
        "Multi-Task CNN":                       ("models/best_model_multitask.pt", "multi"),
        "Unified 20-Class (MobileNet/Shuffle)": ("models/best_model_unified20.pt", "unified"),
    }
    available = {k: v for k, v in candidates.items() if os.path.exists(v[0])}

    if not available:
        st.error("No trained models found. Train at least one model first.")
        return

    opt_choice         = st.selectbox("Model to optimize:", list(available.keys()))
    opt_pt_path, opt_type = available[opt_choice]
    opt_prefix         = opt_pt_path.replace('.pt', '')

    st.info(f"Source: `{opt_pt_path}` → `{opt_prefix}.ts` and `{opt_prefix}.onnx`")

    if opt_type == "unified":
        opt_bb = st.selectbox("Backbone (must match training)",
                              ["mobilenet_v3_small", "shufflenet_v2_x0_5"])
    else:
        opt_bb = "mobilenet_v3_small"

    if not st.button("Run Optimization & Benchmark"):
        return

    clear_cancel("optimization")
    try:
        with st.spinner("Loading model and exporting…"):
            raise_if_cancelled("optimization")

        if opt_type == "single":
            model = DigitCNN(num_classes=10)
            model.load_state_dict(torch.load(opt_pt_path, map_location=cpu_device))
            model.to(cpu_device).eval()
            results = run_optimization_and_benchmark(
                model, opt_pt_path, cpu_device,
                output_prefix=opt_prefix,
                should_stop=lambda: raise_if_cancelled("optimization"),
            )
            onnx_ms = results[-1].get("ms/cell", "N/A")
            if onnx_ms == "N/A":
                st.warning("onnxruntime not installed — ONNX benchmark skipped.")
            else:
                st.success("Export complete. ONNX verified.")
            st.dataframe(pd.DataFrame(results), width='stretch')
            st.markdown(
                f"**TorchScript**: `torch.jit.load('{opt_prefix}.ts')`  \n"
                f"**ONNX**: `onnxruntime.InferenceSession('{opt_prefix}.onnx')`"
            )

        else:
            if opt_type == "multi":
                model = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2)
            else:
                model = UnifiedDigitCNN(backbone=opt_bb, pretrained=False,
                                        num_classes=UNIFIED_NUM_CLASSES)
            model.load_state_dict(torch.load(opt_pt_path, map_location=cpu_device))
            model.to(cpu_device).eval()
            results, verif = run_optimization_and_benchmark_multitask(
                model, opt_pt_path, cpu_device,
                output_prefix=opt_prefix,
                should_stop=lambda: raise_if_cancelled("optimization"),
            )
            if verif.get('error'):
                st.warning("onnxruntime not installed — ONNX benchmark skipped.")
            elif not verif['digit_match'] or not verif['lang_match']:
                st.error(f"ONNX head mismatch: digit={verif['digit_match']} lang={verif['lang_match']}")
            else:
                st.success("ONNX verified: both heads match PyTorch.")
            st.dataframe(pd.DataFrame(results), width='stretch')
            st.markdown(
                f"**TorchScript**: `torch.jit.load('{opt_prefix}.ts')`  \n"
                f"**ONNX**: `onnxruntime.InferenceSession('{opt_prefix}.onnx')` "
                f"— outputs: `digit_output`, `lang_output`"
            )

    except RunCancelled as e:
        st.warning(str(e))
        clear_cancel("optimization")
