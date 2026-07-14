"""Mode 3 — Model Optimization & Benchmarking."""
import glob
import os

import pandas as pd
import streamlit as st
import torch

from src.model import (
    DigitCNN, MultiTaskDigitCNN, UnifiedDigitCNN, UNIFIED_NUM_CLASSES,
    EfficientNetDigitCNN, load_digit_cnn_checkpoint, probe_digit_cnn_architecture,
)
from src.optimize_model import run_optimization_and_benchmark, run_optimization_and_benchmark_multitask
from app.cancel import RunCancelled, clear_cancel, raise_if_cancelled, render_stop_button


def render_optimization_page(device: torch.device) -> None:
    st.markdown("### Model Optimization & Benchmarking")
    st.markdown("Export a trained model to TorchScript + ONNX.")
    render_stop_button("optimization", "Stop optimization")

    cpu_device = torch.device('cpu')

    single_candidates = [
        ("English / Default", "models/best_model.pt"),
        ("English Only",       "models/best_model_english.pt"),
        ("Persian Only",       "models/best_model_persian.pt"),
    ]

    candidates = {}
    for base_label, path in single_candidates:
        if os.path.exists(path):
            arch = probe_digit_cnn_architecture(path)
            arch_tag = "DigitCNN" if arch == "DigitCNN" else "Legacy 2-conv"
            candidates[f"{base_label} ({arch_tag})"] = (path, "single")

    candidates["Multi-Task CNN"] = ("models/best_model_multitask.pt", "multi")
    candidates["Unified 20-Class (MobileNet/Shuffle)"] = ("models/best_model_unified20.pt", "unified")

    # EfficientNet checkpoints are saved per-purpose (e.g. best_model_efficientnet_english.pt),
    # so discover whatever exists rather than hardcoding a single filename.
    for eff_path in sorted(glob.glob("models/best_model_efficientnet_*.pt")):
        purpose_tag = (
            os.path.basename(eff_path)
            .replace("best_model_efficientnet_", "")
            .replace(".pt", "")
        )
        candidates[f"EfficientNet-B0 ({purpose_tag.capitalize()})"] = (eff_path, "efficientnet")

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
            model, arch_used = load_digit_cnn_checkpoint(opt_pt_path, cpu_device, num_classes=10)
            if arch_used == 'LegacyDigitCNN':
                st.info(
                    "This checkpoint was saved with the old 2-conv architecture "
                    "(pre-BatchNorm redesign). Loaded via `LegacyDigitCNN` — "
                    "exports below still work, but consider retraining with the "
                    "current `DigitCNN` for better accuracy."
                )
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

        elif opt_type == "efficientnet":
            model = EfficientNetDigitCNN(num_classes=10, pretrained=False)
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