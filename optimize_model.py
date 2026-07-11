import os
import time
import numpy as np
import torch
import pandas as pd


def run_optimization_and_benchmark(base_model, model_path, cpu_device,
                                    output_prefix='models/best_model'):
    """
    Export single-output model to TorchScript + ONNX and benchmark CPU inference.
    Artifacts written to {output_prefix}.ts and {output_prefix}.onnx.
    """
    results    = []
    dummy      = torch.randn(1, 1, 28, 28).to(cpu_device)
    iters      = 1000
    ts_path    = f'{output_prefix}.ts'
    onnx_path  = f'{output_prefix}.onnx'

    # 1. PyTorch
    pt_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    for _ in range(10):
        base_model(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        base_model(dummy)
    pt_ms = (time.perf_counter() - t0) / iters * 1000
    results.append({"Format": "PyTorch (.pt)", "Size (MB)": f"{pt_size_mb:.3f}",
                    "ms/cell": f"{pt_ms:.3f}", "ms/81-cells": f"{pt_ms*81:.1f}"})

    # 2. TorchScript
    traced = torch.jit.trace(base_model, dummy)
    traced.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    for _ in range(10):
        traced(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        traced(dummy)
    ts_ms = (time.perf_counter() - t0) / iters * 1000
    results.append({"Format": f"TorchScript ({os.path.basename(ts_path)})",
                    "Size (MB)": f"{ts_size_mb:.3f}",
                    "ms/cell": f"{ts_ms:.3f}", "ms/81-cells": f"{ts_ms*81:.1f}"})

    # 3. ONNX
    torch.onnx.export(
        base_model, dummy, onnx_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    try:
        import onnxruntime as ort
        sess      = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        np_in     = dummy.cpu().numpy()
        for _ in range(10):
            sess.run(None, {'input': np_in})
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, {'input': np_in})
        onnx_ms = (time.perf_counter() - t0) / iters * 1000
        results.append({"Format": f"ONNX ({os.path.basename(onnx_path)})",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": f"{onnx_ms:.3f}", "ms/81-cells": f"{onnx_ms*81:.1f}"})
    except ImportError:
        results.append({"Format": f"ONNX ({os.path.basename(onnx_path)})",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": "N/A", "ms/81-cells": "N/A"})

    return results


def run_optimization_and_benchmark_multitask(base_model, model_path, cpu_device,
                                              output_prefix='models/best_model_multitask'):
    """TorchScript + ONNX export for dual-output models (MultiTaskDigitCNN or UnifiedDigitCNN).
    ONNX uses output_names=['digit_output','lang_output'].
    Verifies ONNX matches PyTorch on both heads before benchmarking.
    Artifacts written to {output_prefix}.ts and {output_prefix}.onnx.
    Returns (results_list, verification_dict).
    """
    results   = []
    dummy     = torch.randn(1, 1, 28, 28).to(cpu_device)
    iters     = 1000
    ts_path   = f'{output_prefix}.ts'
    onnx_path = f'{output_prefix}.onnx'

    # 1. PyTorch
    pt_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    for _ in range(10):
        base_model(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        base_model(dummy)
    pt_ms = (time.perf_counter() - t0) / iters * 1000
    results.append({"Format": "PyTorch (.pt)",
                    "Size (MB)": f"{pt_size_mb:.3f}",
                    "ms/cell": f"{pt_ms:.3f}",
                    "ms/81-cells": f"{pt_ms * 81:.1f}"})

    # 2. TorchScript
    traced  = torch.jit.trace(base_model, dummy)
    traced.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    for _ in range(10):
        traced(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        traced(dummy)
    ts_ms = (time.perf_counter() - t0) / iters * 1000
    results.append({"Format": f"TorchScript ({os.path.basename(ts_path)})",
                    "Size (MB)": f"{ts_size_mb:.3f}",
                    "ms/cell": f"{ts_ms:.3f}",
                    "ms/81-cells": f"{ts_ms * 81:.1f}"})

    # 3. ONNX — dual outputs
    torch.onnx.export(
        base_model, dummy, onnx_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['digit_output', 'lang_output'],
        dynamic_axes={
            'input':        {0: 'batch_size'},
            'digit_output': {0: 'batch_size'},
            'lang_output':  {0: 'batch_size'},
        },
    )
    onnx_size_mb  = os.path.getsize(onnx_path) / (1024 * 1024)
    verification  = {"digit_match": False, "lang_match": False, "error": None}

    try:
        import onnxruntime as ort
        sess      = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        np_input  = dummy.cpu().numpy()
        ort_out   = sess.run(None, {'input': np_input})
        pt_d, pt_l = base_model(dummy)
        digit_ok  = np.allclose(ort_out[0], pt_d.detach().cpu().numpy(), atol=1e-5)
        lang_ok   = np.allclose(ort_out[1], pt_l.detach().cpu().numpy(), atol=1e-5)
        verification.update({"digit_match": digit_ok, "lang_match": lang_ok})

        for _ in range(10):
            sess.run(None, {'input': np_input})
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, {'input': np_input})
        onnx_ms = (time.perf_counter() - t0) / iters * 1000
        results.append({"Format": f"ONNX ({os.path.basename(onnx_path)})",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": f"{onnx_ms:.3f}",
                        "ms/81-cells": f"{onnx_ms * 81:.1f}"})
    except ImportError:
        results.append({"Format": f"ONNX ({os.path.basename(onnx_path)})",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": "N/A",
                        "ms/81-cells": "N/A"})
        verification["error"] = "onnxruntime not installed"

    return results, verification
