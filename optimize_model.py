import os
import time
import numpy as np
import torch
import pandas as pd


def run_optimization_and_benchmark(base_model, model_path, cpu_device):
    """
    Converts a PyTorch model to TorchScript and ONNX, and benchmarks CPU inference.
    Returns a list of dictionaries containing the benchmark results.
    """
    results = []
    # Sudoku digit input size is 1x1x28x28
    dummy_input = torch.randn(1, 1, 28, 28).to(cpu_device)
    
    # 1. PyTorch Benchmark
    pt_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Warmup
    for _ in range(10):
        _ = base_model(dummy_input)
    
    start_time = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        _ = base_model(dummy_input)
    pt_time = (time.perf_counter() - start_time) / iters * 1000 # in ms
    
    results.append({"Format": "PyTorch (.pt)", "Size (MB)": f"{pt_size_mb:.3f}", "Inference Latency (ms)": f"{pt_time:.3f}"})
    
    # 2. TorchScript Export & Benchmark
    ts_path = 'models/best_model.ts'
    traced_model = torch.jit.trace(base_model, dummy_input)
    traced_model.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    
    # Warmup
    for _ in range(10):
        _ = traced_model(dummy_input)
    
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = traced_model(dummy_input)
    ts_time = (time.perf_counter() - start_time) / iters * 1000
    
    results.append({"Format": "TorchScript (.ts)", "Size (MB)": f"{ts_size_mb:.3f}", "Inference Latency (ms)": f"{ts_time:.3f}"})
    
    # 3. ONNX Export & Benchmark
    onnx_path = 'models/best_model.onnx'
    torch.onnx.export(
        base_model, dummy_input, onnx_path, 
        export_params=True, opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, ort_inputs)
        
        start_time = time.perf_counter()
        for _ in range(iters):
            _ = ort_session.run(None, ort_inputs)
        onnx_time = (time.perf_counter() - start_time) / iters * 1000
        
        results.append({"Format": "ONNX (.onnx)", "Size (MB)": f"{onnx_size_mb:.3f}", "Inference Latency (ms)": f"{onnx_time:.3f}"})
        
    except ImportError:
        results.append({"Format": "ONNX (.onnx)", "Size (MB)": f"{onnx_size_mb:.3f}", "Inference Latency (ms)": "N/A"})

    return results


def run_optimization_and_benchmark_multitask(base_model, model_path, cpu_device):
    """TorchScript + ONNX export for MultiTaskDigitCNN.
    ONNX uses output_names=['digit_output','lang_output'].
    Verifies ONNX matches PyTorch on both heads before benchmarking.
    Returns (results_list, verification_dict).
    """
    results = []
    dummy   = torch.randn(1, 1, 28, 28).to(cpu_device)
    iters   = 1000

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
    ts_path = 'models/best_model_multitask.ts'
    traced  = torch.jit.trace(base_model, dummy)
    traced.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    for _ in range(10):
        traced(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        traced(dummy)
    ts_ms = (time.perf_counter() - t0) / iters * 1000
    results.append({"Format": "TorchScript (.ts)",
                    "Size (MB)": f"{ts_size_mb:.3f}",
                    "ms/cell": f"{ts_ms:.3f}",
                    "ms/81-cells": f"{ts_ms * 81:.1f}"})

    # 3. ONNX — dual outputs
    onnx_path = 'models/best_model_multitask.onnx'
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
        results.append({"Format": "ONNX (.onnx)",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": f"{onnx_ms:.3f}",
                        "ms/81-cells": f"{onnx_ms * 81:.1f}"})
    except ImportError:
        results.append({"Format": "ONNX (.onnx)",
                        "Size (MB)": f"{onnx_size_mb:.3f}",
                        "ms/cell": "N/A",
                        "ms/81-cells": "N/A"})
        verification["error"] = "onnxruntime not installed"

    return results, verification
