import os
import time
import torch
import torch.nn as nn
from typing import Any

from app.core.model_files import artifact_path_for_model


def run_optimization_and_benchmark(
    base_model: nn.Module,
    model_path: str,
    cpu_device: torch.device,
    models_dir: str = 'models',
) -> list[dict[str, Any]]:
    """
    Converts a PyTorch model to TorchScript and ONNX, and benchmarks CPU inference.
    Returns a list of dicts containing benchmark results.
    """
    results: list[dict[str, Any]] = []
    dummy_input = torch.randn(1, 1, 28, 28).to(cpu_device)

    # 1. PyTorch Benchmark
    pt_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    for _ in range(10):
        _ = base_model(dummy_input)

    iters = 1000
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = base_model(dummy_input)
    pt_time = (time.perf_counter() - start_time) / iters * 1000

    results.append({
        "format": "PyTorch (.pt)",
        "size_mb": round(pt_size_mb, 3),
        "latency_ms": round(pt_time, 3),
        "path": model_path,
    })

    # 2. TorchScript Export & Benchmark
    os.makedirs(models_dir, exist_ok=True)
    ts_path = artifact_path_for_model(model_path, ".ts")
    traced_model = torch.jit.trace(base_model, dummy_input)
    traced_model.save(ts_path)
    ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)

    for _ in range(10):
        _ = traced_model(dummy_input)

    start_time = time.perf_counter()
    for _ in range(iters):
        _ = traced_model(dummy_input)
    ts_time = (time.perf_counter() - start_time) / iters * 1000

    results.append({
        "format": "TorchScript (.ts)",
        "size_mb": round(ts_size_mb, 3),
        "latency_ms": round(ts_time, 3),
        "path": ts_path,
    })

    # 3. ONNX Export & Benchmark
    onnx_path = artifact_path_for_model(model_path, ".onnx")
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

        def to_numpy(tensor: torch.Tensor) -> Any:
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}

        for _ in range(10):
            _ = ort_session.run(None, ort_inputs)

        start_time = time.perf_counter()
        for _ in range(iters):
            _ = ort_session.run(None, ort_inputs)
        onnx_time = (time.perf_counter() - start_time) / iters * 1000

        results.append({
            "format": "ONNX (.onnx)",
            "size_mb": round(onnx_size_mb, 3),
            "latency_ms": round(onnx_time, 3),
            "path": onnx_path,
        })

    except ImportError:
        results.append({
            "format": "ONNX (.onnx)",
            "size_mb": round(onnx_size_mb, 3),
            "latency_ms": None,
            "path": onnx_path,
        })

    return results
