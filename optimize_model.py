import os
import time
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
