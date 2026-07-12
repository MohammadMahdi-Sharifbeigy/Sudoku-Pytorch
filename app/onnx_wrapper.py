"""ONNX and model-wrapper helpers for inference."""
import numpy as np
import torch
import torch.nn as nn


class OnnxInferenceSession:
    """Wraps onnxruntime to behave like a PyTorch model.

    __call__ returns digit logits (first output) as torch.Tensor.
    Call .infer_multitask() for both heads on multi-task models.
    Runs on CPU regardless of device argument.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.sess      = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.n_outputs = len(self.sess.get_outputs())
        self.onnx_path = onnx_path

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.sess.run(None, {'input': x.cpu().numpy()})
        return torch.from_numpy(outputs[0])

    def infer_multitask(self, x: torch.Tensor):
        outputs = self.sess.run(None, {'input': x.cpu().numpy()})
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])

    def eval(self):
        return self

    def to(self, _device):
        return self


class DigitOnlyModelWrapper:
    """Wraps MultiTaskDigitCNN to return only digit logits."""

    def __init__(self, model: nn.Module):
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        digit_logits, _ = self._model(x)
        return digit_logits

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model.to(device)
        return self


class UnifiedDigitOnlyWrapper:
    """Wraps UnifiedDigitCNN (20-class) → digit logits (0–9).

    Folds English + Persian logits per digit:
      logit[0] = max(class_0, class_10)
      logit[d] = class_d + class_{d+10}  for d in 1..9
    """

    def __init__(self, model: nn.Module):
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logits20 = self._model(x)
        logits10 = logits20[:, :10].clone()
        logits10[:, 0]  = torch.maximum(logits20[:, 0], logits20[:, 10])
        logits10[:, 1:] = logits20[:, 1:10] + logits20[:, 11:20]
        return logits10

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model.to(device)
        return self
