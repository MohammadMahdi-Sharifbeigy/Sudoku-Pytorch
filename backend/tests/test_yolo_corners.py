import numpy as np
import pytest
import torch

from app.core.vision.yolo import get_grid_corners_yolo, _best_keypoints


class _KP:
    def __init__(self, xy):
        # mimic ultralytics: keypoints.xy is a torch tensor, shape (1, 4, 2)
        self.xy = torch.tensor([xy], dtype=torch.float32)


class _Boxes:
    def __init__(self, conf):
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _Result:
    def __init__(self, kpts_xy, confs):
        self.keypoints = _KP(kpts_xy)
        self.boxes = _Boxes(confs)


class _FakePose:
    def __init__(self, result):
        self._result = result

    def predict(self, img, conf=0.25, verbose=False, **kw):
        return [self._result]


def test_returns_ordered_corners():
    # one instance, 4 corners shuffled (BR, TL, BL, TR)
    xy = [(340, 340), (60, 60), (60, 340), (340, 60)]
    fake = _FakePose(_Result(xy, [0.9]))
    img = np.zeros((400, 400, 3), np.uint8)
    corners = get_grid_corners_yolo(img, fake)
    assert corners.shape == (4, 2)
    tl, tr, br, bl = corners
    assert tuple(tl) == (60, 60)
    assert tuple(br) == (340, 340)


def test_empty_raises():
    class _Empty(_FakePose):
        def predict(self, img, conf=0.25, verbose=False, **kw):
            return []
    img = np.zeros((400, 400, 3), np.uint8)
    try:
        get_grid_corners_yolo(img, _Empty(None))
        assert False
    except ValueError:
        pass


def test_empty_keypoints_tensor_raises():
    # realistic empty detection: xy has shape (0, 4, 2)
    class _EmptyKP:
        xy = torch.zeros((0, 4, 2), dtype=torch.float32)

    class _R:
        keypoints = _EmptyKP()
        boxes = None

    with pytest.raises(ValueError):
        _best_keypoints(_R())


def test_none_keypoints_raises():
    class _R:
        keypoints = None
        boxes = None

    with pytest.raises(ValueError):
        _best_keypoints(_R())

    fake = _FakePose(_R())
    img = np.zeros((400, 400, 3), np.uint8)
    with pytest.raises(ValueError):
        get_grid_corners_yolo(img, fake)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_tensors_do_not_crash():
    xy = [(340, 340), (60, 60), (60, 340), (340, 60)]

    class _CudaKP:
        def __init__(self, xy):
            self.xy = torch.tensor([xy], dtype=torch.float32).cuda()

    class _CudaBoxes:
        def __init__(self, conf):
            self.conf = torch.tensor(conf, dtype=torch.float32).cuda()

    class _CudaResult:
        def __init__(self, kpts_xy, confs):
            self.keypoints = _CudaKP(kpts_xy)
            self.boxes = _CudaBoxes(confs)

    fake = _FakePose(_CudaResult(xy, [0.9]))
    img = np.zeros((400, 400, 3), np.uint8)
    corners = get_grid_corners_yolo(img, fake)
    assert corners.shape == (4, 2)
