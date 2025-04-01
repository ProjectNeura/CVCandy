from typing import override as _override

from torch import Tensor as _Tensor, bool as _bool, float32 as _float32, ones as _ones, to_dlpack as _to_dlpack
from torch.nn.functional import conv2d as _conv2d

from cvcandy.universal import Metric as _Metric

try:
    from cupy import from_dlpack as _dlpack2np
    from cupyx.scipy.ndimage import distance_transform_edt as _distance_transform_edt
except ImportError:
    from numpy import from_dlpack as _dlpack2np
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt


class DiceSimilarityCoefficient(_Metric):
    @_override
    def compute(self, mask: _Tensor, label: _Tensor) -> float:
        _Metric._args_check(mask, label, _bool)
        volume_sum = label.sum() + mask.sum()
        if volume_sum == 0:
            return 0
        return float(2 * (mask & label).sum() / volume_sum)


class NormalizedSurfaceDistance(_Metric):
    @_override
    def compute(self, mask: _Tensor, label: _Tensor) -> float:
        _, device = _Metric._args_check(mask, label, _bool)
        kernel = _ones((1, 1, 3, 3), dtype=_float32, device=device)
        mask_neighbors = _conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=1)
        label_neighbors = _conv2d(label.unsqueeze(0).unsqueeze(0).float(), kernel, padding=1)
        mask_boundary = mask & (mask_neighbors.squeeze() < 9)
        label_boundary = label & (label_neighbors.squeeze() < 9)
        mask_boundary_np = _dlpack2np(_to_dlpack(mask_boundary)).astype(bool)
        label_boundary_np = _dlpack2np(_to_dlpack(label_boundary)).astype(bool)
        dist_to_mask = _distance_transform_edt(~mask_boundary_np)
        dist_to_label = _distance_transform_edt(~label_boundary_np)
        mask_to_label_dist = dist_to_label[label_boundary_np]
        label_to_mask_dist = dist_to_mask[mask_boundary_np]
        d1 = 0 if mask_to_label_dist.size < 1 else mask_to_label_dist.mean()
        d2 = 0 if label_to_mask_dist.size < 1 else label_to_mask_dist.mean()
        return .5 * float(d1 + d2)
