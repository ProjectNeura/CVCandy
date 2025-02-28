from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

from torch import Tensor as _Tensor, dtype as _dtype, device as _device


class Metric(object, metaclass=_ABCMeta):
    @staticmethod
    def _args_check(mask: _Tensor, label: _Tensor, dtype: _dtype | None = None, device: _device | None = None) -> tuple[
        _dtype, _device]:
        if mask.shape != label.shape:
            raise ValueError(f"Mask ({mask.shape}) and label ({label.shape}) must have the same shape")
        if (mask_dtype := mask.dtype) != label.dtype:
            raise TypeError(f"Mask ({mask.dtype}) and label ({label.dtype}) must be the same type")
        if dtype and mask_dtype != dtype:
            raise TypeError(f"Tensors are expected to be {dtype} type, but instead they are {mask.dtype} type")
        if (mask_device := mask.device) != label.device:
            raise RuntimeError(f"Mask ({mask.device}) and label ({label.device}) must be on the same device")
        if device and mask_device != device:
            raise RuntimeError(f"Tensors are expected to be on {device}, but instead they are on {mask.device}")
        return mask_dtype, mask_device

    @_abstractmethod
    def compute(self, mask: _Tensor, label: _Tensor) -> _Tensor:
        raise NotImplementedError

    def __call__(self, mask: _Tensor, label: _Tensor) -> _Tensor:
        return self.compute(mask, label)
