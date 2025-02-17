import torch
from torch import Tensor

class LogTransform(torch.nn.Module):
    """Log-transform a tensor image.
    This transform does not support PIL Image.
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:

    """

    def __init__(self, m2mm=False, hour2day=False):
        super().__init__()
        self.m2mm = m2mm
        self.hour2day = hour2day

    def precip_transform(x, mm_per_day:bool=False):
        if mm_per_day:
            x = x*24*1000 # mm / day # era5 is needed. daymet is no
        return torch.log1p(x)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        alpha = 1
        if self.m2mm:
            alpha *= 1000
        if self.hour2day:
            alpha *= 24
        
        return torch.log1p(alpha * tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(M2mm={self.m2mm}, Hour2day={self.hour2day})"