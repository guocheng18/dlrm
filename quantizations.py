"""
FP16 and FP8 quantization methods for quantization-aware training.
"""

import torch
from torch.autograd import Function


class FP8QuantDequantSTE(Function):
    @staticmethod
    def forward(ctx, x):
        M = torch.tensor([3.0], device=x.device)
        E = torch.tensor([4.0], device=x.device)
        maxval = torch.tensor([240.0], device=x.device)

        bias = 2**E - torch.log2(maxval) + torch.log2(2 - 2 ** (-M)) - 1

        minval = -maxval
        xc = torch.min(torch.max(x, minval), maxval)

        log_scales = torch.clamp(
            (torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.0)

        scales = 2.0 ** (log_scales - M - bias)

        result = torch.round(xc / scales) * scales
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FP16QuantDequantSTE(Function):
    @staticmethod
    def forward(ctx, x):
        """This assume the original model parameters are all in FP16, but doing so, the parameters only lose precisions"""
        return x.half().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


if __name__ == "__main__":
    x = torch.tensor(0, dtype=torch.float32)
    print(FP8QuantDequantSTE.apply(x))
    print(FP16QuantDequantSTE.apply(x))
