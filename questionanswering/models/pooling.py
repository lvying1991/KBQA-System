import torch
from torch import nn as nn
from torch import autograd


class LogSumExpPooling1d(nn.Module):
    """Applies a 1D LogSumExp pooling over an input signal composed of several input planes.
    LogSumExp is a smooth approximation of the max function.
    在由多个输入平面组成的输入信号上应用1D LogSumExp池。
    LogSumExp是max函数的平滑近似值。

    Examples:
    >>> m = LogSumExpPooling1d()
    >>> input = autograd.Variable(torch.randn(4, 5, 10))
    >>> m(input).squeeze()
    """

    def __init__(self):
        super(LogSumExpPooling1d, self).__init__()

    def forward(self, x):
        x.exp_()
        x = x.sum(dim=-1, keepdim=True)
        x.log_()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
