from typing import Tuple, Union
import torch
from torch import Tensor
from torch import nn


__allowed_activations__ = nn.modules.activation.__all__


def make_activation(name: str) -> nn.Module:
    if name in __allowed_activations__:
        return getattr(nn, name)()
    else:
        raise TypeError(f"Activation {name} not valid!")


class MaxPool2D(nn.Module):
    def __init__(
            self,
            stride: Union[int, tuple] | None,
            kernel_size: int | tuple = None,
            padding: Union[int, tuple, str] | None = 0,
            dilation:  Union[int, tuple] | None = 1
    ) -> None:
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class Flatten(nn.Module):
    """
    The -1 infers the size from the other dimension
    dim=0 is the batch dimension,
    so we are flattening the (N, C, H, W) tensor torch (N, C*H*W)

    Example:
        x = torch.randn((2, 3, 2, 2))
        print(x)
        print(x.view(x.size(dim=0), -1))
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class ChannelPool(nn.Module):
    """
    A PyTorch module that performs channel pooling on an input tensor.
    The input tensor should have shape `(N, C, H, W)`, where `N` is the batch size,
    `C` is the number of channels, `H` is the height, and `W` is the width.

    Channel pooling computes the maximum and average values over the channel dimension
    separately, and concatenates the results along the channel dimension to produce
    an output tensor of shape `(N, 2, H, W)`, i.e. a description of both max and mean
    features along channel dim.

    Args: None

    Returns:
        A PyTorch module that performs channel pooling on an input tensor.

    Example::

        # Create a ChannelPool module
        channel_pool = ChannelPool()

        # Apply channel pooling to an input tensor
        x = torch.randn(16, 64, 32, 32)
        y = channel_pool(x)  # y.shape == torch.Size([16, 2, 32, 32])
    """

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        # torch.max returns type torch.return_types.max: first tensor contains the max values
        # whereas the second tensor is an index tensor showing which input channel the max value occurred in
        return torch.cat((torch.max(x, dim=1)[0].unsqueeze(dim=1), torch.mean(x, dim=1).unsqueeze(dim=1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: Union[int, tuple]) -> None:
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.compress = ChannelPool()
        self.conv = ConvBlock2D(in_channels=2, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding='same', bn=False, activation="Sigmoid")

    def forward(self, x: Tensor) -> Tensor:
        sa = self.compress(x)
        sa = self.conv(sa)
        return torch.mul(x, sa)  # element-wise


class ConvBlock2D(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] | None = 1,
            padding: Union[int, str] | None = 0,
            groups: int | None = 1,
            bias: bool | None = True,
            bn: bool | None = True,
            activation: str | None = None
    ) -> None:
        super(ConvBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, groups=self.groups, bias=self.bias)
        self.add_module("conv", self.conv)
        if activation:
            self.add_module(f"{activation}", make_activation(activation))
        if bn:
            self.add_module("batch_norm", nn.BatchNorm2d(self.out_channels, momentum=0.1, affine=True))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, se_ratio: int) -> None:
        super().__init__()
        self.gmp = nn.AdaptiveMaxPool2d(output_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // se_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // se_ratio, in_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        gmp = self.gmp(x).squeeze(dim=-1).squeeze(dim=-1)
        gap = self.gap(x).squeeze(dim=-1).squeeze(dim=-1)
        mp = self.fc(gmp)
        ap = self.fc(gap)
        s = nn.Sigmoid()(torch.add(mp, ap))
        return x * s.unsqueeze(dim=-1).unsqueeze(dim=-1)  # .expand_as(x)


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            se_ratio: int,
            kernel_size: Union[int, tuple] | None = 4,
            spatial: bool | None = True,
            channel: bool | None = True
    ) -> None:
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.spatial = spatial
        self.channel_gate = ChannelAttention(in_channels=self.in_channels, se_ratio=self.se_ratio) if channel else None
        self.spatial_gate = SpatialAttention(self.kernel_size) if spatial else None

    def forward(self, x: Tensor) -> Tensor:
        cb = self.channel_gate(x) if self.channel_gate else x
        cb = self.spatial_gate(cb) if self.spatial_gate else cb
        return cb + x


class DenseBlock(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: str | None,
            dropout: bool | None = True
    ) -> None:
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation:
            self.add_module(f"{activation}", make_activation(activation))
        if dropout:
            self.add_module("dropout", nn.Dropout(p=0.5))
