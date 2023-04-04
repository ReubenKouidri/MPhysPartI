from models.components import ConvBlock2D, MaxPool2D, CBAM, DenseBlock, Flatten
import torch.nn as nn
from collections import OrderedDict


class S_2RB2D2(nn.Sequential):
    """sequential, 2x RB(2 conv each before attention) 2x Dense"""
    architecture = OrderedDict([
        ("ConvBlock1", ConvBlock2D(1, 32, 10, bn=True, activation="ReLU")),
        ("ConvBlock2", ConvBlock2D(32, 32, 10, bn=True, activation="ReLU")),
        ("MP1", MaxPool2D(2, 2)),
        ("CBAM1", CBAM(32, se_ratio=4, kernel_size=4)),
        ("ConvBlock3", ConvBlock2D(32, 32, 8, bn=True, activation="ReLU")),
        ("ConvBlock4", ConvBlock2D(32, 32, 4, bn=True, activation="ReLU")),
        ("MP2", MaxPool2D(2, 2)),
        ("CBAM2", CBAM(32, se_ratio=4, kernel_size=4)),
        ("Flatten", Flatten()),
        ("DenseBlock1", DenseBlock(15488, 100, activation="ReLU")),
        ("DenseBlock2", DenseBlock(100, 9, activation="ReLU", dropout=False)),
        ("Softmax", nn.Softmax(dim=1))
    ])

    def __init__(self):
        super(S_2RB2D2, self).__init__(S_2RB2D2.architecture)
