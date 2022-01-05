import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, attention=True):
        super(Model, self).__init__()
        self.attention = attention
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dense_1 = nn.Linear(in_features=15488, out_features=100)
        self.dense_2 = nn.Linear(in_features=100, out_features=9)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.layer_1 = self.basic_conv2d(in_planes=1, out_planes=32, kernel_size=10, bn=True, relu=True)
        self.layer_2 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=10, bn=True, relu=True)
        self.layer_3 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=8, bn=True, relu=True)
        self.layer_4 = self.basic_conv2d(in_planes=32, out_planes=32, kernel_size=4, bn=True, relu=True)
        self.block_1 = nn.Sequential(self.layer_1, self.layer_2, self.maxpool1)
        self.block_2 = nn.Sequential(self.layer_3, self.layer_4, self.maxpool2)
        self.second_last = nn.Sequential(self.dense_1, self.relu, self.dropout)
        self.last = nn.Sequential(self.dense_2, self.relu, self.softmax)
        #self.dummy_param = nn.Parameter(torch.empty(0))  # to tell CAM what the device is

    def forward(self, t):

        t = self.block_1(t)
        if self.attention:
            t = self.channel_att(t, 32, 4)

        t = self.block_2(t)
        if self.attention:
            t = self.channel_att(t, 32, 4)

        t = self._flatten(t)
        t = self.second_last(t)
        t = self.last(t)
        return t

    def channel_att(self, input_tensor, channels_in, reduction_factor):
        sequence = nn.Sequential(
            nn.Linear(channels_in, channels_in // reduction_factor),
            nn.ReLU(),
            nn.Linear(channels_in // reduction_factor, channels_in),
            nn.ReLU())
        #device = self.dummy_param.device
        #sequence.to(device)
        sequence = sequence.to('cuda:0') 
        global_mp = self._global_max_pool(input_tensor)
        global_ap = self._global_av_pool(input_tensor)
        global_mp = sequence(global_mp)
        global_ap = sequence(global_ap)
        s = torch.add(global_mp, global_ap)
        s = nn.Sigmoid()(s).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(input_tensor)
        output = torch.mul(input_tensor, s)
        return output

    @staticmethod
    def basic_conv2d(in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=True):
        layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride, stride),
                      padding=padding, bias=bias)
        )
        if relu is not None:
            layer = nn.Sequential(
                layer,
                nn.ReLU()
            )
        if bn is not None:
            layer = nn.Sequential(
                layer,
                nn.BatchNorm2d(out_planes, momentum=0.1, affine=True)
            )
        return layer

    @staticmethod
    def _global_max_pool(data):
        """
            Take in a batch tensor (batch_size, channels, height, width)
            return a tensor of shape (batch_size, channels)
        """
        max_pooled = nn.AdaptiveMaxPool2d(output_size=1)(data)
        max_pooled = max_pooled.squeeze(dim=3).squeeze(dim=2)
        return max_pooled

    @staticmethod
    def _global_av_pool(data):
        """
            see _global_max_pool
        """
        av_pooled = nn.AdaptiveAvgPool2d(output_size=1)(data)
        av_pooled = av_pooled.squeeze(dim=3).squeeze(dim=2)
        return av_pooled

    @staticmethod
    def _flatten(tensor):
        return tensor.view(tensor.size(0), -1)
