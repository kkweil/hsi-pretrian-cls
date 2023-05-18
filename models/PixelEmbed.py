import math
import logging
from functools import partial
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers.helpers import to_2tuple


class PosCNN1D(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN1D, self).__init__()
        self.proj = nn.Sequential(nn.Conv1d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, int(N ** 0.5), int(N ** 0.5))
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, int(N ** 0.5), int(N ** 0.5))
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class PixelEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_channels=204, embed_dim=None,
                 kernel_num_3d=30, kernel_size_3d=(10, 1, 1), stride_3d=(5, 1, 1)):
        super().__init__()
        self.depth = kernel_num_3d * ((embed_dim - kernel_size_3d[0]) // stride_3d[0] + 1)
        # NOTICE: conv3d input shape: (B,C,D,W,H)  D means depth of image, like time series image etc.
        #         Here depth D means the channels of hsi, thus the channel of conv3d is set to 1.
        self.conv2d_1 = nn.Conv2d(in_channels, embed_dim, padding='valid',
                                  kernel_size=(1, 1), stride=(1, 1))

        self.conv3d = nn.Conv3d(1, kernel_num_3d, padding='valid',
                                kernel_size=kernel_size_3d, stride=stride_3d)

        self.conv2d_2 = nn.Conv2d(self.depth, embed_dim, padding='valid',
                                  kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv2d_1(x).unsqueeze(-4)
        x = self.conv3d(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.conv2d_2(x)
        # x = torch.cat(x.split(1, 1), dim=2).squeeze()

        x = x.flatten(2).transpose(1, 2)
        return x


class PixelEmbed_Linear(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_channels=204, embed_dim=None):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.proj(x)
        # x = torch.cat(x.split(1, 1), dim=2).squeeze()
        x = x.reshape(shape[0], shape[1] * shape[2], -1)

        return x


if __name__ == '__main__':
    inputs = torch.randn((20, 204, 15, 15))
    # # inputs = np.random.rand(20, 1, 204, 5, 5).astype('float32')
    # inputs = torch.tensor(inputs)
    # m = PixelEmbed_cov(in_channels=204, embed_dim=786)
    m = PixelEmbed(in_channels=204, embed_dim=786)
    # cnn = nn.Conv2d(204, 4, padding='valid',
    #                 kernel_size=(1, 1), stride=(1, 1))
    # outputs = cnn(inputs)
    outputs = m(inputs)
    p = PosCNN(in_chans=786, embed_dim=786)
    pe = p(outputs)
    # band_split = 60
    # split_size = 204//(band_split-1)
    # x = torch.randn((20, 204, 5, 5))
    # x = x.chunk(70, 1)
    # torch.chunk()
    a = 0
