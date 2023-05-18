import numpy as np
import torch
from torch import nn


class Linear_comb(nn.Module):
    def __init__(self, embed_dim):
        super(Linear_comb, self).__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        cls_token = tokens[:, 0, :].unsqueeze(dim=1)
        tokens = self.proj(tokens[:, 1:, :])
        coefficients = tokens @ cls_token.transpose(1, 2)
        tokens = (tokens * coefficients).sum(1)
        cls_token = cls_token.squeeze() + tokens
        return cls_token


if __name__ == '__main__':
    x = torch.randn((20, 6, 512))
    linear1 = Linear_comb(embed_dim=512)
    y = linear1(x)
    a = 0
