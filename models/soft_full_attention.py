import torch
from torch import nn


class SoftFullAttention(nn.Module):
    def __init__(self, inchannels, h, w):
        super(SoftFullAttention, self).__init__()

        self.P_v = nn.Parameter(torch.rand(h, w)).to(self.device)
        self.c1_v = torch.nn.Conv2d(inchannels, inchannels, (1, 1)).to(self.device)
        self.c2_v = torch.nn.Conv2d(inchannels, inchannels, (1, 1)).to(self.device)

        self.P_h = nn.Parameter(torch.rand(w, h)).to(self.device)
        self.c1_h = torch.nn.Conv2d(inchannels, inchannels, (1, 1)).to(self.device)
        self.c2_h = torch.nn.Conv2d(inchannels, inchannels, (1, 1)).to(self.device)
        self.soft = torch.nn.Softmax(dim=-1).to(self.device)

    def vertical(self, source, target):
        source1 = source + self.P_v
        target1 = target + self.P_v
        source1 = self.c1_v(source1).transpose(1, 3)
        target1 = self.c2_v(target1).transpose(1, 3).transpose(2, 3)
        res1 = torch.matmul(source1, target1)
        # res1 = self.soft(res1)
        res2 = target.transpose(1, 3)
        return torch.matmul(res1, res2).transpose(1, 3)

    def horizontal(self, source, target):
        source1_h = source.transpose(2, 3) + self.P_h
        target1_h = target.transpose(2, 3) + self.P_h
        source1_h = self.c1_h(source1_h).transpose(1, 3)
        target1_h = self.c2_h(target1_h).transpose(1, 3).transpose(2, 3)
        res1_h = torch.matmul(source1_h, target1_h)
        # res1_h = self.soft(res1_h)
        res2_h = target.transpose(2, 3).transpose(1, 3)
        # output D H W
        horizontal = torch.matmul(res1_h, res2_h).transpose(1, 3).transpose(2, 3)
        return horizontal

    def forward(self, source, target):
        return torch.cat([self.horizontal(source, target), self.vertical(source, target)], 1)