import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Correlation(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)
        #todo check indexing
        self.offsety, self.offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                     torch.arange(0, 2 * self.max_hdisp + 1)])

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dy:dy + hei, dx:dx + wid], 1, keepdim=True)
            for dx, dy in zip(self.offsetx.reshape(-1), self.offsety.reshape(-1))
        ], 1)
        # output = F.leaky_relu(output, 0.1)
        return output
        # return self.corr_pwc(in1, in2)

    def corr_pwc(self, refimg_fea, targetimg_fea):
        maxdisp = 4
        b, c, h, w = refimg_fea.shape
        targetimg_fea = F.unfold(targetimg_fea, (2 * maxdisp + 1, 2 * maxdisp + 1), padding=maxdisp).view(b, c,
                                                                                                          2 * maxdisp + 1,
                                                                                                          2 * maxdisp + 1 ** 2,
                                                                                                          h, w)
        cost = refimg_fea.view(b, c, h, w)[:, :, np.newaxis, np.newaxis] * targetimg_fea.view(b, c, 2 * maxdisp + 1,
                                                                                              2 * maxdisp + 1 ** 2, h,
                                                                                              w)
        cost = cost.sum(1)

        b, ph, pw, h, w = cost.size()
        cost = cost.view(b, ph * pw, h, w) / refimg_fea.size(1)
        return cost
