from torch import nn


class VirtualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(VirtualConv2d, self).__init__()
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride * 2, int((stride * 2 - 1 - 3) / 2 * -1), 1, groups,
                      bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        )

    def forward(self, x):
        return self.v(x)
