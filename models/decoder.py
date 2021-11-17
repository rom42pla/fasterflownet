from torch import nn

from models.dynamiconvolution import DynamicMultiHeadConv
from models.virtual_pooling import VirtualConv2d


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    # return nn.Sequential(
    #    nn.Upsample( scale_factor=(2, 2), mode="bilinear"),#nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)
    #    nn.Conv2d(in_planes, out_planes, 3, 1, padding=1),
    # )


def groupconvrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.ReLU( inplace=True),
        nn.GroupNorm(3,out_channels)
    )


class Decoder(nn.Module):
    def __init__(self, in_channels, groups,compression=1):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        # print("decoder  start",x.shape)
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        # print("decoder  stop",x.shape)
        return out


class FasterDecoder(nn.Module):
    def __init__(self, in_channels, groups,compression=1):
        super(FasterDecoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = groupconvrelu(in_channels, 96, 3, 1)
        self.conv2 = groupconvrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = groupconvrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = groupconvrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = groupconvrelu(96, 66, 3, 1)
        self.conv6 = groupconvrelu(66, 33, 3, 1)
        self.conv7 = nn.Conv2d(33, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        # print("decoder  start",x.shape)
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        # print("decoder  stop",x.shape)
        return out
