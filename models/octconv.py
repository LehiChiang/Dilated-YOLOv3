import torch.nn as nn
import math

from models.common import autopad


class OctaveC(nn.Module):
    def __init__(self, c1, c2, k, stride=1, ai=0.5, ao=0.5, p=None, dilation=1, groups=1, bias=False):
        super(OctaveC, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == c1
        assert 0 <= ai <= 1 and 0 <= ao <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = ai, ao
        self.conv_l2l = None if ai == 0 or ao == 0 else \
            nn.Conv2d(int(ai * c1), int(ao * c2), k, 1, autopad(k, p), dilation, math.ceil(ai * groups), bias)
        self.conv_l2h = None if ai == 0 or ao == 1 or self.is_dw else \
            nn.Conv2d(int(ai * c1), c2 - int(ao * c2), k, 1, autopad(k, p), dilation, groups, bias)
        self.conv_h2l = None if ai == 1 or ao == 0 or self.is_dw else \
            nn.Conv2d(c1 - int(ai * c1), int(ao * c2), k, 1, autopad(k, p), dilation, groups, bias)
        self.conv_h2h = None if ai == 1 or ao == 1 else \
            nn.Conv2d(c1 - int(ai * c1), c2 - int(ao * c2), k, 1, autopad(k, p), dilation, math.ceil(groups - ai * groups), bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class OctaveCB(nn.Module):
    def __init__(self, c1, c2, k, s=1, ai=0.5, ao=0.5, p=None, dilation=1, groups=1, bias=False):
        super(OctaveCB, self).__init__()
        self.conv = OctaveC(c1, c2, k, s, ai, ao, p, dilation, groups, bias)
        self.bn_h = None if ao == 1 else nn.BatchNorm2d(int(c2 * (1 - ao)))
        self.bn_l = None if ao == 0 else nn.BatchNorm2d(int(c2 * ao))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class OctaveCBA(nn.Module):
    def __init__(self, c1, c2, k, s=1, ai=0.5, ao=0.5, padding=None, dilation=1, g=1, bias=False):
        super(OctaveCBA, self).__init__()
        self.conv = OctaveC(c1, c2, k, s, ai, ao, padding, dilation, g, bias)
        self.bn_h = None if ao == 1 else nn.BatchNorm2d(int(c2 * (1 - ao)))
        self.bn_l = None if ao == 0 else nn.BatchNorm2d(int(c2 * ao))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l


class OctaveBottleneck(nn.Module):
    def __init__(self, c1, c2, ai=0.5, ao=0.5, output=False, e=0.5, g=1):
        super(OctaveBottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = OctaveCBA(c1, c_, k=1, s=1, ai=ai, ao=ao)
        self.conv2 = OctaveCBA(c_, c2, k=3, s=1, g=g, ai=0 if output else 0.5, ao=0 if output else 0.5)

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None
        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None
        return x_h, x_l