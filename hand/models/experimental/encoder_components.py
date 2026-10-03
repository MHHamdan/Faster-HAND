"""
Encoder components of the proposed HAND architecture (manuscript Eq. 2-4).

Used by `hand.models.experimental.hand_encoder.HAND_Encoder`, which is selectable
via `--encoder hand` but was NOT used for any released checkpoint. See
docs/ablations.md section 1.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, LayerNorm, Dropout
from torch.nn.init import xavier_uniform_
import math


class GatedDepthwiseSeparableConv(Module):
    """
    Gated Depth-wise Separable Convolution (Equation 2 from HAND paper)
    f_2 = σ(W_g * DSConv(f_1)) ⊙ (W_l * DSConv(f_1))
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedDepthwiseSeparableConv, self).__init__()

        # Depth-wise convolution
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=in_channels, bias=False)

        # Point-wise convolutions for gating
        self.pointwise_global = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_local = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply depth-wise convolution
        x_dw = self.depthwise(x)

        # Global and local pathways
        x_global = self.sigmoid(self.pointwise_global(x_dw))
        x_local = self.pointwise_local(x_dw)

        # Gated combination (element-wise multiplication)
        return x_global * x_local


class OctaveConv(Module):
    """
    Octave Convolution (Equation 3 from HAND paper)
    Decomposes features into high and low frequency components
    f_3^H, f_3^L = OctaveConv(f_2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5, stride=1, padding=1):
        super(OctaveConv, self).__init__()

        self.alpha = alpha
        self.stride = stride

        # Channel splits
        self.h_in = int(in_channels * (1 - alpha))
        self.l_in = in_channels - self.h_in
        self.h_out = int(out_channels * (1 - alpha))
        self.l_out = out_channels - self.h_out

        # High frequency pathway
        if self.h_in > 0 and self.h_out > 0:
            self.h2h = Conv2d(self.h_in, self.h_out, kernel_size, stride, padding, bias=False)
        if self.l_in > 0 and self.h_out > 0:
            self.l2h = Conv2d(self.l_in, self.h_out, kernel_size, stride, padding, bias=False)

        # Low frequency pathway
        if self.h_in > 0 and self.l_out > 0:
            self.h2l = Conv2d(self.h_in, self.l_out, kernel_size, stride, padding, bias=False)
        if self.l_in > 0 and self.l_out > 0:
            self.l2l = Conv2d(self.l_in, self.l_out, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        if isinstance(x, tuple):
            x_h, x_l = x
        else:
            # First octave conv - split input channels
            if self.h_in > 0:
                x_h = x[:, :self.h_in, :, :]
            else:
                x_h = None

            if self.l_in > 0:
                x_l = x[:, self.h_in:, :, :]
                # Check if spatial dimensions are large enough for pooling
                if x_l.size(2) >= 2 and x_l.size(3) >= 2:
                    x_l = F.avg_pool2d(x_l, kernel_size=2, stride=2)
                else:
                    # Dimensions too small, disable low-frequency path
                    x_l = None
            else:
                x_l = None

        # High frequency output
        h_out = None
        if self.h_out > 0:
            if x_h is not None and self.h_in > 0:
                h_out = self.h2h(x_h)
            if x_l is not None and self.l_in > 0:
                l2h_conv = self.l2h(x_l)
                # Match exact size of h_out if it exists, otherwise use scale_factor
                if h_out is not None:
                    l2h = F.interpolate(l2h_conv, size=h_out.shape[2:], mode='nearest')
                else:
                    l2h = F.interpolate(l2h_conv, scale_factor=2, mode='nearest')
                h_out = h_out + l2h if h_out is not None else l2h

        # Low frequency output
        l_out = None
        if self.l_out > 0:
            if x_h is not None and self.h_in > 0:
                # Check if spatial dimensions are large enough for pooling
                if x_h.size(2) >= 2 and x_h.size(3) >= 2:
                    h2l = F.avg_pool2d(x_h, kernel_size=2, stride=2)
                    l_out = self.h2l(h2l)
            if x_l is not None and self.l_in > 0:
                l2l = self.l2l(x_l)
                l_out = l_out + l2l if l_out is not None else l2l

        return h_out, l_out


class SqueezeExcitation(Module):
    """
    Squeeze-and-Excitation block (Equation 4 from HAND paper)
    f_4 = SE(f_3^H, f_3^L)

    Channel count is fixed at construction: building the FC layers lazily inside
    forward() would leave them out of the optimizer's parameter list (the optimizer is
    built before the first forward) and out of any checkpoint saved before that call,
    so the block would never actually train.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        self.channels = channels
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = Linear(channels, reduced_channels, bias=False)
        self.fc2 = Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l=None):
        # Combine high and low frequency if both exist
        if x_l is not None:
            x_l_up = F.interpolate(x_l, size=x_h.shape[2:], mode='nearest')
            x = x_h + x_l_up
        else:
            x = x_h

        b, c, _, _ = x.size()
        if c != self.channels:
            raise ValueError(
                "SqueezeExcitation configured for {} channels but received {}".format(
                    self.channels, c))

        # Squeeze: global average pooling
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Excitation: FC bottleneck
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(b, c, 1, 1)

        return x * excitation.expand_as(x)


class OctaveConvV2(Module):
    """
    Corrected octave convolution — Experiment 2.1, arm C.

    `OctaveConv` above is untouched and remains the default, so every Experiment 2
    result stays reproducible. This class differs from it in exactly four ways, each
    targeting the defect identified in docs/ablations.md section 1.1:

    1. **No silent low-frequency disabling.** The original drops the low path whenever
       H < 2 or W < 2, which makes the architecture depend on input height: a line
       image under ~31 px traverses a structurally different network from a taller one,
       and batch padding changes which. Here the pooling kernel adapts per axis
       (`k = 2 if dim >= 2 else 1`), so the low path is *always* present; an axis that
       cannot be halved is simply carried at full resolution on that axis.
    2. **Bilinear upsampling** instead of nearest-neighbour, so restoring the low path
       does not reintroduce the pixel-replication blocking artefacts that nearest
       produces on 1-4 px tall feature maps.
    3. **Explicit dimension verification.** Output shapes are asserted against what the
       channel split predicts, so a silent shape mismatch fails loudly.
    4. **Usage accounting.** `self.usage` records how often each path ran and how often
       an axis could not be pooled, so the behaviour is observable rather than inferred.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5,
                 stride=1, padding=1, verify_dims=True):
        super(OctaveConvV2, self).__init__()

        self.alpha = alpha
        self.stride = stride
        self.verify_dims = verify_dims

        self.h_in = int(in_channels * (1 - alpha))
        self.l_in = in_channels - self.h_in
        self.h_out = int(out_channels * (1 - alpha))
        self.l_out = out_channels - self.h_out

        if self.h_in > 0 and self.h_out > 0:
            self.h2h = Conv2d(self.h_in, self.h_out, kernel_size, stride, padding, bias=False)
        if self.l_in > 0 and self.h_out > 0:
            self.l2h = Conv2d(self.l_in, self.h_out, kernel_size, stride, padding, bias=False)
        if self.h_in > 0 and self.l_out > 0:
            self.h2l = Conv2d(self.h_in, self.l_out, kernel_size, stride, padding, bias=False)
        if self.l_in > 0 and self.l_out > 0:
            self.l2l = Conv2d(self.l_in, self.l_out, kernel_size, stride, padding, bias=False)

        # Plain counters rather than buffers: diagnostic state must not enter the
        # checkpoint or change the parameter count.
        self.usage = {"calls": 0, "low_path_active": 0,
                      "height_unpooled": 0, "width_unpooled": 0}

    @staticmethod
    def _pool(x):
        """Halve each axis that is large enough to halve. Never returns None."""
        kh = 2 if x.size(2) >= 2 else 1
        kw = 2 if x.size(3) >= 2 else 1
        if kh == 1 and kw == 1:
            return x, kh, kw
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw)), kh, kw

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        if isinstance(x, tuple):
            x_h, x_l = x
            kh = kw = 2
        else:
            x_h = x[:, :self.h_in] if self.h_in > 0 else None
            if self.l_in > 0:
                x_l, kh, kw = self._pool(x[:, self.h_in:])
            else:
                x_l, kh, kw = None, 1, 1

        self.usage["calls"] += 1
        if x_l is not None:
            self.usage["low_path_active"] += 1
        if kh == 1:
            self.usage["height_unpooled"] += 1
        if kw == 1:
            self.usage["width_unpooled"] += 1

        h_out = None
        if self.h_out > 0:
            if x_h is not None and self.h_in > 0:
                h_out = self.h2h(x_h)
            if x_l is not None and self.l_in > 0:
                l2h = self.l2h(x_l)
                target = h_out.shape[2:] if h_out is not None else \
                    (l2h.size(2) * kh, l2h.size(3) * kw)
                l2h = self._up(l2h, target)
                h_out = l2h if h_out is None else h_out + l2h

        l_out = None
        if self.l_out > 0:
            if x_h is not None and self.h_in > 0:
                pooled, _, _ = self._pool(x_h)
                l_out = self.h2l(pooled)
            if x_l is not None and self.l_in > 0:
                l2l = self.l2l(x_l)
                if l_out is not None and l_out.shape[2:] != l2l.shape[2:]:
                    l2l = self._up(l2l, l_out.shape[2:])
                l_out = l2l if l_out is None else l_out + l2l

        if self.verify_dims:
            if h_out is not None and h_out.size(1) != self.h_out:
                raise RuntimeError(
                    "OctaveConvV2: high path produced {} channels, expected {}".format(
                        h_out.size(1), self.h_out))
            if l_out is not None and l_out.size(1) != self.l_out:
                raise RuntimeError(
                    "OctaveConvV2: low path produced {} channels, expected {}".format(
                        l_out.size(1), self.l_out))
            if self.l_out > 0 and l_out is None:
                raise RuntimeError(
                    "OctaveConvV2: low path vanished — this is the defect the class exists "
                    "to prevent (input {}x{})".format(x_h.size(2) if x_h is not None else -1,
                                                      x_h.size(3) if x_h is not None else -1))
        return h_out, l_out
