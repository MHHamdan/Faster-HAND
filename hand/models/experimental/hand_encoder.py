"""
HAND Encoder Architecture
Based on HAND paper Section III-A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Conv2d, InstanceNorm2d, ReLU
from hand.models.experimental.encoder_components import (
    GatedDepthwiseSeparableConv,
    OctaveConv,
    OctaveConvV2,
    SqueezeExcitation
)
from hand.models.baseline.fcn_encoder import MixDropout, DepthSepConv2D
import random


class HANDEncoderBlock(Module):
    """
    Single HAND encoder block implementing the architecture from the paper
    Follows equations 1-5 from the HAND paper
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1), dropout=0.4, use_octave=True,
                 alpha=0.5, use_gate=True, use_se=True, use_residual=True,
                 octave_impl="legacy"):
        super(HANDEncoderBlock, self).__init__()

        # Each component is independently switchable so the ablation table can attribute
        # performance to individual contributions rather than to the block as a whole.
        self.use_octave = use_octave
        self.use_gate = use_gate
        self.use_se = use_se
        self.use_residual = use_residual

        # Equation 1: Initial 2D Conv + FCN
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm1 = InstanceNorm2d(out_channels, eps=0.001, momentum=0.99, track_running_stats=False)
        self.relu = ReLU(inplace=True)

        # Equation 2: Gated Depth-wise Separable Convolution.
        # Ablating the gate falls back to an ordinary depth-wise separable conv so that
        # the comparison isolates the gating mechanism, not the parameter budget.
        if self.use_gate:
            self.gated_dsc = GatedDepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.gated_dsc = DepthSepConv2D(out_channels, out_channels, kernel_size=(3, 3))
        self.norm2 = InstanceNorm2d(out_channels, eps=0.001, momentum=0.99, track_running_stats=False)

        # Equation 3: Octave Convolution (optional)
        # "legacy" is the implementation every Experiment 2 result used and stays the
        # default. "corrected" swaps in OctaveConvV2 and bilinear upsampling -- Experiment
        # 2.1 arm C. See docs/ablations.md section 1.1.
        self.octave_impl = octave_impl
        self._upsample_mode = "bilinear" if octave_impl == "corrected" else "nearest"
        if self.use_octave:
            octave_cls = OctaveConvV2 if octave_impl == "corrected" else OctaveConv
            self.octave_conv = octave_cls(out_channels, out_channels, kernel_size=3, alpha=alpha, padding=1)
            # Equation 4: Squeeze-and-Excitation (optional, for ablation)
            self.se = SqueezeExcitation(out_channels, reduction=16) if self.use_se else None
            # Additional conv to ensure output channels match after octave
            self.channel_adjust = Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            # Expand high-frequency only output (alpha=0.5 means h_out = out_channels/2)
            h_out = int(out_channels * (1 - alpha))
            self.h_expand = Conv2d(h_out, out_channels, kernel_size=1, bias=False) if h_out < out_channels else None

        # Equation 5: Gated Convolution + FCN
        self.conv_final = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm_final = InstanceNorm2d(out_channels, eps=0.001, momentum=0.99, track_running_stats=False)

        # MixDropout as described in paper
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        # Equation 1: f_1 = Conv2D(FCN(X))
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        pos = random.randint(1, 4)
        if pos == 1:
            x = self.dropout(x)

        # Equation 2: Gated DSC
        f_2 = self.gated_dsc(x)
        f_2 = self.norm2(f_2)
        f_2 = self.relu(f_2)

        if pos == 2:
            f_2 = self.dropout(f_2)

        # Equations 3-4: Octave Conv + SE
        if self.use_octave:
            f_3_h, f_3_l = self.octave_conv(f_2)
            # Properly combine high and low frequency components
            if f_3_l is not None and f_3_h is not None:
                # Upsample low frequency to match high frequency spatial dimensions
                f_3_l_up = (F.interpolate(f_3_l, size=f_3_h.shape[2:],
                                          mode='bilinear', align_corners=False)
                            if self._upsample_mode == 'bilinear'
                            else F.interpolate(f_3_l, size=f_3_h.shape[2:], mode='nearest'))
                # Concatenate along channel dimension
                f_3_combined = torch.cat([f_3_h, f_3_l_up], dim=1)
                # Adjust to correct number of channels
                f_3_combined = self.channel_adjust(f_3_combined)
                f_4 = self.se(f_3_combined, None) if self.se is not None else f_3_combined
            elif f_3_h is not None:
                # Only the high-frequency path survived (input too small to pool).
                # Expand back to out_channels *before* SE so the SE block always sees
                # the channel count it was constructed for.
                f_4 = self.h_expand(f_3_h) if self.h_expand is not None else f_3_h
                if self.se is not None:
                    f_4 = self.se(f_4, None)
            else:
                f_4 = f_2
        else:
            f_4 = f_2

        if pos == 3:
            f_4 = self.dropout(f_4)

        # Equation 5: Final conv
        f_5 = self.conv_final(f_4)
        f_5 = self.norm_final(f_5)
        f_5 = self.relu(f_5)

        if pos == 4:
            f_5 = self.dropout(f_5)

        return f_5


class HAND_Encoder(Module):
    """
    Complete HAND Encoder
    5 advanced convolutional blocks as described in the paper
    """
    def __init__(self, params):
        super(HAND_Encoder, self).__init__()

        self.dropout = params.get("dropout", 0.5)
        input_channels = params.get("input_channels", 3)
        enc_dim = params.get("enc_dim", 256)

        # Ablation switches (default: full model as described in the paper)
        use_octave = params.get("enc_use_octave", True)
        use_gate = params.get("enc_use_gate", True)
        use_se = params.get("enc_use_se", True)
        self.use_residual = params.get("enc_use_residual", True)
        alpha = params.get("enc_octave_alpha", 0.5)
        # Experiment 2.1: octave implementation and encoder depth are both selectable.
        # Defaults reproduce Experiment 2 exactly.
        octave_impl = params.get("enc_octave_impl", "legacy")
        n_blocks = params.get("enc_blocks", 5)
        opts = dict(use_gate=use_gate, use_se=use_se, alpha=alpha,
                    octave_impl=octave_impl)
        self.octave_impl = octave_impl
        self.n_blocks = n_blocks

        # Following the paper's architecture progression
        # Starting with standard conv, then using advanced blocks

        # Initial processing (7x7 kernel as mentioned in paper)
        self.init_conv = Conv2d(input_channels, 16, kernel_size=7, padding=3, stride=1)
        self.init_norm = InstanceNorm2d(16, eps=0.001, momentum=0.99, track_running_stats=False)
        self.relu = ReLU(inplace=True)

        # 5 advanced convolutional blocks
        # Strides give a total reduction of H/32, W/8, matching the dataset manager's
        # height_divisor=32 / width_divisor=8 and DAN's FCN_Encoder output geometry.
        # The five down-sampling stages are fixed: they define the H/32, W/8 output
        # geometry the decoder and dataset manager assume. Depth beyond five is added as
        # stride-(1,1) blocks interleaved after each stage, so a deeper encoder keeps
        # exactly the same output shape -- which is what makes a depth-matched comparison
        # against the 10-block FCN_Encoder meaningful (Experiment 2.1 arm D).
        stages = [
            # (in, out, stride, use_octave). Stage 1 stays single-frequency: octave conv
            # needs spatial extent to pool.
            (16, 32, (2, 2), False),
            (32, 64, (2, 2), use_octave),
            (64, 128, (2, 2), use_octave),
            (128, 128, (2, 1), use_octave),
            (128, enc_dim, (2, 1), use_octave),
        ]
        if n_blocks < len(stages):
            raise ValueError("enc_blocks must be >= {}".format(len(stages)))
        extra = n_blocks - len(stages)
        # distribute the identity blocks as evenly as possible over the five stages
        per_stage = [extra // len(stages) + (1 if i < extra % len(stages) else 0)
                     for i in range(len(stages))]

        blocks = []
        for (c_in, c_out, stride, oct_on), n_extra in zip(stages, per_stage):
            blocks.append(HANDEncoderBlock(c_in, c_out, stride=stride,
                                           dropout=self.dropout, use_octave=oct_on, **opts))
            for _ in range(n_extra):
                blocks.append(HANDEncoderBlock(c_out, c_out, stride=(1, 1),
                                               dropout=self.dropout, use_octave=oct_on, **opts))
        self.blocks = ModuleList(blocks)

    def forward(self, x):
        # Initial large kernel convolution (paper mentions 7x7)
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.relu(x)

        # Pass through the 5 advanced blocks. Identity-shaped blocks get a residual
        # connection, as in the FCN encoder this is compared against; blocks that change
        # channel count or stride cannot, so they pass through directly.
        for block in self.blocks:
            xt = block(x)
            x = x + xt if (self.use_residual and x.size() == xt.size()) else xt

        return x


class ComplexityAssessmentNetwork(Module):
    """
    Complexity Assessment Network (Equation 23 from HAND paper)
    C(x) = φ(Encoder(x)) ∈ [0, 1]
    """
    def __init__(self, enc_dim=256, hidden_dim=128):
        super(ComplexityAssessmentNetwork, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d((4, 16))  # Adaptive pooling

        # Scale-aware architecture (Appendix B)
        self.fc1 = nn.Linear(enc_dim * 4 * 16, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        """
        features: encoder output [B, C, H, W]
        returns: complexity score [B, 1] in range [0, 1]
        """
        # Adaptive pooling to fixed size
        x = self.global_pool(features)

        # Flatten
        b, c, h, w = x.size()
        x = x.view(b, -1)

        # Feature projection
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        # Complexity score
        x = self.fc3(x)
        complexity = self.sigmoid(x)

        return complexity
