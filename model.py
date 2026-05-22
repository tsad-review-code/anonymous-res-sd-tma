import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import RevIN1d


# ==========================================
# Core Module 1: Temporal Mean Attention (TMA)
# Function: acts as an adaptive low-pass filter by dynamically evaluating
#           the global energy distribution within a local temporal window,
#           effectively smoothing high-frequency spike noise in real-world
#           industrial scenarios.
# ==========================================
class TMeanSEBlock1d(nn.Module):
    def __init__(self, channels, reduction=4):
        super(TMeanSEBlock1d, self).__init__()
        # Ensure that the reduced dimension is at least 1.
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()
        # Global mean pooling: removes local high-frequency oscillations along
        # the temporal axis and extracts a pure channel-energy descriptor.
        y = x.mean(dim=2)  # [Batch, Channels]

        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1)
        # Dynamic channel-wise gating: adaptively suppresses high-frequency
        # noise channels while preserving key channels that contain low-frequency
        # anomaly evolution patterns.
        return x * y


# ==========================================
# Core Module 2: Residual Spatiotemporal Decoupling Block (Res-SD Block)
# Function: strictly decouples the extraction of temporal evolution features
#           and spatial topological features to prevent feature coupling among
#           multivariate physical variables. The residual connection mitigates
#           gradient vanishing and preserves the low-frequency baseline
#           information of the original signal.
# ==========================================
class ResSDBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super(ResSDBlock, self).__init__()
        # Step A: Physical isolation (Depthwise)
        # Forces each convolution kernel to slide only along the temporal axis
        # of a single channel, cutting off cross-interference among heterogeneous
        # physical variables.
        self.depthwise = nn.Conv1d(in_c, in_c, kernel_size=k, stride=1, padding=k // 2, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)

        # Step B: Topological fusion (Pointwise)
        # Performs cross-channel linear combination to reconstruct transient
        # spatial topological information in the multivariate sensor network.
        self.pointwise = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        # Step C: Temporal Mean Attention (TMA denoising module)
        self.se = TMeanSEBlock1d(out_c)

        # Residual shortcut connection
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            # If the number of channels changes, a 1x1 convolution is used to
            # align dimensions and ensure stable feature transmission.
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        # 1. Preserve a backup of the original signal for retaining the
        # low-frequency baseline in the subsequent residual fusion.
        residual = self.shortcut(x)

        # 2. Spatiotemporal decoupled feature extraction:
        # first temporal evolution extraction, then spatial topological fusion.
        out = self.depthwise(x)
        out = F.relu(self.bn1(out), inplace=True)

        out = self.pointwise(out)
        out = self.bn2(out)

        # 3. Dynamic mean-based channel weighting for adaptive suppression of
        # high-frequency spike noise.
        out = self.se(out)

        # 4. Residual addition: fuses purified anomaly-related features with the
        # original low-frequency baseline.
        out += residual
        return F.relu(out, inplace=True)



''''# ==========================================
# Ablation version: standard residual block (without spatiotemporal decoupling, w/o SD)
# Function: mixes temporal and spatial information again to demonstrate the core
#           value of the decoupling mechanism in mitigating multivariate channel interference.
# ==========================================
class ResSDBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super(ResSDBlock, self).__init__()

        # Ablation modification: remove the physical isolation mechanism of
        # Depthwise and Pointwise convolutions.
        # Replace it with a standard 1D convolution, forcing channels with
        # different physical meanings to be coupled during temporal feature extraction.
        self.standard_conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=1, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_c)

        # Keep the initialization of mean-based channel attention.
        # In the w/o TMA experiment, its forward call is commented out.
        self.se = TMeanSEBlock1d(out_c)

        # Keep the residual shortcut connection unchanged.
        # This ensures that performance differences originate only from the
        # decoupled convolution mechanism, excluding interference caused by
        # changes in network depth.
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        # 1. Preserve a backup of the original signal.
        residual = self.shortcut(x)

        # 2. Conventional feature purification without decoupling:
        # temporal and channel dimensions are jointly extracted.
        out = self.standard_conv(x)
        out = self.bn(out)

        # 3. Dynamic mean-based weighting.
        # In the experiment that purely verifies the effect of decoupling,
        # this line is kept commented out to control variables.
        # out = self.se(out)

        # 4. Residual addition.
        out += residual
        return F.relu(out, inplace=True)
'''
class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,
                 revin_affine: bool = False,
                 revin_eps: float = 1e-5,
                 revin_min_sigma: float = 1e-5
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # ====================================================
        # Note: since the front-end Kinematic Expansion module is introduced,
        # if the input is univariate (in_channels=1), it will be dynamically
        # expanded into three physical dimensions: original value, velocity,
        # and acceleration. Therefore, the actual input channel number of the
        # first layer should be set to 3.
        # ====================================================
        self.actual_in_channels = 3 if in_channels == 1 else in_channels
        #self.actual_in_channels = in_channels

        # 0. Basic anti-drift module initialized with the actual channel number.
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=self.actual_in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        # 1. Build the residual spatiotemporal decoupling network.
        blocks = []
        for i in range(len(self.layers)):
            # Use actual_in_channels as the input channel number of the first layer.
            in_c = layers[i - 1] if i > 0 else self.actual_in_channels
            out_c = self.layers[i]
            k = self.kss[i]
            # Directly call the residual decoupling block.
            blocks.append(ResSDBlock(in_c, out_c, k))

        self.convblocks = nn.ModuleList(blocks)

        # 2. Backend prediction heads while keeping the interface strictly consistent.
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.classification_head = nn.Linear(self.layers[-1] * 2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):

        # ====================================================
        # Core innovation: univariate Kinematic Expansion.
        # Placed at the front end of the network, it computes the discrete
        # first-order velocity and second-order acceleration, expanding an
        # isolated 1D sequence into a 3D dynamic state space and effectively
        # alleviating the information scarcity problem in univariate data.
        # ====================================================
        if x.size(1) == 1:
            v = torch.diff(x, dim=-1, prepend=x[:, :, :1])
            a = torch.diff(v, dim=-1, prepend=v[:, :, :1])
            x = torch.cat([x, v, a], dim=1)

        # ====================================================
        # Downstream flow: at this stage, univariate input has been expanded
        # into C=3 dynamic features, while multivariate input remains unchanged.
        # ====================================================

        # 1. Instance-level normalization.
        # RevIN normalizes all physical channels separately to remove distribution drift.
        if self.revin is not None:
            x = self.revin.norm(x)

        # 2. Pass through the residual spatiotemporal decoupling network sequentially.
        for block in self.convblocks:
            x = block(x)

        # 3. Global pooling compresses features into a clean 1D embedding.
        h = self.fc_embedding(x).flatten(start_dim=1)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)
