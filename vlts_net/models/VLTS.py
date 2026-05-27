"""Visual-conditioned low-rank target subspace AVSS network.

Main design goals
-----------------
1. Keep the same practical AVSS interface:
   forward(input_wav, mouth_emb) -> denoised_mag, denoised_phase,
   est_spec_uncompress, estimated_waveforms.
2. Recast the audio-visual interaction as target subspace estimation rather
   than ordinary feature gating or cross-modal attention.
3. Implement the signal-processing motivated decomposition used by the paper:
   - learnable visual-to-spectral condition projection;
   - band-wise visual-conditioned low-rank basis generation;
   - ridge-stabilized least-squares projection onto the visual target-support
     subspace;
   - approximate orthogonal residual decomposition;
   - anchor-conditioned residual target recovery;
   - mixture-derived magnitude-phase reconstruction.
4. Preserve stable end-to-end training with only the original separation loss.

CPU note
--------
The model does not use top-k routing, Python-side attention masks, or repeated
window creation. The subspace projection solves small K-by-K systems on the
active tensor device. For CPU-only debugging, setting OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1 can avoid small-convolution thread oversubscription.

Expected inputs
---------------
input_wav: (T), (B, T), or (B, 1, T)
mouth_emb: (B, C_v, T_v) or (B, C_v, T_v, 1)

Core tensor layout inside the model
-----------------------------------
Time-frequency feature maps use (B, C, T, F).
STFT tensors use PyTorch's complex layout (B, F, T).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    from .base_av_model import BaseAVModel
except Exception:
    class BaseAVModel(nn.Module):
        """Fallback base class for standalone debugging outside vlts_net.models."""

        def __init__(self, sample_rate: int = 16000):
            super().__init__()
            self.sample_rate = sample_rate


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _valid_num_groups(channels: int, preferred: int = 8) -> int:
    """Choose a GroupNorm group count that always divides channels."""
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _crop_or_pad_freq(x: Tensor, target_freq: int) -> Tensor:
    """Match the last dimension to target_freq after transposed convolution."""
    current = x.size(-1)
    if current == target_freq:
        return x
    if current > target_freq:
        return x[..., :target_freq]
    return F.pad(x, (0, target_freq - current))


def _zero_init(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


# -----------------------------------------------------------------------------
# Spectral utilities
# -----------------------------------------------------------------------------


def power_compress(x: Tensor, compression: float = 0.3, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    """Power-law compress a complex STFT.

    Args:
        x: Complex STFT with shape (B, F, T).
        compression: Power-law compression factor.

    Returns:
        compressed_complex_as_channels: (B, 2, F, T)
        compressed_magnitude: (B, F, T)
        phase: (B, F, T)
    """
    if not torch.is_complex(x):
        x = torch.view_as_complex(x.contiguous())
    mag = x.abs().clamp_min(eps).pow(compression)
    phase = torch.angle(x)
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.stack([real, imag], dim=1), mag, phase


def power_uncompress(x: Tensor, compression: float = 0.3, eps: float = 1e-8) -> Tensor:
    """Invert power-law compression and return a complex spectrogram."""
    if not torch.is_complex(x):
        x = torch.view_as_complex(x.contiguous())
    mag = x.abs().clamp_min(eps).pow(1.0 / compression)
    phase = torch.angle(x)
    return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


# -----------------------------------------------------------------------------
# Lightweight convolutional building blocks
# -----------------------------------------------------------------------------


class ConvNormAct(nn.Module):
    """Conv2d + GroupNorm + PReLU.

    GroupNorm is used instead of BatchNorm to reduce instability for small AVSS
    batches. It also avoids CPU-side running-stat updates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        if isinstance(dilation, int):
            dh, dw = dilation, dilation
        else:
            dh, dw = dilation
        padding = ((kh - 1) // 2 * dh, (kw - 1) // 2 * dw)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kh, kw),
            stride=stride,
            padding=padding,
            dilation=(dh, dw),
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_valid_num_groups(out_channels), out_channels)
        self.act = nn.PReLU(out_channels) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """Stable residual depthwise-separable TF convolution block."""

    def __init__(self, channels: int, kernel_size: Tuple[int, int] = (3, 3), dilation: Tuple[int, int] = (1, 1)):
        super().__init__()
        kh, kw = kernel_size
        dh, dw = dilation
        padding = ((kh - 1) // 2 * dh, (kw - 1) // 2 * dw)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_valid_num_groups(channels), channels)
        self.act = nn.PReLU(channels)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.act(self.norm(y))
        return x + self.scale * y


class TFAxisBlock(nn.Module):
    """Time-frequency modeling without heavy attention.

    Each block models temporal context and frequency context separately, then
    applies a small channel-mixing projection. This is cheaper and more stable
    than top-k attention while still giving global-ish TF context through stacked
    dilated convolutions.
    """

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.time_block = DepthwiseSeparableBlock(channels, kernel_size=(3, 1), dilation=(dilation, 1))
        self.freq_block = DepthwiseSeparableBlock(channels, kernel_size=(1, 5), dilation=(1, 1))
        self.channel_mlp = nn.Sequential(
            ConvNormAct(channels, channels * 2, kernel_size=1),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_num_groups(channels), channels),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.time_block(x)
        y = self.freq_block(y)
        y = self.channel_mlp(y)
        return self.act(x + self.scale * y)


class EncoderNoDownsample(nn.Module):
    """Initial encoder that keeps the original STFT frequency resolution."""

    def __init__(self, in_channels: int, channels: int, depth: int = 3):
        super().__init__()
        blocks = [ConvNormAct(in_channels, channels, kernel_size=1)]
        for i in range(depth):
            blocks.append(DepthwiseSeparableBlock(channels, dilation=(2 ** i, 1)))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderDownsampleFreq(nn.Module):
    """Encoder that downsamples only the frequency axis by 2."""

    def __init__(self, in_channels: int, channels: int, depth: int = 3):
        super().__init__()
        self.pre = EncoderNoDownsample(in_channels, channels, depth=depth)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.GroupNorm(_valid_num_groups(channels), channels),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.pre(x))


class FreqDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.GroupNorm(_valid_num_groups(channels), channels),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class FreqUpsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.GroupNorm(_valid_num_groups(channels), channels),
            nn.PReLU(channels),
        )

    def forward(self, x: Tensor, target_freq: int) -> Tensor:
        return _crop_or_pad_freq(self.up(x), target_freq)


# -----------------------------------------------------------------------------
# Proposed modules
# -----------------------------------------------------------------------------


class VisualToSpectralPrior(nn.Module):
    """Learnable visual-to-spectral prior projection.

    The visual stream is not simply copied along frequency. It predicts a
    convex combination of learnable spectral bases. The basis map gives a coarse
    target-speech prior over frequency, while the temporal visual embedding keeps
    lip-motion and speaking-dynamics information.
    """

    def __init__(self, visual_channels: int, channels: int, fft_bins: int, num_bases: int = 16):
        super().__init__()
        self.fft_bins = fft_bins
        self.num_bases = num_bases
        self.temporal_encoder = nn.Sequential(
            ConvNormAct(visual_channels, channels, kernel_size=(5, 1)),
            DepthwiseSeparableBlock(channels, kernel_size=(5, 1)),
            DepthwiseSeparableBlock(channels, kernel_size=(5, 1), dilation=(2, 1)),
            DepthwiseSeparableBlock(channels, kernel_size=(5, 1), dilation=(4, 1)),
        )
        self.alpha_head = nn.Conv2d(channels, num_bases, kernel_size=1)
        self.basis_logits = nn.Parameter(torch.randn(num_bases, fft_bins) * 0.02)
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
        )

    def forward(self, mouth_emb: Tensor, target_frames: int) -> Tensor:
        # mouth_emb: (B, C_v, T_v) or (B, C_v, T_v, 1)
        if mouth_emb.ndim == 3:
            mouth_emb = mouth_emb.unsqueeze(-1)
        if mouth_emb.ndim != 4:
            raise ValueError(f"Expected mouth_emb as (B,C,Tv) or (B,C,Tv,1), got {tuple(mouth_emb.shape)}")

        v = F.interpolate(mouth_emb, size=(target_frames, 1), mode="bilinear", align_corners=False)
        base = self.temporal_encoder(v)  # (B, C, T, 1)

        alpha = self.alpha_head(base).squeeze(-1)  # (B, K, T)
        alpha = F.softmax(alpha, dim=1)
        # Each basis has mean around 1 after scaling. This avoids large visual
        # magnitudes early in training.
        basis = F.softmax(self.basis_logits, dim=-1) * float(self.fft_bins)  # (K, F)
        spectral_prior = torch.einsum("bkt,kf->btf", alpha, basis).unsqueeze(1)  # (B,1,T,F)

        v_map = base * spectral_prior  # broadcast over frequency
        return self.refine(v_map)


class BandWiseLowRankBasisGenerator(nn.Module):
    """Generate a visual-conditioned low-rank target-support basis per band.

    The visual feature map is first pooled within speech-oriented frequency
    bands. A learnable band dictionary provides stable target-support directions,
    while the visual stream predicts a small dynamic adaptation.  This avoids an
    unconstrained visual projection and keeps the decomposition close to a
    band-wise subspace model.
    """

    def __init__(self, channels: int, num_bands: int = 16, rank: int = 16, delta_scale: float = 0.01):
        super().__init__()
        if rank >= channels:
            raise ValueError(f"rank must be smaller than channels for a low-rank subspace, got rank={rank}, C={channels}.")
        self.channels = channels
        self.num_bands = num_bands
        self.rank = rank
        self.delta_scale = delta_scale

        # Stable band-dependent acoustic basis.  It is normalized in forward(),
        # so the projection geometry does not depend on the raw parameter norm.
        self.base_basis = nn.Parameter(torch.randn(num_bands, channels, rank) * (channels ** -0.5))

        self.context = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
        )
        self.delta_head = _zero_init(nn.Conv2d(channels, channels * rank, kernel_size=1))

    @staticmethod
    def _band_edges(num_bins: int, num_bands: int, device: torch.device) -> Tensor:
        # Equal-width bands are used for robustness and simplicity.  A Bark/Mel
        # partition can be plugged in here without changing the module interface.
        edges = torch.linspace(0, num_bins, steps=num_bands + 1, device=device)
        edges = torch.round(edges).long()
        edges[0] = 0
        edges[-1] = num_bins
        # Ensure every band has at least one bin when num_bins >= num_bands.
        if num_bins >= num_bands:
            for i in range(1, num_bands):
                edges[i] = torch.clamp(edges[i], min=edges[i - 1] + 1, max=num_bins - (num_bands - i))
        return edges

    def _pool_bands(self, visual: Tensor) -> Tuple[Tensor, Tensor]:
        # visual: (B,C,T,F) -> band context: (B,C,T,num_bands)
        f_bins = visual.size(-1)
        edges = self._band_edges(f_bins, self.num_bands, visual.device)
        band_feats = []
        for b in range(self.num_bands):
            start = int(edges[b].item())
            end = int(edges[b + 1].item())
            band_feats.append(visual[..., start:end].mean(dim=-1, keepdim=True))
        return torch.cat(band_feats, dim=-1), edges

    def forward(self, visual: Tensor) -> Tuple[Tensor, Tensor]:
        # Returns basis U with shape (B,T,num_bands,C,K).
        band_context, edges = self._pool_bands(visual)
        h = self.context(band_context)  # (B,C,T,num_bands)
        delta = self.delta_head(h)  # (B,C*K,T,num_bands)
        bsz, _, frames, bands = delta.shape
        delta = delta.view(bsz, self.channels, self.rank, frames, bands)
        delta = delta.permute(0, 3, 4, 1, 2).contiguous()  # (B,T,Band,C,K)

        base = self.base_basis.unsqueeze(0).unsqueeze(0)  # (1,1,Band,C,K)
        basis = base + self.delta_scale * delta
        basis = F.normalize(basis, p=2, dim=-2, eps=1e-6)
        return basis, edges


class RidgeSubspaceProjector(nn.Module):
    """Ridge-stabilized least-squares projection onto visual subspaces."""

    def __init__(self, ridge: float = 1e-3):
        super().__init__()
        self.ridge = ridge

    def forward(self, audio: Tensor, basis: Tensor, edges: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # audio: (B,C,T,F), basis: (B,T,num_bands,C,K)
        bsz, channels, frames, f_bins = audio.shape
        rank = basis.size(-1)
        anchor_bands = []
        eye = torch.eye(rank, device=audio.device, dtype=torch.float32).view(1, 1, rank, rank)

        for b in range(basis.size(2)):
            start = int(edges[b].item())
            end = int(edges[b + 1].item())
            a_band = audio[..., start:end].permute(0, 2, 3, 1).contiguous()  # (B,T,Fb,C)
            u = basis[:, :, b]  # (B,T,C,K)

            # Solve (U^T U + lambda I)c = U^T a.  The solve is done in fp32 for
            # numerical stability under AMP, then cast back to the input dtype.
            u32 = u.float()
            a32 = a_band.float()
            gram = torch.einsum("btck,btcl->btkl", u32, u32) + self.ridge * eye
            rhs = torch.einsum("btck,btfc->btkf", u32, a32)
            coeff = torch.linalg.solve(gram, rhs)  # (B,T,K,Fb)
            anchor = torch.einsum("btck,btkf->btfc", u32, coeff)
            anchor = anchor.to(dtype=audio.dtype).permute(0, 3, 1, 2).contiguous()  # (B,C,T,Fb)
            anchor_bands.append(anchor)

        a_parallel = torch.cat(anchor_bands, dim=-1)
        a_perp = audio - a_parallel

        # A compact diagnostic: smaller values indicate better approximate
        # orthogonality between residual and the visual-conditioned basis.
        with torch.no_grad():
            residual_corr = []
            for b in range(basis.size(2)):
                start = int(edges[b].item())
                end = int(edges[b + 1].item())
                r_band = a_perp[..., start:end].permute(0, 2, 3, 1).float()
                u = basis[:, :, b].float()
                corr = torch.einsum("btck,btfc->btkf", u, r_band).abs().mean()
                residual_corr.append(corr)
            orthogonality_error = torch.stack(residual_corr).mean()
        return a_parallel, a_perp, orthogonality_error


class AnchorConditionedResidualTargetEstimator(nn.Module):
    """Recover target details from the residual using the projected target anchor.

    The residual is not treated as interference only.  It contains interference
    residuals and target details that are weakly covered by the visual condition,
    such as timbre, high-frequency consonants, fine spectral variations, and
    phase-related structures.  This module estimates target-related residuals
    using anchor-residual coherence instead of a plain compensation convolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.residual_candidate = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3), dilation=(2, 1)),
        )
        self.anchor_query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.residual_key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.selector = nn.Sequential(
            ConvNormAct(channels * 4 + 1, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, a_perp: Tensor, a_parallel: Tensor, visual: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        candidate = self.residual_candidate(a_perp)
        q = F.normalize(self.anchor_query(a_parallel), p=2, dim=1, eps=1e-6)
        k = F.normalize(self.residual_key(candidate), p=2, dim=1, eps=1e-6)
        coherence = (q * k).sum(dim=1, keepdim=True)  # (B,1,T,F)
        selector = self.selector(torch.cat([a_parallel, a_perp, candidate, visual, coherence], dim=1))
        a_perp_s = self.residual_scale * selector * candidate
        return a_perp_s, selector, coherence


class VisualConditionedLowRankAVEncoder(nn.Module):
    """Visual-conditioned low-rank target subspace decomposition encoder.

    This module replaces visual gating with a signal-processing motivated
    sequence: visual-conditioned low-rank basis estimation, ridge projection,
    approximate orthogonal residual decomposition, residual target recovery, and
    target-oriented acoustic reparameterization.
    """

    def __init__(
        self,
        channels: int,
        num_bands: int = 16,
        rank: int = 16,
        ridge: float = 1e-3,
        num_refine_blocks: int = 4,
    ):
        super().__init__()
        self.basis_generator = BandWiseLowRankBasisGenerator(channels, num_bands=num_bands, rank=rank)
        self.projector = RidgeSubspaceProjector(ridge=ridge)
        self.residual_estimator = AnchorConditionedResidualTargetEstimator(channels)
        self.reparameterizer = nn.Sequential(
            ConvNormAct(channels * 4, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            _zero_init(nn.Conv2d(channels, channels, kernel_size=1)),
        )
        refine_dilations = [1, 2, 4, 8]
        self.target_refiner = nn.Sequential(
            *[TFAxisBlock(channels, dilation=refine_dilations[i % len(refine_dilations)]) for i in range(num_refine_blocks)]
        )
        self.target_confidence = nn.Sequential(
            ConvNormAct(channels * 3 + 1, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Small layer scale preserves the original acoustic representation at
        # the beginning of training and avoids early projection-induced loss spikes.
        self.reparam_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    def forward(self, audio: Tensor, visual: Tensor, return_intermediates: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        if audio.shape != visual.shape:
            raise ValueError(f"audio and visual feature shapes must match, got {audio.shape} and {visual.shape}.")

        basis, edges = self.basis_generator(visual)
        a_parallel, a_perp, orthogonality_error = self.projector(audio, basis, edges)
        a_perp_s, residual_selector, anchor_residual_coherence = self.residual_estimator(a_perp, a_parallel, visual)
        a_star = a_parallel + a_perp_s

        delta = self.reparameterizer(torch.cat([a_star, a_parallel, a_perp_s, visual], dim=1))
        f_av = audio + self.reparam_scale * delta
        f_av = self.target_refiner(f_av)

        confidence = self.target_confidence(torch.cat([a_parallel, a_perp_s, visual, anchor_residual_coherence], dim=1))
        stats: Dict[str, Tensor] = {
            "target_confidence": confidence,
            # Keep the old key for compatibility with downstream code that expects
            # a channel-wise guidance map.
            "gain": confidence,
            "orthogonality_error": orthogonality_error,
        }
        if return_intermediates:
            stats.update(
                {
                    "subspace_basis": basis,
                    "band_edges": edges,
                    "A_parallel": a_parallel,
                    "A_perp": a_perp,
                    "A_perp_s": a_perp_s,
                    "A_star": a_star,
                    "residual_selector": residual_selector,
                    "anchor_residual_coherence": anchor_residual_coherence,
                }
            )
        return f_av, stats


class SubspaceGuidedComplexBottleneck(nn.Module):
    """Fuse target-reparameterized AV features with complex spectral observations."""

    def __init__(self, channels: int, gain_channels: Optional[int] = None):
        super().__init__()
        gain_channels = channels if gain_channels is None else gain_channels
        self.gain_channels = gain_channels
        self.gate = nn.Sequential(
            ConvNormAct(channels * 2 + gain_channels, channels, kernel_size=1),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 2 + 1 + gain_channels, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            _zero_init(nn.Conv2d(channels, channels, kernel_size=1)),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, f_av: Tensor, f_mp: Tensor, gain: Tensor) -> Tuple[Tensor, Tensor]:
        if gain.shape[-2:] != f_mp.shape[-2:]:
            gain = F.interpolate(gain, size=f_mp.shape[-2:], mode="bilinear", align_corners=False)
        q = self.gate(torch.cat([f_av, f_mp, gain], dim=1))
        delta = self.fuse(torch.cat([q * f_mp, f_av, q, gain], dim=1))
        return f_mp + self.res_scale * delta, q


class ResidualMagnitudeDecoder(nn.Module):
    """Mixture-anchored residual magnitude estimator.

    Instead of using a [0,1] mask, it predicts a bounded log-ratio. This permits
    target magnitude to exceed mixture magnitude when phase cancellation occurs,
    but keeps the prediction stable.
    """

    def __init__(self, channels: int, alpha: float = 1.25):
        super().__init__()
        self.alpha = alpha
        self.up = FreqUpsample(channels)
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            ConvNormAct(channels, channels, kernel_size=1),
            _zero_init(nn.Conv2d(channels, 1, kernel_size=1)),
        )

    def forward(self, x: Tensor, mixture_mag: Tensor) -> Tensor:
        # x: (B,C,T,Fd), mixture_mag: (B,F,T), output: (B,F,T)
        target_freq = mixture_mag.size(1)
        x = self.up(x, target_freq=target_freq)
        log_ratio = self.alpha * torch.tanh(self.body(x))  # (B,1,T,F)
        ratio = torch.exp(log_ratio).squeeze(1).permute(0, 2, 1)  # (B,F,T)
        return mixture_mag * ratio


class ResidualPhaseRotator(nn.Module):
    """Noisy-phase anchored residual phase rotation on the unit circle."""

    def __init__(self, channels: int, gamma_max: float = 0.75):
        super().__init__()
        self.gamma_max = gamma_max
        self.up = FreqUpsample(channels)
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            ConvNormAct(channels, channels, kernel_size=1),
        )
        self.delta_head = _zero_init(nn.Conv2d(channels, 2, kernel_size=1))
        self.gamma_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.constant_(self.gamma_head.bias, -1.0)

    def forward(self, x: Tensor, mixture_phase: Tensor) -> Tensor:
        # x: (B,C,T,Fd), mixture_phase: (B,F,T), output: (B,F,T)
        target_freq = mixture_phase.size(1)
        x = self.up(x, target_freq=target_freq)
        h = self.body(x)
        delta = torch.tanh(self.delta_head(h))  # (B,2,T,F)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_head(h))  # (B,1,T,F)

        phase_tf = mixture_phase.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,F)
        unit_y = torch.cat([torch.cos(phase_tf), torch.sin(phase_tf)], dim=1)
        unit_hat = F.normalize(unit_y + gamma * delta, p=2, dim=1, eps=1e-8)
        phase_hat = torch.atan2(unit_hat[:, 1], unit_hat[:, 0])  # (B,T,F)
        return phase_hat.permute(0, 2, 1)  # (B,F,T)


class SharedResidualMagnitudeDecoder(nn.Module):
    """Mixture-anchored residual magnitude estimator using a shared upsampled decoder feature.

    This version removes one duplicated transposed-convolution path by receiving
    the already upsampled bottleneck feature from the main network. The magnitude
    and phase heads still keep separate refinement bodies, so the output behavior
    remains task-specific while the expensive frequency upsampling is computed
    only once.
    """

    def __init__(self, channels: int, alpha: float = 1.25):
        super().__init__()
        self.alpha = alpha
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            ConvNormAct(channels, channels, kernel_size=1),
            _zero_init(nn.Conv2d(channels, 1, kernel_size=1)),
        )

    def forward(self, x_up: Tensor, mixture_mag: Tensor) -> Tensor:
        # x_up: (B,C,T,F), mixture_mag: (B,F,T), output: (B,F,T)
        x_up = _crop_or_pad_freq(x_up, mixture_mag.size(1))
        log_ratio = self.alpha * torch.tanh(self.body(x_up))
        ratio = torch.exp(log_ratio).squeeze(1).permute(0, 2, 1)
        return mixture_mag * ratio


class SharedResidualPhaseRotator(nn.Module):
    """Noisy-phase anchored residual phase rotation using a shared decoder feature."""

    def __init__(self, channels: int, gamma_max: float = 0.75):
        super().__init__()
        self.gamma_max = gamma_max
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(channels, kernel_size=(3, 3)),
            ConvNormAct(channels, channels, kernel_size=1),
        )
        self.delta_head = _zero_init(nn.Conv2d(channels, 2, kernel_size=1))
        self.gamma_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.constant_(self.gamma_head.bias, -1.0)

    def forward(self, x_up: Tensor, mixture_phase: Tensor) -> Tensor:
        # x_up: (B,C,T,F), mixture_phase: (B,F,T), output: (B,F,T)
        x_up = _crop_or_pad_freq(x_up, mixture_phase.size(1))
        h = self.body(x_up)
        delta = torch.tanh(self.delta_head(h))
        gamma = self.gamma_max * torch.sigmoid(self.gamma_head(h))

        phase_tf = mixture_phase.permute(0, 2, 1).unsqueeze(1)
        unit_y = torch.cat([torch.cos(phase_tf), torch.sin(phase_tf)], dim=1)
        unit_hat = F.normalize(unit_y + gamma * delta, p=2, dim=1, eps=1e-8)
        phase_hat = torch.atan2(unit_hat[:, 1], unit_hat[:, 0])
        return phase_hat.permute(0, 2, 1)


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------



class VLTS(BaseAVModel):
    """Visual-conditioned low-rank target subspace AVSS network.

    The constructor intentionally follows the uploaded CAMB-Net/VLTS case so
    that the same YAML/config entry can instantiate either model without adding
    a separate VLTS-specific configuration. Unused CAMB-Net arguments are kept
    for interface compatibility.
    """

    def __init__(
        self,
        out_channels: int = 128,
        in_channels: int = 512,
        vpre_channels: int = 512,
        vin_channels: int = 64,
        vout_channels: int = 64,
        num_blocks: int = 16,
        upsampling_depth: int = 4,
        enc_kernel_size: int = 21,
        num_sources: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        compression: float = 0.3,
    ):
        super().__init__(sample_rate=sample_rate)

        # Keep CAMB-Net argument names, but map them to this model internally.
        # out_channels controls the hidden feature width.
        # vpre_channels is the input visual embedding dimension.
        channels = int(out_channels)
        visual_channels = int(vpre_channels)
        visual_bases = 32
        bottleneck_blocks = int(num_blocks)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.compression = compression
        self.fft_bins = n_fft // 2 + 1
        self.num_sources = num_sources
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self._model_args = {
            "out_channels": out_channels,
            "in_channels": in_channels,
            "vpre_channels": vpre_channels,
            "vin_channels": vin_channels,
            "vout_channels": vout_channels,
            "num_blocks": num_blocks,
            "upsampling_depth": upsampling_depth,
            "enc_kernel_size": enc_kernel_size,
            "num_sources": num_sources,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "compression": compression,
        }

        # Stored only for config traceability. They are part of the CAMB-Net
        # interface but are not needed by this low-rank subspace design.
        self.in_channels = in_channels
        self.vin_channels = vin_channels
        self.vout_channels = vout_channels
        self.upsampling_depth = upsampling_depth

        self.register_buffer("hann_window", torch.hann_window(n_fft), persistent=False)

        # Input encoders.
        self.visual_prior = VisualToSpectralPrior(
            visual_channels=visual_channels,
            channels=channels,
            fft_bins=self.fft_bins,
            num_bases=visual_bases,
        )
        self.magnitude_encoder = EncoderNoDownsample(1, channels, depth=4)
        self.mp_encoder = EncoderDownsampleFreq(3, channels, depth=4)

        # Main AV interaction and complex spectral bottleneck.
        # This is the paper-critical module: it produces the visual-conditioned
        # low-rank target anchor, orthogonal residual, and residual target details.
        self.av_encoder = VisualConditionedLowRankAVEncoder(
            channels,
            num_bands=16,
            rank=16,
            ridge=1e-3,
            num_refine_blocks=4,
        )
        self.av_downsample = FreqDownsample(channels)
        self.bottleneck_fusion = SubspaceGuidedComplexBottleneck(channels, gain_channels=channels)

        dilations = [1, 2, 4, 8]
        tf_blocks = []
        for i in range(bottleneck_blocks):
            tf_blocks.append(TFAxisBlock(channels, dilation=dilations[i % len(dilations)]))
        self.bottleneck = nn.Sequential(*tf_blocks)

        # Output decoders.  The magnitude and phase heads share the expensive
        # frequency upsampling step to reduce runtime while keeping separate
        # task-specific refinement heads for performance.
        self.decoder_upsample = FreqUpsample(channels)
        self.magnitude_decoder = SharedResidualMagnitudeDecoder(channels)
        self.phase_rotator = SharedResidualPhaseRotator(channels)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Stable default initialization for newly trained models."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Skip modules intentionally zero-initialized.
                if torch.count_nonzero(module.weight).item() == 0:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                if torch.count_nonzero(module.weight).item() == 0:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Re-apply stability-specific bias for the phase residual rotation head.
        nn.init.constant_(self.phase_rotator.gamma_head.bias, -1.0)

    @staticmethod
    def _prepare_waveform(input_wav: Tensor) -> Tensor:
        if input_wav.ndim == 1:
            return input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            return input_wav
        if input_wav.ndim == 3 and input_wav.size(1) == 1:
            return input_wav.squeeze(1)
        raise ValueError(f"Expected waveform shape (T), (B,T), or (B,1,T), got {tuple(input_wav.shape)}.")

    def _stft(self, wav: Tensor) -> Tensor:
        window = self.hann_window.to(device=wav.device, dtype=wav.dtype)
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            onesided=True,
            return_complex=True,
        )

    def _istft(self, spec: Tensor, length: int) -> Tensor:
        window = self.hann_window.to(device=spec.device, dtype=spec.real.dtype)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            length=length,
        )

    def forward(self, input_wav: Tensor, mouth_emb: Tensor, return_intermediates: bool = False):
        """Run end-to-end AVSS.

        Args:
            input_wav: (T), (B,T), or (B,1,T).
            mouth_emb: (B,Cv,Tv) or (B,Cv,Tv,1), usually lip/face features.
            return_intermediates: if True, additionally return subspace and
                residual-decomposition maps for visualization. This does not
                affect training.

        Returns:
            denoised_mag: (B,F,T), compressed-domain target magnitude.
            denoised_phase: (B,F,T), target phase estimate.
            est_spec_uncompress: (B,F,T,2), uncompressed real/imag spectrogram for the original matrix loss.
            estimated_waveforms: (B,1,L), waveform estimate.
            intermediates: optional dict with evidence maps.
        """
        input_wav = self._prepare_waveform(input_wav)
        original_num_samples = input_wav.size(-1)

        stft = self._stft(input_wav)
        _, compressed_mag, mixture_phase = power_compress(stft, compression=self.compression)
        target_frames = compressed_mag.size(-1)

        # Convert STFT tensors from (B,F,T) to feature maps (B,C,T,F).
        audio_mag_input = compressed_mag.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,F)
        cos_phase = torch.cos(mixture_phase).permute(0, 2, 1).unsqueeze(1)
        sin_phase = torch.sin(mixture_phase).permute(0, 2, 1).unsqueeze(1)
        mp_input = torch.cat([audio_mag_input, audio_mag_input * cos_phase, audio_mag_input * sin_phase], dim=1)

        audio_feat = self.magnitude_encoder(audio_mag_input)  # (B,C,T,F)
        visual_feat = self.visual_prior(mouth_emb, target_frames=target_frames)  # (B,C,T,F)
        f_av, evidence = self.av_encoder(audio_feat, visual_feat, return_intermediates=return_intermediates)  # (B,C,T,F)

        f_av_down = self.av_downsample(f_av)  # (B,C,T,F/2)
        f_mp = self.mp_encoder(mp_input)  # (B,C,T,F/2)
        h, bottleneck_gate = self.bottleneck_fusion(f_av_down, f_mp, evidence["gain"])
        h = self.bottleneck(h)

        # Shared full-resolution decoder feature.  This avoids running two
        # separate ConvTranspose2d+Norm+PReLU upsampling paths for magnitude and
        # phase, which is a safe speed-up because both branches consume the same
        # bottleneck representation before their task-specific heads.
        h_up = self.decoder_upsample(h, target_freq=compressed_mag.size(1))
        denoised_mag = self.magnitude_decoder(h_up, compressed_mag)
        denoised_phase = self.phase_rotator(h_up, mixture_phase)

        denoised_complex = torch.complex(
            denoised_mag * torch.cos(denoised_phase),
            denoised_mag * torch.sin(denoised_phase),
        )
        enhanced_complex = power_uncompress(denoised_complex, compression=self.compression)
        estimated_waveforms = self._istft(enhanced_complex, length=original_num_samples).unsqueeze(1)

        # The original VLTS/CAMB-Net loss in vlts_net/losses/matrix.py expects
        # an uncompressed spectrogram in real-imaginary layout, i.e. (B, F, T, 2),
        # and indexes it as spec[..., 0] and spec[..., 1].  Returning PyTorch's
        # complex tensor directly would have shape (B, F, T) and will trigger
        # "too many indices for tensor of dimension 3" inside the loss.
        est_spec_uncompress = torch.view_as_real(enhanced_complex.contiguous())

        if not return_intermediates:
            return denoised_mag, denoised_phase, est_spec_uncompress, estimated_waveforms

        intermediates = {
            **evidence,
            "bottleneck_gate": bottleneck_gate,
            "visual_feature": visual_feat,
            "audio_feature": audio_feat,
        }
        return denoised_mag, denoised_phase, est_spec_uncompress, estimated_waveforms, intermediates

    def get_model_args(self) -> Dict[str, Union[int, float]]:
        return dict(self._model_args)


# -----------------------------------------------------------------------------
# Loss helper. This is optional and does not introduce an auxiliary objective.
# -----------------------------------------------------------------------------


def negative_si_snr_loss(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-invariant SNR loss for waveform-level training.

    estimate/target: (B,T) or (B,1,T). Use this as the only loss if you want to
    follow the no-extra-loss setting discussed for the paper.
    """
    if estimate.ndim == 3:
        estimate = estimate.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    proj = (torch.sum(estimate * target, dim=-1, keepdim=True) * target) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    noise = estimate - proj
    si_snr = 10.0 * torch.log10((torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps))
    return -si_snr.mean()
