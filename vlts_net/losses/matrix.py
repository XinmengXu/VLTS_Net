###
# Author: Kai Li
# Date: 2021-06-09 16:43:09
# LastEditors: Please set LastEditors
# LastEditTime: 2021-12-03 17:52:13
###
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class PairwiseNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, denoised_mag, denoised_pha, est_spec_uncompress, estimated_waveforms, targets):
        targets_1 = torch.stft(
            targets.squeeze(1),
            512,
            256,
            window=torch.hann_window(512).to(targets),
            onesided=True,
            return_complex=False
            )
        ests_spec,_,_ = power_compress(est_spec_uncompress)  
        ests_mag = denoised_mag
        ests_phase = denoised_pha
        targets_spec, targets_mag, targets_phase = power_compress(targets_1)           
        ests = estimated_waveforms
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        assert targets.size() == ests.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(ests, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
            loss_wav = torch.abs(ests - targets).mean(dim=-1, keepdim=True) + self.EPS
            loss_mag = (((ests_mag - targets_mag) ** 2).mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)) + self.EPS
            loss_ri = (((est_spec_uncompress[:, :, :, 0] - targets_1[:, :, :, 0].float()) ** 2).mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + self.EPS +
           ((est_spec_uncompress[:, :, :, 1] - targets_1[:, :, :, 1].float()) ** 2).mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + self.EPS)
            loss_ip, loss_gd, loss_iaf = phase_losses(ests_phase, targets_phase)#.item()
            loss_pha = (loss_ip + loss_gd + loss_iaf + self.EPS).to(targets)	
            #loss = 0.1*(2 *  loss_wav + 9 * loss_mag + 1 * loss_ri + 3 * loss_pha) - 0.1 * pair_wise_sdr
            loss = (9 * loss_mag + 1 * loss_ri + 3 * loss_pha) - 0.25 * pair_wise_sdr

            if torch.isnan(loss_wav).any():
                print("NaN detected in loss_wav.")
            if torch.isnan(loss_mag).any():
                print("NaN detected in loss_mag.")
            if torch.isnan(loss_ri).any():
                print("NaN detected in loss_ri.")
            if torch.isnan(loss_ip).any() or torch.isnan(loss_gd).any() or torch.isnan(loss_iaf).any():
                print("NaN detected in phase_losses.")
        # print("sisnr:", -pair_wise_sdr)
        # print("maphaloss:",(9 * loss_mag + 1 * loss_ri + 3 * loss_pha))
        return loss#

class PairwiseNegSDR1(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, denoised_mag, denoised_pha, est_spec_uncompress, estimated_waveforms, targets):
        targets_1 = torch.stft(
            targets.squeeze(1),
            512,
            256,
            window=torch.hann_window(512).to(targets),
            onesided=True,
            return_complex=False
            )
        ests_spec,_,_ = power_compress(est_spec_uncompress)  
        ests_mag = denoised_mag
        ests_phase = denoised_pha
        targets_spec, targets_mag, targets_phase = power_compress(targets_1)           
        ests = estimated_waveforms
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        assert targets.size() == ests.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(ests, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
            loss_wav = torch.abs(ests - targets).mean(dim=-1, keepdim=True) + self.EPS
            loss_mag = (((ests_mag - targets_mag) ** 2).mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)) + self.EPS
            loss_ri = (((ests_spec[:, 0, :, :] - targets_spec[:, 0, :, :]) ** 2).mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + self.EPS +
           ((ests_spec[:, 1, :, :] - targets_spec[:, 1, :, :]) ** 2).mean(dim=-1, keepdim=True).mean(dim=1, keepdim=True) + self.EPS)
            loss_ip, loss_gd, loss_iaf = phase_losses(ests_phase, targets_phase)#.item()
            loss_pha = (loss_ip + loss_gd + loss_iaf + self.EPS).to(targets)	
            loss = 2 *  loss_wav + 9 * loss_mag + 1 * loss_ri + 3 * loss_pha - pair_wise_sdr
            if torch.isnan(loss_wav).any():
                print("NaN detected in loss_wav.")
            if torch.isnan(loss_mag).any():
                print("NaN detected in loss_mag.")
            if torch.isnan(loss_ri).any():
                print("NaN detected in loss_ri.")
            if torch.isnan(loss_ip).any() or torch.isnan(loss_gd).any() or torch.isnan(loss_iaf).any():
                print("NaN detected in phase_losses.")
        #print(-pair_wise_sdr)
        return -pair_wise_sdr

class SingleSrcNegSDR(_Loss):
    def __init__(
        self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8
    ):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        targets = targets.squeeze(1)
        ests = ests.squeeze(1)
        if targets.size() != ests.size() or targets.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=1, keepdim=True)
            mean_estimate = torch.mean(ests, dim=1, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(ests * targets, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * targets / s_target_energy
        else:
            # [batch, time]
            scaled_target = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.EPS
        )
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_est = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_est
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(ests * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)

def power_compress(x):
    real = x[..., 0].float()
    imag = x[..., 1].float()
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = torch.pow(mag, 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1), mag, phase

# aliases
pairwise_neg_sisdr = PairwiseNegSDR1("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")

def phase_losses(phase_r, phase_g):
    dim_freq = 512 // 2 + 1
    dim_time = phase_r.size(-1)
    eps = 1e-8
    # 鐢熸垚 gd_matrix 鍜?iaf_matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    # 浣跨敤 anti_wrapping_function锛屽苟淇濇寔褰㈢姸 (4, 1, 1)
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g), dim=(1, 2), keepdim=True) + eps
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g), dim=(1, 2), keepdim=True) + eps
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g), dim=(1, 2), keepdim=True) + eps

    # 妫€鏌ユ槸鍚︽湁 NaN
    if torch.isnan(ip_loss).any() or torch.isnan(gd_loss).any() or torch.isnan(iaf_loss).any():
        print("NaN detected in loss calculation.")
    
    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    # 浣跨敤杈冨ぇ鐨勫亸绉绘潵绋冲畾闄ゆ硶锛屼笖浣跨敤 torch.clamp 闄愬埗鑼冨洿
    #offset = 1e-6
    #wrapped_x = x - torch.round((x + offset) / (2 * np.pi)) * 2 * np.pi
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) + 1e-8
