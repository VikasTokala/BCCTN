import torch
from DCNN.loss import BinaryMask
EPS= 10-6
import numpy as np






def ild_db(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2)

    return ild_value



def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()

    return ipd_value

def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=0)
    elif avg_mode == "time":
        return s.mean(dim=1)
    elif avg_mode == None:
        return s
    
def speechMask(stft_l,stft_r, threshold=15):
    # breakpoint()
    _,time_bins = stft_l.shape
    thresh_l,_ = (((stft_l.abs())**2)).max(dim=1) 
    thresh_l_db = 10*torch.log10(thresh_l) - threshold
    thresh_l_db=thresh_l_db.unsqueeze(1).repeat(1,1,time_bins)
    
    thresh_r,_ = (((stft_r.abs())**2)).max(dim=1) 
    thresh_r_db = 10*torch.log10(thresh_r) - threshold
    thresh_r_db=thresh_r_db.unsqueeze(1).repeat(1,1,time_bins)
    
    
    bin_mask_l = BinaryMask(threshold=thresh_l_db)
    bin_mask_r = BinaryMask(threshold=thresh_r_db)
    
    mask_l = bin_mask_l(20*torch.log10((stft_l.abs())))
    mask_r = bin_mask_r(20*torch.log10((stft_r.abs())))
    mask = torch.bitwise_and(mask_l.int(), mask_r.int())
    
    return mask





def gcc_phat_itd(stft_left, stft_right, target_stft_l, target_stft_r, fs=16000):
    """
    Calculate Interaural Time Difference (ITD) using GCC-PHAT from complex STFT.

    Args:
    stft_left (torch.Tensor): Complex STFT of the left channel with shape [freq_bins, time_frames].
    stft_right (torch.Tensor): Complex STFT of the right channel with shape [freq_bins, time_frames].
    fs (int): Sampling rate of the signal.

    Returns:
    torch.Tensor: Estimated ITD for each frequency bin in seconds.
    """
    # Calculate cross-correlation
    cross_corr = stft_left * torch.conj(stft_right)
    mask=speechMask(target_stft_l, target_stft_r,threshold=10)
    breakpoint()
    # Calculate GCC-PHAT
    gcc_phat = torch.fft.ifft(cross_corr / (torch.abs(cross_corr) + 1e-6), dim=-1)

    # Find peak index
    peak_indices = torch.argmax(torch.abs(gcc_phat), dim=-1)

    # Calculate time delay
    freq_bins = gcc_phat.shape[0]
    max_delay = stft_left.shape[-1] / (2 * fs)  # Maximum possible delay

    itd = (peak_indices - stft_left.shape[-1] // 2) * max_delay 

    return itd

# Example usage

