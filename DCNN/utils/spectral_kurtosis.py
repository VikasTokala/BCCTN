import torch
import torch.nn as nn
import torchaudio.transforms as tt
from DCNN.feature_extractors import Stft, IStft


class Spectral_Kurtosis(nn.Module):
    def __init__(self, fs=16000) -> None:
        super().__init__()

        self.kurtosis = Kurtosis()
        self.fs = fs
        self.dBA = dBA_Torcolli(self.fs)

        # As our sampling rate is limited to 16kHz, the bands are reconfigured to these values
        # which is different from the original paper which uses 48kHz sampling rate.
        self.Lower_band = [50, 750]
        self.Middle_band = [750, 3500]
        self.Upper_band = [3500, 8000]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """" x - reference/target signal power spectrum
             y - test signal power spectrum
             Spectral Kurtosis based on Torcoli, M., "An Improved Measure of Musical Noise 
            Based on Spectral Kurtosis", 2019 IEEE Workshop on Applications of 
            Signal Processing to Audio and Acoustics, New Paltz, NY, 2019. """

        X_dBA = self.dBA(x)
        Y_dBA = self.dBA(y)

        # Removing the zero frames (dB = -Inf) and the 0 frequency DC bin

        X_dBA_nz = X_dBA[:, 1:, ~torch.isinf(Y_dBA.sum(1).sum(0))]
        Y_dBA_nz = Y_dBA[:, 1:, ~torch.isinf(Y_dBA.sum(1).sum(0))]
        # breakpoint()
        # Limiting and shifting the values to be non-negative
        thr_X = 10*torch.log10((torch.mean(10**(X_dBA_nz/10)))) - 20
        X_plus = torch.max(X_dBA_nz, thr_X) - thr_X

        thr_Y = 10*torch.log10((torch.mean(10**(Y_dBA_nz/10)))) - 20
        Y_plus = torch.max(Y_dBA_nz, thr_Y) - thr_Y

        # Discarding the frames for which N_out = 0 for all bins
        # breakpoint()
        X_plus_nz = X_plus[:, :, torch.nonzero(Y_plus.sum(1).sum(0))].squeeze()
        Y_plus_nz = Y_plus[:, :, torch.nonzero(Y_plus.sum(1).sum(0))].squeeze()

        # Calculating frequeny for each bin
        _, freq_bins, _ = X_plus_nz.shape
        f = (torch.arange(1, freq_bins, device=x.device)/freq_bins) * self.fs/2

        Lower_band_bins = [0, 0]
        Middle_band_bins = [0, 0]
        Upper_band_bins = [0, 0]

        # Calculating closest bins numbers to the band frequencies
        # breakpoint()
        Lower_band_bins[0] = torch.argmin((f - self.Lower_band[0]).abs())
        Lower_band_bins[1] = torch.argmin((f - self.Lower_band[1]).abs())
        Middle_band_bins[0] = torch.argmin((f - self.Middle_band[0]).abs())
        Middle_band_bins[1] = Middle_band_bins[0] + 1
        # Plus 1 to remove overlapping regions
        Middle_band_bins[1] = torch.argmin((f - self.Middle_band[1]).abs())
        Upper_band_bins[0] = torch.argmin((f - self.Upper_band[0]).abs())
        # Plus 1 to remove overlapping regions
        Upper_band_bins[1] = Upper_band_bins[0]+1
        Upper_band_bins[1] = torch.argmin((f - self.Upper_band[1]).abs())

        # Splitting the data into sub-bands

        X_lower = X_plus_nz[:, Lower_band_bins[0]:Lower_band_bins[1], :]
        X_middle = X_plus_nz[:, Middle_band_bins[0]:Middle_band_bins[1], :]
        X_upper = X_plus_nz[:, Upper_band_bins[0]:Upper_band_bins[1], :]
        Y_lower = Y_plus_nz[:, Lower_band_bins[0]:Lower_band_bins[1], :]
        Y_middle = Y_plus_nz[:, Middle_band_bins[0]:Middle_band_bins[1], :]
        Y_upper = Y_plus_nz[:, Upper_band_bins[0]:Upper_band_bins[1], :]

        # Computing the Sub-Band A-weighted Kurtosis

        X_lower_kurt = self.kurtosis(X_lower)
        X_middle_kurt = self.kurtosis(X_middle)
        X_upper_kurt = self.kurtosis(X_upper)
        Y_lower_kurt = self.kurtosis(Y_lower)
        Y_middle_kurt = self.kurtosis(Y_middle)
        Y_upper_kurt = self.kurtosis(Y_upper)

        # Computing the Sub-Band log-Kurtosis ratio
        delta_lower_kurt = (torch.log(Y_lower_kurt/X_lower_kurt)).abs()
        delta_middle_kurt = (torch.log(Y_middle_kurt/X_middle_kurt)).abs()
        delta_upper_kurt = (torch.log(Y_upper_kurt/X_upper_kurt)).abs()

        # Limiting the values to 0.5
        delta_lower_kurt[delta_lower_kurt > 0.5] = 0.5
        delta_middle_kurt[delta_middle_kurt > 0.5] = 0.5
        delta_upper_kurt[delta_upper_kurt > 0.5] = 0.5

        # Removing the NaN values if any

        delta_lower_kurt[torch.isnan(delta_lower_kurt)] = 0
        delta_middle_kurt[torch.isnan(delta_middle_kurt)] = 0
        delta_upper_kurt[torch.isnan(delta_upper_kurt)] = 0

        # Calculating sub-band energy weights
        _, Kb_l, _ = Y_lower.shape
        _, Kb_m, _ = Y_middle.shape
        _, Kb_u, _ = Y_upper.shape

        w_lower = 10*torch.log10(1/Kb_l * torch.sum(10 ** (Y_lower/10),dim=1))
        w_middle = 10*torch.log10(1/Kb_m * torch.sum(10 ** (Y_middle/10),dim=1))
        w_upper = 10*torch.log10(1/Kb_u * torch.sum(10 ** (Y_upper/10),dim=1))

        # breakpoint()

        # Computing the energy weighted mean of the log-Kurtosis ratio using the band which
        # has max value

        W_lower = w_lower.sum()
        W_middle = w_middle.sum()
        W_upper = w_upper.sum()

        temp_lower = torch.sum(w_lower * delta_lower_kurt,dim=1) / W_lower
        temp_middle = torch.sum(w_middle * delta_middle_kurt,dim=1) / W_middle
        temp_upper = torch.sum(w_upper * delta_upper_kurt,dim=1) / W_upper

        Spec_Kurt = torch.stack((temp_lower, temp_middle, temp_upper))

        Spec_kurt = Spec_Kurt.max()

        # Spec_Kurt[torch.isnan(Spec_Kurt)] = 0

        # print('Spec_kurt', Spec_Kurt)
        # print('/n Lower-', temp_lower, ' Middle - ',
        #       temp_middle, ' Upper - ', temp_upper)

        # if torch.count_nonzero(Spec_Kurt) == 0:
        #     Spec_kurt_mean = 0

        # else:
        #     Spec_kurt_mean = Spec_Kurt.mean()

        # print('\n Mean', Spec_kurt_mean)

        return Spec_kurt


class Kurtosis(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor):

        _, K, _ = X.shape

        X_bar = (1/K) * torch.sum(X, dim=1).unsqueeze(1)

        num = (1/K) * torch.sum((X-X_bar)**4, dim=1)
        den = ((1/K)*torch.sum((X-X_bar)**2, dim=1)**2)
        # breakpoint()
        kurt = num/(den)
 
        return kurt


class dBA_Torcolli(nn.Module):
    def __init__(self, fs=16000):
        super().__init__()

        self.fs = fs
        # A-weighting filter co-efficients
        # self.c1 = 12194.217**2
        self.c2 = 20.598997**2
        self.c3 = 107.65265**2
        self.c4 = 737.86223**2
        self.c1 = 12194.217**2

    def forward(self, x: torch.Tensor):

        _, _, nbins = x.shape
        # breakpoint()
        # evaluation of A-weighting filter in the frequeny domain
        f2_ = (self.fs * 0.5 * (torch.arange(0, nbins, device=x.device))/nbins)
        f2 = f2_**2
        num = self.c1*(f2**2)
        # den = (torch.pow((self.c2 + f2),2)) * (self.c3 + f2) * (self.c4 +f2) * (torch.pow((self.c5 + f2),2))
        den = (f2 + self.c2) * torch.sqrt((f2+self.c3)
                                          * (f2+self.c4)) * (f2+self.c1)
        Aw = 1.2589 * num / den
        # breakpoint()
        # Converting to dBA
        # breakpoint()
        # breakpoint()

        dBA = 10*torch.log10(Aw * ((x.abs()) ** 2))

        # breakpoint()

        return dBA
