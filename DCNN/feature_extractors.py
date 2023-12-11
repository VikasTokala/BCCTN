import torch
import torch.nn as nn
import nnAudio

class Stft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, win_length=None,
                 onesided=True, is_complex=True):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = n_dft if win_length is None else win_length
        self.onesided = onesided
        self.is_complex = is_complex

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
                       win_length=self.win_length, onesided=self.onesided,
                       return_complex=True, window=window, normalized=True)
        
        y = y[:, 1:] # Remove DC component (f=0hz)

        # y.shape == (batch_size*channels, time, freqs)

        if not self.is_complex:
            y = torch.view_as_real(y)
            y = y.movedim(-1, 1) # move complex dim to front

        return y


class IStft(Stft):

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        window=window,normalized=True)

        return y

