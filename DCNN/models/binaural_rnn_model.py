import torch

from .model_mse import DCNN


class BinauralDCNN(DCNN):
    def forward(self, inputs):
        # batch_size, binaural_channels, time_bins = inputs.shape

        # inputs = inputs.flatten(end_dim=1)
        # output = super().forward(inputs)
        # output = output.unflatten(0, (batch_size, binaural_channels))
        output_left = super().forward(inputs[:, 0])
        output_right = super().forward(inputs[:, 1])

        output = torch.stack([output_left, output_right], dim=1)

        return output
