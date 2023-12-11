import torch
import torch.nn as nn
import DCNN.utils.complexPyTorch.complexLayers as torch_complex


class FAL_enc(torch.nn.Module):
    """This is an attention layer based on frequency transformation"""

    def __init__(self, in_channels, out_channels, f_length=256):
        super(FAL_enc, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_fal_r = 5  # Channels to be used within the FTB
        self.f_length = f_length

        self.amp_pre = nn.Sequential(
            torch_complex.ComplexConv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=(
                7, 1), stride=1, padding=[3, 0]),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU(),

            torch_complex.ComplexConv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(
                1, 7), stride=1, padding=[0, 3]),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()
        )

        self.conv_1_multiply_1_1 = nn.Sequential(
            torch_complex.ComplexConv2d(
                in_channels=self.out_channels, out_channels=self.c_fal_r, kernel_size=1, stride=1, padding=0),
            torch_complex.NaiveComplexBatchNorm2d(self.c_fal_r),
            torch_complex.ComplexReLU()
        )
        self.conv_1D = nn.Sequential(
            torch_complex.ComplexConv2d(self.f_length * self.c_fal_r, self.out_channels,
                                        kernel_size=(9, 1), stride=1, padding=(4, 0)),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()
        )
        self.frec_fc = torch_complex.ComplexLinear(
            self.f_length, self.f_length)
        self.conv_1_multiply_1_2 = nn.Sequential(
            torch_complex.ComplexConv2d(
                2 * self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()

        )
        self.conv_suf = torch_complex.ComplexConv2d(
            self.out_channels, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        bsize, ch, f_len, seg_length = inputs.shape
        inputs = inputs.reshape(bsize, ch, seg_length, f_len)
        # breakpoint()
        inputs = self.amp_pre(inputs)
        # breakpoint()

        x = self.conv_1_multiply_1_1(inputs)  # [B,c_ftb_r,segment_length,f]

        # [B,c_ftb_r*f,segment_length]
        x = x.view(-1, self.f_length * self.c_fal_r, seg_length)
        x = x.unsqueeze(-1)
        # breakpoint()
        x = self.conv_1D(x)  # [B,c_a,segment_length]

        # [B,c_a,segment_length,1]
        # x = x.view(-1, self.out_channels, seg_length, 1)
        # breakpoint()
        x = x * inputs  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]

        # x= x.reshape(-1, self.out_channels, seg_length,self.f_length)

        x = self.frec_fc(x)  # [B,c_a,segment_length,f]

        x = torch.cat((x, inputs), dim=1)  # [B,2*c_a,segment_length,f]

        outputs = self.conv_1_multiply_1_2(x)  # [B,c_a,segment_length,f]
        # breakpoint()
        # outputs = self.conv_suf(outputs)
        outputs = outputs.transpose(2, 3)
        return outputs


class FAL_dec(torch.nn.Module):
    """This is an attention layer based on frequency transformation"""

    def __init__(self, in_channels, out_channels, f_length=256):
        super(FAL_dec, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_fal_r = 5  # Channels to be used within the FTB
        self.f_length = f_length

        self.amp_pre = nn.Sequential(
            torch_complex.ComplexConvTranspose2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=(
                7, 1), stride=1, padding=[3, 0]),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU(),

            torch_complex.ComplexConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(
                1, 7), stride=1, padding=[0, 3]),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()
        )

        self.conv_1_multiply_1_1 = nn.Sequential(
            torch_complex.ComplexConvTranspose2d(
                in_channels=self.out_channels, out_channels=self.c_fal_r, kernel_size=1, stride=1, padding=0),
            torch_complex.NaiveComplexBatchNorm2d(self.c_fal_r),
            torch_complex.ComplexReLU()
        )
        self.conv_1D = nn.Sequential(
            torch_complex.ComplexConvTranspose2d(self.f_length * self.c_fal_r, self.out_channels,
                                        kernel_size=(9, 1), stride=1, padding=(4, 0)),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()
        )
        self.frec_fc = torch_complex.ComplexLinear(
            self.f_length, self.f_length)
        self.conv_1_multiply_1_2 = nn.Sequential(
            torch_complex.ComplexConvTranspose2d(
                2 * self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            torch_complex.NaiveComplexBatchNorm2d(self.out_channels),
            torch_complex.ComplexReLU()

        )
        self.conv_suf = torch_complex.ComplexConvTranspose2d(
            self.out_channels, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        bsize, ch, f_len, seg_length = inputs.shape
        inputs = inputs.reshape(bsize, ch, seg_length, f_len)
        # breakpoint()
        inputs = self.amp_pre(inputs)
        # breakpoint()

        x = self.conv_1_multiply_1_1(inputs)  # [B,c_ftb_r,segment_length,f]

        # [B,c_ftb_r*f,segment_length]
        x = x.view(-1, self.f_length * self.c_fal_r, seg_length)
        x = x.unsqueeze(-1)
        # breakpoint()
        x = self.conv_1D(x)  # [B,c_a,segment_length]

        # [B,c_a,segment_length,1]
        # x = x.view(-1, self.out_channels, seg_length, 1)
        # breakpoint()
        x = x * inputs  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]

        # x= x.reshape(-1, self.out_channels, seg_length,self.f_length)

        x = self.frec_fc(x)  # [B,c_a,segment_length,f]

        x = torch.cat((x, inputs), dim=1)  # [B,2*c_a,segment_length,f]

        outputs = self.conv_1_multiply_1_2(x)  # [B,c_a,segment_length,f]
        # breakpoint()
        # outputs = self.conv_suf(outputs)
        outputs = outputs.transpose(2, 3)
        return outputs