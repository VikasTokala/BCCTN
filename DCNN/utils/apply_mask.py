import torch    
import torch.nn.functional as F


def apply_mask(x, specs, masking_mode="E"):
    real, imag = specs.real, specs.imag
    mask_real, mask_imag = x.real, x.imag

    # mask_real = F.pad(mask_real, [0, 0, 1, 0])
    # mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

    if masking_mode == "E":
        out_spec = x.abs()*specs.abs()*torch.exp(1j*(x.angle() + specs.angle()))
        out_spec = F.pad(out_spec, [0, 0, 1, 0])
        return out_spec

    elif masking_mode == "C":
        real, imag = real * mask_real - imag * \
            mask_imag, real * mask_imag + imag * mask_real
    elif masking_mode == "R":
        real, imag = real * mask_real, imag * mask_imag

    # Pad DC component, which was removed
    real = F.pad(real, [0, 0, 1, 0])
    imag = F.pad(imag, [0, 0, 1, 0])

    # Generate output signal
    out_spec = torch.complex(real, imag)

    return out_spec
