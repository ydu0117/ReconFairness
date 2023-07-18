import torch
import h5py
from typing import Optional, Sequence, Tuple, Union
import sys
sys.path.append('../../')

from data.masking import MaskFunc, create_mask_for_mask_type
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, center_crop, normalize, normalize_instance

from tensorboardX import SummaryWriter

writer = SummaryWriter()

def apply_mask(data: torch.Tensor, mask_func: MaskFunc, offset: Optional[int]=None, 
                seed: Optional[Union[int, Tuple[int, ...]]]=None, padding: Optional[Sequence[int]]=None):
    
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:]) # (1, 1, 640, 372, 2)
    mask, num_low_freqs = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., :padding[0], :] = 0
        mask[..., padding[1]:, :] = 0
        
    # * add 0.0 removes the sign of the zeros    
    masked_data = data * mask + 0.0 
    
    return masked_data, mask, num_low_freqs

file_name = 'file1002515.h5'

hf = h5py.File(file_name, 'r')
kspace = hf['kspace'][25]
target = hf['reconstruction_esc'][25]


# !1
tensor_kspace = to_tensor(kspace)
target_tensor = to_tensor(target / target.max())

#writer.add_image("target", target_tensor.unsqueeze(0))
tensor_kspace_vis = complex_abs(torch.log(tensor_kspace/tensor_kspace.max()))
#writer.add_image("tensor_kspace", tensor_kspace_vis.unsqueeze(0))

#! 1.5 add noise
noise_lvl = 1e-5
if noise_lvl != 0.0:
    noisy_tensor_kspace = tensor_kspace + torch.randn_like(tensor_kspace) * noise_lvl + 0.0
else:
    noisy_tensor_kspace = tensor_kspace

noisy_tensor_kspace_vis = complex_abs(torch.log(noisy_tensor_kspace/noisy_tensor_kspace.max()))
#writer.add_image("noisy_tensor_kspace", noisy_tensor_kspace_vis.unsqueeze(0))

#! 2 convert to image domain
raw_image = ifft2c(tensor_kspace)
noisy_raw_image = ifft2c(noisy_tensor_kspace)

raw_image_vis = complex_abs(raw_image / raw_image.max())
#writer.add_image("raw_image", raw_image_vis.unsqueeze(0))
noisy_raw_image_vis = complex_abs(noisy_raw_image / noisy_raw_image.max())
#writer.add_image("noisy_raw_image", noisy_raw_image_vis.unsqueeze(0))

#! 3 crop


crop_size = (target.shape[-2], target.shape[-1])

cropped_raw_image = complex_center_crop(raw_image, crop_size)
cropped_noisy_raw_image = complex_center_crop(noisy_raw_image, crop_size)

cropped_raw_image_vis = complex_abs(cropped_raw_image / cropped_raw_image.max())
#writer.add_image("cropped_raw_image", cropped_raw_image_vis.unsqueeze(0))
cropped_noisy_raw_image_vis = complex_abs(cropped_noisy_raw_image / cropped_noisy_raw_image.max())
#writer.add_image("cropped_noisy_raw_image", cropped_noisy_raw_image_vis.unsqueeze(0))

target = cropped_raw_image

#! 4 convert back to kspace
cropped_raw_tensor = fft2c(cropped_raw_image)
cropped_noisy_raw_tensor = fft2c(cropped_noisy_raw_image)

cropped_raw_tensor_vis = complex_abs(torch.log(cropped_raw_tensor/cropped_raw_tensor.max()))
#writer.add_image("cropped_raw_tensor", cropped_raw_tensor_vis.unsqueeze(0))
cropped_noisy_raw_tensor_vis = complex_abs(torch.log(cropped_noisy_raw_tensor/cropped_noisy_raw_tensor.max()))
#writer.add_image("cropped_noisy_raw_tensor", cropped_noisy_raw_tensor_vis.unsqueeze(0))

mask = None

#! 5 apply mask
if mask is None:
    seed = None if not True else tuple(map(ord, file_name))
    noisy_masked_kspace, mask_, _ = apply_mask(cropped_noisy_raw_tensor, 
                                                create_mask_for_mask_type("random",[0.08], [4]),
                                                seed=seed)
    masked_kspace = cropped_raw_tensor * mask_ + 0.0
    mask = mask_
else:
    noisy_masked_kspace = cropped_noisy_raw_tensor * mask + 0.0
    masked_kspace = cropped_raw_tensor * mask + 0.0

masked_kspace_vis = complex_abs(torch.log(masked_kspace/masked_kspace.max()))
#writer.add_image("masked_kspace", masked_kspace_vis.unsqueeze(0))    

noisy_masked_kspace_vis = complex_abs(torch.log(noisy_masked_kspace/noisy_masked_kspace.max()))
#writer.add_image("noisy_masked_kspace", noisy_masked_kspace_vis.unsqueeze(0))

mask_vis = torch.tile(mask_, [320, 1, 1]).permute(2, 0, 1)
#writer.add_image("mask", mask_vis)

#! 6 get input
noisy_image = ifft2c(noisy_masked_kspace)
masked_image = ifft2c(masked_kspace)

input_noisy_image, mean, std = normalize_instance(noisy_image, eps=1e-11)

noisy_image_vis = complex_abs(input_noisy_image / input_noisy_image.max())
#writer.add_image("noisy_image", noisy_image_vis.unsqueeze(0))
masked_image_vis = complex_abs(masked_image / masked_image.max())
#writer.add_image("masked_image", masked_image_vis.unsqueeze(0))
input_noisy_image_vis = complex_abs(input_noisy_image / input_noisy_image.max())
#writer.add_image("input_noisy_image", input_noisy_image_vis.unsqueeze(0))

#! 7 normalize target
target = normalize(target, mean, std)


#! 8 DC
lambda_ = torch.tensor([0.5])
print(lambda_.shape)
A_x = fft2c(input_noisy_image)
A_x_mask = (1 - mask) * 
k_dc = (1 - mask) * A_x + mask * (lambda_ * A_x + (1 - lambda_) * cropped_raw_tensor)
x_dc = ifft2c(k_dc)
print(input_noisy_image.shape, cropped_raw_tensor.shape, mask.shape, lambda_.shape, A_x.shape, k_dc.shape, x_dc.shape)

x_dc_vis = complex_abs(x_dc / x_dc.max())
writer.add_image("x_dc", x_dc_vis.unsqueeze(0))
