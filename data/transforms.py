import torch
import numpy as np
import sys
from typing import Union, Optional, Tuple, Sequence, NamedTuple

from .masking import MaskFunc

sys.path.append('..')
from utils.coils import rss
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, center_crop, normalize, normalize_instance

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



class OasisSample(NamedTuple):
    image: torch.Tensor
    target: torch.Tensor
    mean: float
    std: float
    fname: str
    slice_num: int
    metadata: dict
    max_value: float



class OasisDataTransform:
    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True) -> None:

        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(self, kspace: torch.Tensor, target: torch.Tensor, fname: str, slice_num: int, metadata: dict, mask: np.ndarray):
        """
        Args:
            kspace (np.ndarray): Input k-space of shape (num_coils, rows, cols) for multi-coil data
                or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            target (np.ndarray): Target image.
            fname (str): File name.
            slice_num (int): slice index.

        Returns: A tuple containing,
            image: zero-filled input image,
            output: the reconstruction
            target: target,
            mean: the mean used for normalization,
            std: the standard deviations used for normalization,
            fname: the filename,
            slice_num: and the slice number.
        """

        kspace = torch.fft.fftshift(kspace, dim=(-3, -2))
        max_value = 255.0
        # ! 1 apply mask
        if mask is None and self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask, _ = apply_mask(kspace, self.mask_func, seed=seed)
        else:
            masked_kspace = kspace * mask + 0.0

        # ! 2 get input images
        shift_masked_kspace = torch.fft.ifftshift(masked_kspace, dim=[-3, -2])
        input_masked_image = torch.view_as_real(
            torch.fft.ifftn(
                torch.view_as_complex(shift_masked_kspace), dim=(-2, -1), norm="ortho"
            )
        )
        input_masked_image = complex_abs(input_masked_image)

        # ! 3 normalize input image
        input_masked_image, mean, std = normalize_instance(input_masked_image, eps=1e-11)

        # ! 4 normalize target
        if target is not None:
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = complex_abs(normalize(target, mean, std, eps=1e-11))
            target = target.clamp(-6, 6)

        return OasisSample(
            image=input_masked_image.float(),
            target=target.float(),
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            metadata=metadata,
            max_value=max_value
        )


class UnetSample(NamedTuple):
    
    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    
    
class UnetDataTransform:
    def __init__(self, challenge:str, mask_func: Optional[MaskFunc]=None, use_seed: bool=True, noise_lvl: float=0.0) -> None:
        assert challenge in ('singlecoil', 'multicoil'), f'Challenge should be either singlecoil or multicoil, got {challenge}'
        
        self.mask_func = mask_func
        self.challenge = challenge
        self.use_seed = use_seed
        self.noise_lvl = noise_lvl
        
    def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: dict, fname: str, slice_num:int):
        """

        Args:
            kspace (np.ndarray): Input k-space of shape (num_coils, rows, cols) for multi-coil data
                or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            target (np.ndarray): Target image.
            attrs (Dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name.
            slice_num (int): slice index.
            
        Returns: A tuple containing, 
            image: zero-filled input image, 
            output: the reconstruction
            target: target, 
            mean: the mean used for normalization, 
            std: the standard deviations used for normalization, 
            fname: the filename, 
            slice_num: and the slice number.
        """
        # # !1 convert to tensor
        # kspace_torch = to_tensor(kspace)
        # max_value = attrs["max"] if "max" in attrs else 0.0
        
        # # !2 apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        # else:
        #     masked_kspace = kspace_torch
            
        # # !3 apply ifft2 to get zero filled solution
        # image = ifft2c(masked_kspace)
        
        # # !4 crop input image to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
            
        # # ?check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])
            
        # image = complex_center_crop(image, crop_size)
        
        # # !5 absolute value
        # image = complex_abs(image)
        
        # if self.challenge == 'multicoil':
        #     image = rss(image)
            
        # # !6 normalize input image
        # image, mean, std = normalize_instance(image, eps=1e-11)
        # image = image.clamp(-6, 6)
        
        # # !7 normalize target
        # if target is not None:
        #     target_torch = center_crop(to_tensor(target), crop_size)
        #     target_torch = normalize(target_torch, mean, std, eps=1e-11)
        #     target_torch = target_torch.clamp(-6, 6)
        # else:
        #     target_torch = torch.tensor([0])

        # !1 convert to tensor
        tensor_kspace = to_tensor(kspace)
        max_value = attrs["max"] if "max" in attrs else 0.0

        # !1.5 add noise
        if self.noise_lvl != 0.0:
            noisy_tensor_kspace = tensor_kspace + torch.randn_like(tensor_kspace) * self.noise_lvl + 0.0
        else:
            noisy_tensor_kspace = tensor_kspace

        
        #! 2 convert to image
        tensor_image = ifft2c(tensor_kspace)
        noisy_image = ifft2c(noisy_tensor_kspace)
        
        #! 3 crop input image to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
        # ?check for FLAIR 203
        if noisy_image.shape[-2] < crop_size[1]:
            crop_size = (noisy_image.shape[-2], noisy_image.shape[-2])
            
        cropped_clean_image = complex_center_crop(tensor_image, crop_size)
        cropped_noisy_image = complex_center_crop(noisy_image, crop_size)       
        
        #! 4 apply fft2 to get related k-space
        target_tensor_kspace = fft2c(cropped_clean_image)
        noisy_tensor_kspace = fft2c(cropped_noisy_image)
        
        #! 5 apply mask
        if mask is None and self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            noisy_masked_kspace, mask_, _ = apply_mask(noisy_tensor_kspace, self.mask_func, seed=seed)
            masked_kspace = target_tensor_kspace * mask_ + 0.0
        else:
            mask_ = mask
            noisy_masked_kspace = noisy_tensor_kspace * mask_ + 0.0
            masked_kspace = target_tensor_kspace * mask_ + 0.0
            
        #! 6 get input images
        input_noisy_image = complex_abs(ifft2c(noisy_masked_kspace))
        input_zf_image = ifft2c(masked_kspace)
        target_image = ifft2c(target_tensor_kspace) 
        
        #! 7 normalize input image
        input_noisy_image, mean, std = normalize_instance(input_noisy_image, eps=1e-11)
        input_zf_image = normalize(input_zf_image, mean, std, eps=1e-11)
        
        #! 6 normalize target
        if target is not None:
            target_torch = center_crop(to_tensor(target), crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = complex_abs(normalize(target_image, mean, std, eps=1e-11))
            target_torch = target_torch.clamp(-6, 6)
        
            
        return UnetSample(
            image=input_noisy_image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value
        )
            

class UnrolledSample(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    image: torch.Tensor
    zf_image: torch.Tensor
    kspace: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float


class DCDIDNDataTransform:
    def __init__(
        self,
        challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        noise_lvl: float = 0.0,
    ):
        assert challenge in ('singlecoil', 'multicoil'), f'Challenge should be either singlecoil or multicoil, got {challenge}'

        self.mask_func = mask_func
        self.challenge = challenge
        self.use_seed = use_seed
        self.noise_lvl = noise_lvl
        
    def __call__(
        self, 
        kspace: np.ndarray, 
        mask: np.ndarray, 
        target: np.ndarray, 
        attrs: dict, 
        fname: str, 
        slice_num: int, 
        ):
        
        # !1 convert to tensor
        tensor_kspace = to_tensor(kspace)
        max_value = attrs["max"] if "max" in attrs else 0.0
        
        # !1.5 add noise
        if self.noise_lvl != 0.0:
            noisy_tensor_kspace = tensor_kspace + torch.randn_like(tensor_kspace) * self.noise_lvl + 0.0
        else:
            noisy_tensor_kspace = tensor_kspace
        
        #! 2 convert to image
        tensor_image = ifft2c(tensor_kspace)
        noisy_image = ifft2c(noisy_tensor_kspace)
        
        #! 3 crop input image to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
        # ?check for FLAIR 203
        if noisy_image.shape[-2] < crop_size[1]:
            crop_size = (noisy_image.shape[-2], noisy_image.shape[-2])
            
        cropped_clean_image = complex_center_crop(tensor_image, crop_size)
        cropped_noisy_image = complex_center_crop(noisy_image, crop_size)       
        
        #! 4 apply fft2 to get related k-space
        target_tensor_kspace = fft2c(cropped_clean_image)
        noisy_tensor_kspace = fft2c(cropped_noisy_image)
        
        #! 5 apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            noisy_masked_kspace, mask_, _ = apply_mask(noisy_tensor_kspace, self.mask_func, seed=seed)
            masked_kspace = target_tensor_kspace * mask_ + 0.0
        else:
            mask_ = mask
            noisy_masked_kspace = noisy_tensor_kspace * mask_ + 0.0
            masked_kspace = target_tensor_kspace * mask_ + 0.0
            
        #! 6 get input images
        input_noisy_image = ifft2c(noisy_masked_kspace)
        input_zf_image = ifft2c(masked_kspace)
        target_image = ifft2c(target_tensor_kspace) 
        
        #! 7 normalize input image
        input_noisy_image, mean, std = normalize_instance(input_noisy_image, eps=1e-11)
        input_zf_image = normalize(input_zf_image, mean, std, eps=1e-11)
        
        #! 6 normalize target
        target_image = normalize(target_image, mean, std, eps=1e-11)
        clean_kspace = fft2c(target_image)

        # print("image", input_noisy_image.shape, input_noisy_image.dtype)
        # print("zf_image", input_zf_image.shape, input_zf_image.dtype)
        # print("kspace", clean_kspace.shape, clean_kspace.dtype)
        # print("mask", mask_.shape, mask_.dtype)
        # print("target", target_image.shape, target_image.dtype)
        # print("fname", fname)
        # print("slice_num", slice_num)
        # print("mean", mean)
        # print("std", std)
        # print("max_value", max_value)
        return UnrolledSample(
            image=input_noisy_image, # noisy input image 
            zf_image = input_zf_image, # input image without noise
            kspace = clean_kspace, # masked kspace without noise
            mask = mask_, # mask
            target = target_image, # clean target 
            fname=fname,
            slice_num=slice_num,
            mean=mean,
            std=std,
            max_value=max_value
        )
            
        
            
            
            
        
    