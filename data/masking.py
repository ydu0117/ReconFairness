import torch
import numpy as np

import contextlib

from typing import Optional, Sequence, Tuple, Union

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]] = None):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    
    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    
    When called, ``MaskFunc`` uses internal functions create mask by 
    1) creating a mask for the k-space center, 
    2) create a mask outside of the k-space center, and 
    3) combining them into a total mask. 
    The internals are handled by ``sample_mask``, which calls ``calculate_center_mask`` 
    for (1) and ``calculate_acceleration_mask`` for (2). 
    The combination is executed in the ``MaskFunc`` ``__call__`` function.
    """
    
    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int], allow_any_combination: bool = True, seed: Optional[int] = None):
        """

        Args:
            center_fractions (Sequence[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly each time.
            accelerations (Sequence[int]): Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided, then one of these is chosen uniformly each time.
            allow_any_combination (bool, optional): Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``. Defaults to True.
            seed (Optional[int], optional): Seed for starting the internal random number generator of the
                ``MaskFunc``. Defaults to None.
        """
        
        assert len(center_fractions) == len(accelerations), 'Number of center fractions should match the number of accelerations.'
        
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)
        
    def __call__(self, shape: Sequence[int], offset: Optional[int] = None, seed: Optional[Union[int, Tuple[int, ...]]] = None) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask

        Args:
            shape (Sequence[int]): Shape of the mask to be created.
            offset (Optional[int], optional): Offset from 0 to begin mask. If no offset is given, 
                then one is selected randomly. Defaults to None.
            seed (Optional[Union[int, Tuple[int, ...]]], optional): seed for RNG for reproducibility . Defaults to None.

        Returns:
            Tuple[torch.Tensor, int]: A 2-tuple containing
            1) the k-space mask and
            2) the number of center frequency lines
        """
        
        assert len(shape) >= 3, "Shape should have 3 or more dimensions"
        
        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(shape, offset)
            
        return torch.max(center_mask, accel_mask), num_low_frequencies
    
    def choose_acceleration(self):
        # random combination
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(self.accelerations)
        # choose according to the same index
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]
    
    def calculate_center_mask(self, shape: Sequence[int], num_low_freqs: int):
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs, "Center fractions should be equal to the number of low frequencies"
        
        return mask
    
    def calculate_acceleration_mask(self, num_cols, acceleration, offset, num_low_freqs):
        
        raise NotImplementedError
    
    def reshape_mask(self, mask, shape):
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols # [1, width, 1]
        
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
    
    def sample_mask(self, shape: Sequence[int], offset: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """_summary_

        Args:
            shape (Sequence[int]): Shape of the k-space to subsample.
            offset (Optional[int], optional): Offset from 0 to begin mask (for equispaced masks). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: A 3-tuple containing 
            1) the mask for the center of k-space, 
            2) the mask for the high frequencies of k-space, and 
            3) the integer count of low frequency samples.
        """
        
        num_cols = shape[-2] # width
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(num_cols, acceleration, offset, num_low_frequencies), shape,
        )
        
        return center_mask, acceleration_mask, num_low_frequencies
    
    
class RandomMaskFunc(MaskFunc):
    """
    
        The mask selects a subset of columns from the input k-space data. If the
        k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
            corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: 
        prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs). 
        This ensures that the expected number of columns selected is equal to (N / acceleration).
    
    """
    def calculate_acceleration_mask(self, num_cols, acceleration, offset, num_low_freqs):
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        
        return self.rng.uniform(size=num_cols) < prob
    
    
class EquispacedMaskFunc(MaskFunc):
    
    def calculate_acceleration_mask(self, num_cols, acceleration, offset, num_low_freqs):
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))
            
        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1
        
        return mask
        
        
def create_mask_for_mask_type(mask_type, center_fractions, accelerations):
    if mask_type == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        return ValueError(f"Unrecognized mask type {mask_type}")
    
    
def mask_center(x: torch.Tensor, mask_from: int, mask_to:int):
    
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    
    return mask


def batched_mask_center(x, mask_from, mask_to):
    
    assert mask_from.shape == mask_to.shape, "mask_from and mask_to should have the same shape"
    assert mask_from.ndim == 1, "mask_from and mask_to should be 1D"
    if not mask_from.shape[0] == 1:
        assert x.shape[0] == mask_from.shape[0] or x.shape[0] == mask_to.shape[0], "mask_from and mask_to should have the same batch size"
        
    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]
            
    return mask