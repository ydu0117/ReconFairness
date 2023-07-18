import os
import logging
import random
import torch
import pickle
import h5py
import yaml
import numpy as np
import xml.etree.ElementTree as etree
from pathlib import Path
from warnings import warn
from typing import NamedTuple, Dict, Any, Union, Optional, Callable, Tuple, Sequence
import csv
from glob import glob
import nibabel as nib


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(key: str, data_config_file="fastmri_dirs.yaml") -> Path:
    """_summary_

    Args:
        key (_type_): key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file (str, optional): Default path config file to fetch path
            from. Defaults to "fastmri_dirs.yaml".

    Returns:
        Path: _description_
    """
    
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "Dataset/fastMRI/",
            "brain_path": "Dataset/FastMRI/",
            "log_path": "logs/",
        }
        
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)
        
        data_dir = default_config[key]
        
        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]
            
    return Path(data_dir)


class OasisMRIRawDataSample(NamedTuple):
    """Basic data type for fastMRI raw data.

    Elements:
        fname: id for each patient, Path
        slice_ind: slice index, int
        metadata: metadata for each volume, Dict
    """
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class OasisSliceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: list,
            data_root: list,
            transform: Optional[Callable] = None,
            sample_rate: Optional[float] = None
    ):
        self.root = root
        self.transform = transform
        self.sample_rate = sample_rate
        self.data_root = data_root
        self.raw_samples = self.create_sample_list(self.root)

    def __getitem__(self, index):
        # * get data from raw_samples and feed into transform
        fname, dataslice, metadata = self.raw_samples[index]
        raw_target = self.read_raw_data(self.data_root, fname, dataslice)
        _, raw_kspace = self.im2kp(torch.tensor(raw_target))
        if self.transform is None:
            sample = (raw_kspace, torch.tensor(raw_target), fname, dataslice, metadata)
        else:
            sample = self.transform(raw_kspace, torch.tensor(raw_target), fname, dataslice, metadata, mask=None)

        return sample

    def __len__(self):
        return len(self.raw_samples)

    def create_sample_list(self, list_path):
        raw_samples = []
        with open(list_path) as metalist:
            datalist = csv.reader(metalist, delimiter=',')
            for row in datalist:
                # if datalist.line_num == 1:
                #     continue
                fname = row[0]
                metadata = {'Gender': row[1],
                            'Age': self.age_group(row[2]),
                            }
                slices = int(row[-1])
                mid = round(slices/2)
                half_sli = round(slices * self.sample_rate / 2)
                for num in range((mid - half_sli),(mid + half_sli)):
                    new_raw_sample = OasisMRIRawDataSample(fname, num, metadata)
                    raw_samples.append(new_raw_sample)
            metalist.close()
        return raw_samples

    def age_group(self, age):
        # Group age data into three categories
        young_age = 40
        old_age = 65
        if int(age) <= young_age:
            age_group = 'Y'
        elif int(age) > old_age:
            age_group = 'O'
        else:
            age_group = 'M'
        return age_group

    def read_raw_data(self, data_path, patient_id, dataslice):
        # read data and select slices based on patient id from data_list

        image_path = glob(os.path.join(data_path, patient_id, 'PROCESSED', 'MPRAGE', 'T88_111', '*t88_gfc.img'))
        image_data = nib.load(image_path[0]).get_fdata()
        raw_image = np.transpose(image_data.squeeze(-1), (1, 0, 2))
        raw_slices = raw_image[dataslice]

        return raw_slices

    def im2kp(self,image_data, norm_type="ortho"):
        # perform fft in last two dimensions of input data
        kspace_complex_data = torch.fft.fft2(image_data, dim=(-2, -1), norm=norm_type)
        kspace_split_data = torch.view_as_real(kspace_complex_data)


        return kspace_complex_data, kspace_split_data

class FastMRIRawDataSample(NamedTuple):
    """Basic data type for fastMRI raw data.

    Elements:
        fname: Path for each h5 file, Path
        slice_ind: slice index, int
        metadata: metadata for each volume, Dict
    """
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    

class SliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: list[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None
    ):
        """A PyTorch Dataset for loading fastMRI data.

        Args:
            root (Union[str, Path, os.PathLike]): Path to the dataset directory (e.g. knee_path/train, brain_path/test, etc.)
            challenge (str): "singlecoil" or "multicoil".
            transform (Optional[Callable], optional): A callable object that takes a raw data sample as input and returns a transformed version. Defaults to None.
                The transform function should take ['kspace', 'target', 'attributes', 'filename', and 'slice'] as inputs.
                'target' maybe None if the dataset is a test dataset.
            use_dataset_cache (bool, optional): Whether to cache dataset metadata. Useful for large dataset. Defaults to False.
            sample_rate (Optional[float], optional): A float between 0 and 1. Defaults to 1 if None is given. Either sample_rate or volume_sample_rate should be set.
            volume_sample_rate (Optional[float], optional): _description_. The same as sample_rate. Defaults to 1 if None is given.
            dataset_cache_file (Union[str, Path, os.PathLike], optional): A file in which to cache dataset information for faster load times. Defaults to "Dataset/fastMRI/dataset_cache.pkl".
            num_cols (Optional[Tuple[int]], optional): If provided, only slices with the desired number of columns will be considered. Defaults to None.
            raw_sample_filter (Optional[Callable], optional): A callable object that takes an raw_sample metadata as input and return a boolean indicating whether the raw_sample should be included in the dataset. Defaults to None.

        """
        # * Choose between singlecoil and multicoil
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f"Challenge should be singlecoil or multicoil, got {challenge}.")
        # * Check if sample_rate and volume_sample_rate are both set
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("Only one of sample_rate and volume_sample_rate should be set.")
        
        
        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        # * set different target for singlecoil or multicoil
        self.recons_key = ("reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss")
        
        # * The list of files
        self.raw_samples = []
        # * if there is a filter
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter
            
        # * set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0
            
        # load dataset cache
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}
            
        # check cache
        # if yes, use the metadata from cache
        # if not, regenerate the metadata
        
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]
                    
            
        # * slice level subsampling
        if sample_rate < 1.0:
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        # * volume level subsampling
        elif volume_sample_rate < 1.0:
            volume_names = sorted(list(set(f[0].stem for f in self.raw_samples)))
            random.shuffle(volume_names)
            num_volumes = round(len(volume_names) * volume_sample_rate)
            sampled_vols = volume_names[num_volumes]
            self.raw_samples = [
                raw_sample for raw_sample in self.raw_samples if raw_sample[0].stem in sampled_vols
            ]
            
        if num_cols:
            self.raw_samples = [
                ex for ex in self.raw_samples if ex[2]["encoding_size"][1] in num_cols
            ]
    
    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i):
        # * get data from raw_samples and feed into transform
        fname, dataslice, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            
        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)
            
        return sample
        
    def _retrieve_metadata(self, fname):
        """_summary_

        Args:
            fname (_type_): 

        Returns:
            _type_: _description_
        """
        with h5py.File(fname, 'r') as hf:
            et_root = etree.fromstring(hf['ismrmrd_header'][()])
            
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])), #640
                int(et_query(et_root, enc + ["y"])), # 372
                int(et_query(et_root, enc + ["z"])), # 1
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])), # 320
                int(et_query(et_root, rec + ["y"])), # 320
                int(et_query(et_root, rec + ["z"])), # 1
            )
            
            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"])) # 167
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1 # 334
            
            padding_left = enc_size[1] // 2 - enc_limits_center # 372 // 2 - 167 = 19
            padding_right = padding_left + enc_limits_max # 19 + 334 = 353
            
            num_slices = hf["kspace"].shape[0]
            
            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs
            }
            
        return metadata, num_slices

