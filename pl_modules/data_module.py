from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import csv
import os
import random
import sys
sys.path.append('..')
from data import SliceDataset, OasisSliceDataset

from typing import Callable, Optional

class BalancedGenderAgeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, gender_key, age_key):
        self.dataset = dataset
        self.gender_key = gender_key
        self.age_key = age_key
        self.indices = self._get_indices()

    def __iter__(self):
        # random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def _get_indices(self):
        indices_by_group = {}
        for index, sample in enumerate(self.dataset):
            age = sample.metadata[self.age_key]
            gender = sample.metadata[self.gender_key]

            # Create a tuple of gender and age group
            gender_age_tuple = (gender, age)

            if gender_age_tuple not in indices_by_group:
                indices_by_group[gender_age_tuple] = []
            indices_by_group[gender_age_tuple].append(index)

        indices_by_group = {key: random.sample(value, len(value)) for key, value in indices_by_group.items()}
        indices = [item for sublist in zip(*(indices for indices in indices_by_group.values())) for item in sublist]


        return indices
class OasisDataModule(pl.LightningDataModule):

    def __init__(
            self,
            list_path: Path,
            data_path: Path,
            train_transform: Callable,
            val_transform: Callable,
            test_transform: Callable,
            test_path: Optional[Path] = None,
            sample_rate: Optional[float] = None,
            val_sample_rate: Optional[float] = None,
            test_sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            val_volume_sample_rate: Optional[float] = None,
            test_volume_sample_rate: Optional[float] = None,
            train_filter: Optional[Callable] = None,
            val_filter: Optional[Callable] = None,
            test_filter: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            batch_size: int = 1,
            num_workers: int = 4,
    ):
        """_summary_

        Args:

            data_path (Path): Path to root data directory. For example, if `knee/path`
                is the root directory with subdirectories `singlecoil_train` and
                `singlecoil_val`, you would input `knee/path` for data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            train_transform (Callable): A transform object for the training dataset.
            val_transform (Callable): A transform object for the validation dataset.
            test_transform (Callable): A transform object for the test dataset.
            test_split (str, optional): Name of test split from ("test", "challenge").
                                        Defaults to "test".
            test_path (Optional[Path], optional):  An optional test path. Passing this overwrites
                                        data_path and test_split. Defaults to None.
            sample_rate (Optional[float], optional): Fraction of slices of the training data split to use.
                                        Can be set to less than 1.0 for rapid prototyping. If not set,
                                        it defaults to 1.0. To subsample the dataset either set
                                        sample_rate (sample by slice) or volume_sample_rate (sample by
                                        volume), but not both. Defaults to None.
            val_sample_rate (Optional[float], optional): Same as sample_rate, but for val split. Defaults to None.
            test_sample_rate (Optional[float], optional): Same as sample_rate, but for test split. Defaults to None.
            volume_sample_rate (Optional[float], optional): Same as sample rate but in volume. Defaults to None.
            val_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for val split. Defaults to None.
            test_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for test split. Defaults to None.
            train_filter (Optional[Callable], optional):  A callable which takes as input a training example
                                        metadata, and returns whether it should be part of the training
                                        dataset. Defaults to None.
            val_filter (Optional[Callable], optional): Same as train_filter but for val split. Defaults to None.
            test_filter (Optional[Callable], optional): Same as train_filter but for test split. Defaults to None.
            use_dataset_cache (bool, optional):  Whether to cache dataset metadata. This is
                                        very useful for large datasets like the brain data. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_workers (int, optional): Number of workers for PyTorch dataloader. Defaults to 4.
        """
        super().__init__()

        # assert self._check_both_not_none(sample_rate, volume_sample_rate), "sample_rate and volume_sample_rate cannot both be set"
        # assert self._check_both_not_none(val_sample_rate, val_volume_sample_rate), "val_sample_rate and val_volume_sample_rate cannot both be set"
        # assert self._check_both_not_none(test_sample_rate, test_volume_sample_rate), "test_sample_rate and test_volume_sample_rate cannot both be set"
        self.list_path = list_path
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache = use_dataset_cache
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(self,
                            data_transform: Callable,
                            data_partition: str,
                            sample_rate: Optional[float] = None) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            list_path = str(self.list_path) + "_train.csv"
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
        else:
            is_train = False
            if data_partition == "val":
                list_path = str(self.list_path) + "_val.csv"
                sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
            elif data_partition == "test":
                list_path = str(self.list_path) + "_test.csv"
                sample_rate = self.test_sample_rate if sample_rate is None else sample_rate


        dataset = OasisSliceDataset(
            root=list_path,
            data_root=self.data_path,
            transform=data_transform,
            sample_rate=sample_rate
        )
        gender_key = 'Gender'
        age_key = 'Age'

        # sampler = BalancedGenderAgeSampler(dataset, gender_key, age_key)
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )
        # for batch in dataloader:
        #     print(batch)

        return dataloader
    # revise
    def prepare_data(self):

            list_paths = [
                str(self.list_path) + "_train.csv",
                str(self.list_path) + "_val.csv",
                str(self.list_path) + "_test.csv"
            ]

            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform
            ]

            for i, (list_path, data_transform) in enumerate(zip(list_paths, data_transforms)):
                sample_rate = self.sample_rate
                _ = OasisSliceDataset(
                    root=list_path,
                    data_root=self.data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition="test")

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_path", type=Path, default=None, help="Path to the root data directory")
        parser.add_argument("--test_path", type=Path, default=None,
                            help="Path to the test data directory, overwrites data-path and test_split.")
        parser.add_argument("--challenge", type=str, choices=("singlecoil", "multicoil"), default="singlecoil",
                            help="Which challenge")
        parser.add_argument("--test_split", type=str, choices=("test", "challenge"), default="test",
                            help="Which data partition to use as test split")
        parser.add_argument("--sample_rate", type=float, default=None, help="Fraction of slices to use for training")
        parser.add_argument("--val_sample_rate", type=float, default=None,
                            help="Fraction of slices to use for validation")
        parser.add_argument("--test_sample_rate", type=float, default=None,
                            help="Fraction of slices to use for testing")
        parser.add_argument("--volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for training")
        parser.add_argument("--val_volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for validation")
        parser.add_argument("--test_volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for testing")
        parser.add_argument("--use_dataset_cache", type=bool, default=True,
                            help="Whether to cache dataset metadata in memory")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

        return parser

    @staticmethod
    def _check_both_not_none(v1, v2):
        return True if (v1 is not None) and (v2 is not None) else False
class FastMriDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        ):
        """_summary_

        Args:
            data_path (Path): Path to root data directory. For example, if `knee/path`
                is the root directory with subdirectories `singlecoil_train` and
                `singlecoil_val`, you would input `knee/path` for data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            train_transform (Callable): A transform object for the training dataset.
            val_transform (Callable): A transform object for the validation dataset.
            test_transform (Callable): A transform object for the test dataset.
            test_split (str, optional): Name of test split from ("test", "challenge"). 
                                        Defaults to "test".
            test_path (Optional[Path], optional):  An optional test path. Passing this overwrites 
                                        data_path and test_split. Defaults to None.
            sample_rate (Optional[float], optional): Fraction of slices of the training data split to use.
                                        Can be set to less than 1.0 for rapid prototyping. If not set,
                                        it defaults to 1.0. To subsample the dataset either set
                                        sample_rate (sample by slice) or volume_sample_rate (sample by
                                        volume), but not both. Defaults to None.
            val_sample_rate (Optional[float], optional): Same as sample_rate, but for val split. Defaults to None.
            test_sample_rate (Optional[float], optional): Same as sample_rate, but for test split. Defaults to None.
            volume_sample_rate (Optional[float], optional): Same as sample rate but in volume. Defaults to None.
            val_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for val split. Defaults to None.
            test_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for test split. Defaults to None.
            train_filter (Optional[Callable], optional):  A callable which takes as input a training example
                                        metadata, and returns whether it should be part of the training
                                        dataset. Defaults to None.
            val_filter (Optional[Callable], optional): Same as train_filter but for val split. Defaults to None.
            test_filter (Optional[Callable], optional): Same as train_filter but for test split. Defaults to None.
            use_dataset_cache (bool, optional):  Whether to cache dataset metadata. This is
                                        very useful for large datasets like the brain data. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_workers (int, optional): Number of workers for PyTorch dataloader. Defaults to 4.
        """
        super().__init__()
        
        # assert self._check_both_not_none(sample_rate, volume_sample_rate), "sample_rate and volume_sample_rate cannot both be set"
        # assert self._check_both_not_none(val_sample_rate, val_volume_sample_rate), "val_sample_rate and val_volume_sample_rate cannot both be set"
        # assert self._check_both_not_none(test_sample_rate, test_volume_sample_rate), "test_sample_rate and test_volume_sample_rate cannot both be set"  
        
        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache = use_dataset_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _create_data_loader(self, 
                            data_transform: Callable, 
                            data_partition: str, 
                            sample_rate: Optional[float]=None, 
                            volume_sample_rate: Optional[float]=None) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = self.val_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = self.test_sample_rate if sample_rate is None else sample_rate
                volume_sample_rate = self.test_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
                raw_sample_filter = self.test_filter
                
        if data_partition in ("test", "challenge") and self.test_path is not None:
            data_path = self.test_path
        else:
            data_path = self.data_path / f"{self.challenge}_{data_partition}"
            
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            challenge=self.challenge,
            use_dataset_cache=self.use_dataset_cache,
            raw_sample_filter=raw_sample_filter
        )
        
        sampler = None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )
        
        return dataloader

    def prepare_data(self):
        
        if self.use_dataset_cache:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
        
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                self.test_path
            ]
            
            data_transforms = [ 
                self.train_transform,
                self.val_transform,
                self.test_transform
            ]
            
            for i, (data_path, data_transform) in enumerate(zip(data_paths, data_transforms)):
                sample_rate = self.sample_rate
                volume_sample_rate = self.volume_sample_rate
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")
    
    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")
    
    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition=self.test_split)
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument("--data_path", type=Path, default=None, help="Path to the root data directory")
        parser.add_argument("--test_path", type=Path, default=None, help="Path to the test data directory, overwrites data-path and test_split.")
        parser.add_argument("--challenge", type=str, choices=("singlecoil", "multicoil"), default="singlecoil", help="Which challenge")
        parser.add_argument("--test_split", type=str, choices=("test", "challenge"), default="test", help="Which data partition to use as test split")
        parser.add_argument("--sample_rate", type=float, default=None, help="Fraction of slices to use for training")
        parser.add_argument("--val_sample_rate", type=float, default=None, help="Fraction of slices to use for validation")
        parser.add_argument("--test_sample_rate", type=float, default=None, help="Fraction of slices to use for testing")
        parser.add_argument("--volume_sample_rate", type=float, default=None, help="Fraction of volumes to use for training")
        parser.add_argument("--val_volume_sample_rate", type=float, default=None, help="Fraction of volumes to use for validation")
        parser.add_argument("--test_volume_sample_rate", type=float, default=None, help="Fraction of volumes to use for testing")
        parser.add_argument("--use_dataset_cache", type=bool, default=True, help="Whether to cache dataset metadata in memory")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
        
        return parser       
        
    
    @staticmethod
    def _check_both_not_none(v1, v2):
        return True if (v1 is not None) and (v2 is not None) else False