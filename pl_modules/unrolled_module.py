from argparse import ArgumentParser
from typing import Any, Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

import piqa
import torch
import numpy as np
from torch.nn import functional as F
from collections import defaultdict

from .mri_module import MriModule

import sys
sys.path.append('../')
from models.unrolled import UnrolledNetwork
from models.datalayer import DataConsistency
from models.didn import DIDN
from models.conditional import Cond_DIDN
from utils import evaluate
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c


class UnrolledModule(MriModule):
    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        pad_data: bool = True,
        reg_model: str = 'DIDN',
        data_term: str = 'DataConsistency',
        num_iter: int = 10,
        num_chans: int = 32,
        n_res_blocks: int = 10,
        global_residual: bool = True,
        shared_params: bool = True,
        save_space: bool =False,
        reset_cache: bool =False,
        lambda_=None,
        lr: float = 1e-3,
        lr_step_size: int = 15,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        **kwargs
    ):
        super(UnrolledModule, self).__init__(**kwargs)
        self.save_hyperparameters()
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.num_iter = num_iter
        
        self.model_config = {
            'in_chans': in_chans,
            'out_chans': out_chans,
            'pad_data': pad_data,
        }
                
        if reg_model == 'DIDN':
            self.model = DIDN
            self.model_config.update({
                'num_chans': num_chans,
                'n_res_blocks': n_res_blocks,
                'global_residual': global_residual,
            })
        elif reg_model == 'Cond_DIDN':
            self.model = Cond_DIDN
            self.model_config.update({
                'num_chans': num_chans,
                'n_res_blocks': n_res_blocks,
                'global_residual': global_residual,
            })
        else:
            raise NotImplemented(f'Regularization model {reg_model} not implemented.')
        
        self.datalayer_config = {}

        if data_term == 'DataConsistency':
            self.datalayer = DataConsistency
        else:
            raise NotImplemented(f'Data term {data_term} not implemented.')
        
        self.shared_params = shared_params
        self.save_space = save_space
        self.reset_cache = reset_cache
        self.lambda_ = lambda_
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.unrolled = UnrolledNetwork(
            num_iter=self.num_iter,
            model=self.model,
            model_config=self.model_config,
            datalayer=self.datalayer,
            datalayer_config=self.datalayer_config,
            shared_params=self.shared_params,
            save_space=self.save_space,
            reset_cache=self.reset_cache,
        )
    def lambda_scheduler(self):
        lambda_ = 1 - torch.sin(torch.tensor((torch.pi / 2) * (self.current_epoch // 10) * 0.1)) + torch.randn(1) * 0.1
        return torch.tensor([0.0]) if lambda_ < 0 else torch.tensor([1.0]) if lambda_ > 1 else lambda_
    
    def forward(self, image, k, mask, lambda_):
        return self.unrolled(image, k, mask, lambda_)
    
    def _post_process(self, output, mean, std):
        mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        return complex_abs(output * std + mean)
    
    def training_step(self, batch, batch_idx):        
        # random lambda_ using a scheduler        
        if self.lambda_ is None:
            lambda_ = self.lambda_scheduler()
        else:
            lambda_ = torch.from_numpy(np.array(self.lambda_).astype(np.float32))
        
        lambda_ = lambda_.to(batch.image.device) 

        output = self(batch.image, batch.kspace, batch.mask, lambda_)
        post_output = self._post_process(output, batch.mean, batch.std)
        post_target = self._post_process(batch.target, batch.mean, batch.std)
        ssim = piqa.SSIM(n_channels=1).cuda()
        loss = F.l1_loss(output, batch.target) + 0.1 * (
            1 - ssim((post_output/post_output.max()).unsqueeze(1), (post_target/post_target.max()).unsqueeze(1))
        )
        
        self.log("loss", loss.detach())
        self.log("lambda", lambda_.detach())
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        if self.lambda_ is None:
            lambda_ = np.random.uniform(0, 1)
        else:
            lambda_ = self.lambda_

        lambda_ = torch.from_numpy(np.array(lambda_).astype(np.float32)).to(batch.image.device).unsqueeze(0)
        # we set the lambda manually
        output = self(batch.image, batch.kspace, batch.mask, lambda_)
        post_image = self._post_process(batch.image, batch.mean, batch.std)
        post_output = self._post_process(output, batch.mean, batch.std)
        post_target = self._post_process(batch.target, batch.mean, batch.std)
        ssim = piqa.SSIM(n_channels=1).cuda()

        val_logs = {
            "batch_idx": batch_idx,
            "image": post_image,
            "kspace": batch.kspace,
            "mask": batch.mask,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": post_output,
            "target": post_target,
            "val_loss": F.l1_loss(output, batch.target) + 0.1 * (
                1 - ssim((post_output/post_output.max()).unsqueeze(1), (post_target/post_target.max()).unsqueeze(1))
                )
        }
        
        for k in ("batch_idx", "image", "kspace", "mask", "fname",
                    "slice_num", "max_value", "output", "target", "val_loss"):
            if k not in val_logs.keys():
                raise ValueError(f"Missing key {k} in val_logs")
        
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise ValueError(f"Output has wrong shape {val_logs['output'].shape}")
        
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise ValueError(f"Target has wrong shape {val_logs['target'].shape}")
        
        # * pick an image to log
        if self.val_log_indices is None:
            self.val_log_indices = list(np.random.permutation(len(self.trainer.val_dataloaders))[: self.num_log_images])            
        
        # * log the image to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
            
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_image_idx_{batch_idx}"
                
                image = val_logs["image"][i].unsqueeze(0)
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                k0 = val_logs["kspace"][i].unsqueeze(0)
                error = torch.abs(target - output)
                image = image / image.max()
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                k0 = complex_abs(ifft2c(k0)) / complex_abs(ifft2c(k0)).max()
                mask = torch.tile(val_logs["mask"][i], [320, 1, 1]).permute(2, 0, 1)

                
                self.log_image(f"{key}/image", image)
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/recon", output)
                self.log_image(f"{key}/error", error)
                self.log_image(f"{key}/k0", k0)
                self.log_image(f"{key}/mask", mask)
        
        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            
            mse_vals[fname][slice_num] = torch.tensor(evaluate.mse(target, output)).view(1)
            target_norms[fname][slice_num] = torch.tensor(evaluate.mse(target, np.zeros_like(target))).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)).view(1)
            max_vals[fname] = maxval
        
        pred = {
            "val_loss": val_logs["val_loss"],
            "mse_vals": mse_vals,
            "target_norms": target_norms,
            "ssim_vals": ssim_vals,
            "max_vals": max_vals
        }        
        self.validation_step_outputs.append(pred)
        return pred
            
    def test_step(self, batch, batch_idx):
        
        if self.lambda_ is None:
            lambda_ = np.random.uniform(0, 1)
        else:
            lambda_ = self.lambda_
        lambda_ = torch.from_numpy(np.array(lambda_).astype(np.float32)).to(batch.image.device).unsqueeze(0)
        # we set the lambda manually
        output = self(batch.image, batch.kspace, batch.mask, lambda_)
        post_image = self._post_process(batch.image, batch.mean, batch.std)
        post_output = self._post_process(output, batch.mean, batch.std)
        post_target = self._post_process(batch.target, batch.mean, batch.std)

        return {
            "batch_idx": batch_idx,
            "image": post_image,
            "fname": batch.fname,
            "slice": batch.slice_num,
            "max_value": batch.max_value,
            "output": post_output,
        }
            
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, 
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
        )
    
        return [optim], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)
        
        parser.add_argument("--in_chans", type=int, default=2)
        parser.add_argument("--out_chans", type=int, default=2)
        parser.add_argument("--pad_data", type=bool, default=True)
        parser.add_argument("--reg_model", type=str, default="DIDN")
        parser.add_argument("--data_term", type=str, default="DataConsistency")
        parser.add_argument("--num_iter", type=int, default=1)
        parser.add_argument("--num_chans", type=int, default=64)
        parser.add_argument("--n_res_blocks", type=int, default=5)
        parser.add_argument("--global_residual", type=bool, default=False)
        parser.add_argument("--shared_params", type=bool, default=True)
        parser.add_argument("--save_space", type=bool, default=False)
        parser.add_argument("--reset_cache", type=bool, default=False)
        parser.add_argument("--lambda_", type=float, default=None)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr_step_size", type=int, default=15)
        parser.add_argument("--lr_gamma", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        
        return parser