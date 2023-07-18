from argparse import ArgumentParser

import pickle
import torch
import numpy as np
from torch.nn import functional as F
from pytorch_lightning import LightningModule

import sys
sys.path.append('../')
from models.unrolled import UnrolledNetwork
from models.datalayer import DataConsistency
from models.didn import DIDN
from models.conditional import Cond_DIDN

class UnrolledGradModule(LightningModule):
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
        super(UnrolledGradModule, self).__init__(**kwargs)
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
        self.mid_grad = []
        self.output_grad = []
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
    
    def training_step(self, batch, batch_idx):        
        # random lambda_ using a scheduler        
        if self.lambda_ is None:
            lambda_ = self.lambda_scheduler()
        else:
            lambda_ = torch.from_numpy(np.array(self.lambda_).astype(np.float32))
        
        lambda_ = lambda_.to(batch.image.device) 

        mid, output = self(batch.image, batch.kspace, batch.mask, lambda_)
        
        mid.register_hook(lambda grad: self.mid_grad.append(grad))
        output.register_hook(lambda grad: self.output_grad.append(grad))
        
        loss = F.l1_loss(output, batch.target) 
        
        return loss        
    def on_train_epoch_end(self):
        for i in range(len(self.mid_grad)):
            self.mid_grad[i] = self.mid_grad[i].cpu().detach().numpy()
            self.output_grad[i] = self.output_grad[i].cpu().detach().numpy()
        
        pickle.dump(self.mid_grad, open('mid_grad.pkl', 'wb'))
        pickle.dump(self.output_grad, open('output_grad.pkl', 'wb'))
        
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