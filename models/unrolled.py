import numpy as np
import torch
import torch.nn as nn

class UnrolledNetwork(nn.Module):
    def __init__(
        self,
        num_iter: int,
        model: nn.Module,
        model_config: dict,
        datalayer: nn.Module,
        datalayer_config: dict,
        shared_params: bool =True,
        save_space: bool =True,
        reset_cache: bool =False,
    ):
        """_summary_

        Args:
            num_iter (int): _description_
            model (nn.Module): _description_
            model_config (dict): _description_
            datalayer (nn.Module): _description_
            datalayer_config (dict): _description_
            shared_params (bool, optional): _description_. Defaults to True.
            save_space (bool, optional): _description_. Defaults to False.
            reset_cache (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        
        self.shared_params = shared_params
        if self.shared_params:
            self.num_iter = 1
        else:
            self.num_iter = num_iter
            
        self.num_iter_total = num_iter
        self.is_trainable = [True] * num_iter
        
        self.gradR = torch.nn.ModuleList(
            [model(**model_config) for _ in range(self.num_iter)]
        )
        
        self.gradD = torch.nn.ModuleList(
            [datalayer(**datalayer_config) for _ in range(self.num_iter)]
        )
        
        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space

        self.reset_cache = reset_cache

    def forward(self, x, y, mask, lambda_):
        x_all = [x]
        x_half_all = []
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1]+1, self.num_iter)
            
        for i in range(num_iter):
            x = x.permute(0, 3, 1, 2)
            x_thalf = (x - self.gradR[i%self.num_iter](x, lambda_)).permute(0, 2, 3, 1)
            x = self.gradD[i%self.num_iter](x_thalf, y, mask, lambda_)
            x_all.append(x)
            x_half_all.append(x_thalf)
            
        return x_all[-1]
    
    def forward_save_space(self, x, y, mask, lambda_):
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1]+1, self.num_iter)
            
        for i in range(num_iter):
            x = x.permute(0, 3, 1, 2)
            x_thalf = (x - self.gradR[i%self.num_iter](x, lambda_)).permute(0, 2, 3, 1)
            x = self.gradD[i%self.num_iter](x_thalf, y, mask, lambda_)
            
            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()
            
        return x
    
    
    def freeze(self, i):
        """ freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = False
        self.is_trainable[i] = False

    def unfreeze(self, i):
        """ freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = True
        self.is_trainable[i] = True

    def freeze_all(self):
        """ freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.freeze(i)

    def unfreeze_all(self):
        """ freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.unfreeze(i)

    def copy_params(self, src_i, trg_j):
        """ copy i-th cascade net parameters to j-th cascade net parameters """
        src_params = self.gradR[src_i].parameters()
        trg_params = self.gradR[trg_j].parameters()

        for i, (trg_param, src_param) in enumerate(
                zip(trg_params, src_params)):
            trg_param.data.copy_(src_param.data)

    def stage_training_init(self):
        self.freeze_all()
        self.unfreeze(0)
        print(self.is_trainable)

    def stage_training_transition_i(self, copy=False):
        if not self.shared_params:
            # if all unlocked, don't do anything
            if not np.all(self.is_trainable):
                for i in range(self.num_iter):

                    # if last cascade is reached, unlock all
                    if i == self.num_iter - 1:
                        self.unfreeze_all()
                        break

                    # freeze current i, unlock next. copy parameter if specified
                    if self.is_trainable[i]:
                        self.freeze(i)
                        self.unfreeze(i+1)
                        if copy:
                            self.copy_params(i, i+1)
                        break