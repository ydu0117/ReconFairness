from argparse import ArgumentParser
from typing import Any
import torch
import numpy as np
from torch.nn import functional as F
from collections import defaultdict
import pickle
from .mri_module import MriModule
import torch.nn as nn
import sys

sys.path.append('../')
from models.unet import Unet,ConvBlock
from utils import evaluate
from utils.complex import complex_abs


class UnetGradModule(MriModule):
    def __init__(
            self,
            in_chans: int = 1,
            out_chans: int = 1,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            lr: float = 1e-3,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            **kwargs
    ):
        """_summary_

        Args:
            in_chans (int, optional): _description_. Defaults to 1.
            out_chans (int, optional): _description_. Defaults to 1.
            chans (int, optional): _description_. Defaults to 32.
            num_pool_layers (int, optional): _description_. Defaults to 4.
            drop_prob (float, optional): _description_. Defaults to 0.0.
            lr (float, optional): _description_. Defaults to 1e-3.
            lr_step_size (int, optional): _description_. Defaults to 40.
            lr_gamma (float, optional): _description_. Defaults to 0.1.
            weight_decay (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.output_grad = []
        self.features = []
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )
        
        self.image_arr = []
        self.output_arr = []

    def forward(self, image):
        if image.ndim == 3:
            return self.unet(image.unsqueeze(1)).squeeze(1)
        else:
            return self.unet(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # def grad_cam(self, conv_features, grad):
    #
    #     pooled_gradients = torch.mean(grad, dim=(2, 3))
    #
    #     # Weight the last ConvBlock output
    #     weighted_output = conv_features * pooled_gradients
    #
    #     # Sum along the channel dimension
    #     grad_cam = torch.sum(weighted_output, dim=1).squeeze(0)
    #
    #         # ReLU and normalization
    #     grad_cam = F.relu(grad_cam)
    #     grad_cam /= torch.max(grad_cam)
    #     # Compute the Grad-CAM map
    #     grad_cam = F.relu(torch.sum(conv_features, dim=1))
    #     grad_cam = F.interpolate(grad_cam.unsqueeze(1), size=conv_features.shape[2:], mode='bilinear', align_corners=False)
    #     grad_cam = torch.clamp(grad_cam, min=0.0)
    #
    #     return grad_cam


    def training_step(self, batch, batch_idx):
        std = batch.std.unsqueeze(1).unsqueeze(2)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        # last_conv_block = None

        for module_name, module in self.unet.up_conv[-1].named_modules():
            if isinstance(module, nn.Sequential):
                for sub_module_name, sub_module in module.named_children():
                    # if isinstance(sub_module, nn.Conv2d):
                        # last_layer = sub_module
                        # print(last_layer)
                    if isinstance(sub_module, ConvBlock):
                        conv4_layer = sub_module.layers[4]
                        print(conv4_layer)

        output = self(batch.image)
        loss = F.l1_loss(output, batch.target)
        loss.backward(retain_graph=True)
        gradient = conv4_layer.weight.grad
        feature_map = conv4_layer.output
        print('feature_map:', feature_map)
        # print('gradient:', gradient)
        self.output_grad.append(gradient)
        image = batch.image * std + mean
        image = image / image.max()
        self.image_arr.append(image)
        output = output * std + mean
        output = output / output.max()
        self.output_arr.append(output)
        # gcam = self.grad_cam(output, self.output_grad)

        self.log("train_loss", loss.detach())

        return loss
    
    def on_train_epoch_end(self):
        for i in range(len(self.image_arr)):
            self.image_arr[i] = self.image_arr[i].cpu().detach().numpy()
            self.output_arr[i] = self.output_arr[i].cpu().detach().numpy()
            self.output_grad[i] = self.output_grad[i].cpu().detach().numpy()

        pickle.dump(self.image_arr, open('image.pkl', 'wb+'))
        pickle.dump(self.output_arr, open('output.pkl', 'wb+'))
        pickle.dump(self.output_grad, open('output_grad.pkl', 'wb+'))

    def configure_optimizers(self):

        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        parser.add_argument("--in_chans", type=int, default=1)
        parser.add_argument("--out_chans", type=int, default=1)
        parser.add_argument("--chans", type=int, default=32)
        parser.add_argument("--num_pool_layers", type=int, default=4)
        parser.add_argument("--drop_prob", type=float, default=0.0)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr_step_size", type=int, default=40)
        parser.add_argument("--lr_gamma", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)

        return parser
