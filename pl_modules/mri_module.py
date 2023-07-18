from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics.metric import Metric

import sys
sys.path.append('../')
from utils import evaluate
from utils import io


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(LightningModule):
    
    def __init__(
        self,
        num_log_images:int=16
        ):
        """
        Number of images to log in tensorboard. Defaults to 16.

        Args:
            num_log_images (int, optional): Defaults to 16.
        """
        super().__init__()
        
        self.num_log_images = num_log_images
        self.val_log_indices = None
        self.test_log_indices = None
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
    
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, self.global_step)

        
    def on_validation_epoch_end(self):
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for val_log in self.validation_step_outputs:
            losses.append(val_log["val_loss"].view(-1))
            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"].keys():
                max_vals[k] = val_log["max_vals"][k]
            #
        assert mse_vals.keys() == target_norms.keys() == ssim_vals.keys() == max_vals.keys(), "Mismatched keys"

        
        # apply means across image volumes
        metrics = {"nmse": 0.0, "ssim": 0.0, "psnr": 0.0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            target_norm = torch.mean(torch.cat([v.view(-1) for _, v in target_norms[fname].items()]))
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))
            
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(torch.tensor(len(losses), dtype=torch.float))
        
        self.log("validation_loss", val_loss/tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"validation_metrics/{metric}", value/tot_examples)
            
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):

        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for test_log in self.test_step_outputs:
            # losses.append(test_log["val_loss"].view(-1))
            for k in test_log["mse_vals"].keys():
                mse_vals[k].update(test_log["mse_vals"][k])
            for k in test_log["target_norms"].keys():
                target_norms[k].update(test_log["target_norms"][k])
            for k in test_log["ssim_vals"].keys():
                ssim_vals[k].update(test_log["ssim_vals"][k])
            for k in test_log["max_vals"].keys():
                max_vals[k] = test_log["max_vals"][k]
            #
        assert mse_vals.keys() == target_norms.keys() == ssim_vals.keys() == max_vals.keys(), "Mismatched keys"

        # apply means across image volumes
        metrics = {"nmse": 0.0, "ssim": 0.0, "psnr": 0.0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            target_norm = torch.mean(torch.cat([v.view(-1) for _, v in target_norms[fname].items()]))
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(
                torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))

        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        # val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        # tot_slice_examples = self.TotSliceExamples(torch.tensor(len(losses), dtype=torch.float))
        #
        # self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples)

        self.test_step_outputs.clear()
    # def on_test_epoch_end(self):
    #
    #     outputs = defaultdict(dict)
    #
    #     for log in self.test_logs:
    #         for i, (fname, slice_num) in enumerate(zip(log['fname'], log['slice'])):
    #             outputs[fname][slice_num] = log['output'][i]
    #
    #     # stack all the slices for each file
    #     for fname in outputs:
    #         outputs[fname] = np.stack([out for _, out in sorted(outputs[fname].items())])
    #
    #     #use the default_root_dir if we have a trainer, otherwise save to cwd
    #     if hasattr(self, "trainer"):
    #         save_dir = Path(self.trainer.default_root_dir)/"reconstructions"
    #     else:
    #         save_dir = Path.cwd()/f"reconstructions"
    #
    #     self.print(f"Savings reconstructions to {save_dir}")
    #
    #     io.save_reconstructions(outputs, save_dir)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_log_images", type=int, default=16, help="Number of images to log")
        
        return parser