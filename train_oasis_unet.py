import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pl_modules import OasisDataModule, UnetOasisModule
from data.mri_data import fetch_dir
from data.masking import create_mask_for_mask_type
from data.transforms import OasisDataTransform


torch.set_float32_matmul_precision('medium')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def build_args(k_num):
    parser = ArgumentParser()

    # basic args
    path_config = Path("Dataset/oasis_dirs.yaml")
    num_gpus = 3
    batch_size = 6
    data_path = fetch_dir("data_path", path_config)
    list_path = f'Dataset/oasis_unbalanced_{k_num}'
    print(list_path)
    default_root_dir = fetch_dir("log_path", path_config) / f"unet_unbalanced_{k_num}"
    print(default_root_dir)

    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--mask_type", default="random", type=str, choices=["random", "equispaced"])
    parser.add_argument("--center_fractions", default=[0.08], type=list)
    parser.add_argument("--accelerations", default=[4], type=list)
    parser.add_argument("--ckpt_path", default=None, type=str)

    
    parser = OasisDataModule.add_data_specific_args(parser)
    parser = UnetOasisModule.add_model_specific_args(parser)
    parser.set_defaults(
        list_path=list_path,
        data_path=data_path,
        gpus=num_gpus,
        sample_rate=0.6,
        seed=24,
        batch_size=batch_size,
        default_root_dir=default_root_dir,
        max_epochs=20,
        test_path=None
    )

    args = parser.parse_args()

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=5,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = str(ckpt_list[-1])

    print(args.ckpt_path)

    return args


def main(k_num):
    args = build_args(k_num)
    pl.seed_everything(args.seed)

    # * data
    # masking
    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    # data transform
    train_transform = OasisDataTransform(mask_func=mask, use_seed=False)
    val_transform = OasisDataTransform(mask_func=mask, use_seed=True)
    test_transform = OasisDataTransform(mask_func=mask, use_seed=True)

    # # pl data module
    data_module = OasisDataModule(
        list_path=args.list_path,
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        val_sample_rate=args.sample_rate,
        test_sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # * model
    model = UnetOasisModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        k_num=k_num
    )

    # * trainer
    trainer = pl.Trainer(
        logger=True,
        callbacks=args.callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,

    )

    # * run
    if args.mode == 'train':
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')


if __name__ == '__main__':
    for k in range(5):
        print(f'Fold {k} starts training!!')
        main(k)
    print(f'Finish {k+1} folds training!!')