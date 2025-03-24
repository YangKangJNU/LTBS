
import argparse
import json
import os

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HYDRA_FULL_ERROR'] = '1'
from src.models.model import LTBS as LTBS
from src.dataset import VADataset
from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from numpy import seterr
seterr(all='raise')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--log_step', default=100)
    parser.add_argument('--val_step', default=2000)
    parser.add_argument('--ckpt', default='')
    args = parser.parse_args()
    return args

@hydra.main(version_base=None, config_path="configs/v1", config_name="default.yaml")
def main(cfg: DictConfig):
    args = parse_args()

    train_dataset = VADataset(args.root, 'train')
    val_dataset = VADataset(args.root, 'val')
    model = LTBS(cfg.model)
    trainer = Trainer(
        model,
        num_warmup_steps=100,
        learning_rate=args.lr,
        grad_accumulation_steps = 1,
        tensorboard_log_dir=args.output_dir,
        checkpoint_path = os.path.join(args.output_dir, 'l2s.pt'),
        log_file = os.path.join(args.output_dir, 'l2s.txt')
    )


    trainer.train(train_dataset,
                   args.epochs, 
                   args.batch_size, 
                   num_workers=args.num_workers, 
                   val_dataset=val_dataset,
                   log_step=args.log_step, 
                   val_step=args.val_step,
                   ckpt_path=args.ckpt)

if __name__ == '__main__':
    main()
