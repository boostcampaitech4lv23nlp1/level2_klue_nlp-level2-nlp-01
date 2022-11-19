import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataloader import *
from models import *
import pickle as pkl
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/bert-base', type=str)
    parser.add_argument('--model_name', default='klue/bert-base', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/dataset/train/train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])
    
    try:
        wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    wandb.init(project="level2", name= f"{args.model_name}")
    wandb_logger = WandbLogger('level2')    

    dataloader = Dataloader(
        args.tokenizer_name,
        args.batch_size,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        shuffle=True
    )

    model = Model(args.model_name, args.learning_rate)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1,
        precision=16,
        logger=wandb_logger
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    
    torch.save(f'./{args.model_name}.pt')