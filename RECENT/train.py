import re
import argparse

import torch
import wandb
import transformers
import torchmetrics
import pandas as pd
import pickle as pkl
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from dataloader import *
from models import *
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/multiple_train.csv')
    parser.add_argument('--dev_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/multiple_val.csv')
    parser.add_argument('--test_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/multiple_val.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])
    
    try:
        wandb.login(key='YOUR WANDB KEY')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    wandb.init(project="level2", name= f"{args.model_name}-recent-multiple")
    wandb_logger = WandbLogger('level2')

    dataloader = Dataloader(args.tokenizer_name,
                            args.batch_size,
                            args.train_path,
                            args.dev_path,
                            args.test_path,
                            args.predict_path,
                            shuffle=True)
    
    model = Model(args.model_name, args.learning_rate)

    checkpoint_callback = ModelCheckpoint(dirpath="/opt/ml/models/recent/multiple", save_top_k=2, monitor="val_micro_f1", mode="max", filename="recent+{epoch}+{val_micro_f1:.3f}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator = 'gpu',
                        devices = 1,
                        max_epochs=args.max_epoch, 
                        log_every_n_steps=1,
                        precision=16,
                        logger = wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor]
                        )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
