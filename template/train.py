import re
import os
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    
    parser.add_argument('--masking', default=True, type=bool)
    parser.add_argument('--pooling', default=True, type=bool)
    parser.add_argument('--criterion', default='cross_entropy', type=str)  # cross_entropy, focal_loss
    parser.add_argument('--shuffle', default=True, type=bool)

    parser.add_argument('--n_folds', default=5, type=bool)
    parser.add_argument('--split_seed', default=42, type=bool)

    parser.add_argument('--train_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/removed_paren_train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/removed_paren_val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/level2_klue_nlp-level2-nlp-01/dataset/train/removed_paren_val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])
    
    try:
        wandb.login()
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    wandb.init(project="kfolds", name= f"{args.model_name}")
    wandb_logger = WandbLogger('kfolds')


    n_iter = 0
    results = []
    for k in range(args.n_folds):
        print(f'=======================fold {k}==========================')
        dataloader = Dataloader(args.tokenizer_name, args.batch_size,
                                args.train_path, args.dev_path, args.test_path, args.predict_path, args.masking, args.shuffle,
                                k, args.n_folds, args.split_seed)

        model = Model(args.model_name, args.learning_rate, args.pooling, args.criterion)
        model_name = re.sub(r'[/]', '-', args.model_name)

        checkpoint_callback = ModelCheckpoint(
            dirpath="/opt/ml/template/kfolds/",
            save_top_k=1,
            mode="max",
            monitor="val_micro_f1",
            filename="{model_name}+kfold+{k}-{epoch}+{val_micro_f1:.3f}"
        )
            
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=args.max_epoch, 
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
            precision=16,
            logger=wandb_logger
            )

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        score = trainer.test(model=model, datamodule=dataloader)

        results.extend(score)

        print('best model score: ', checkpoint_callback.best_model_score)
        print('best model path: ', checkpoint_callback.best_model_path)
        
        