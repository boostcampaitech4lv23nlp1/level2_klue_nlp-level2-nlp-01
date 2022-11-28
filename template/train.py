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
    # cuda debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    
    parser.add_argument('--masking', default=True, type=bool)
    parser.add_argument('--pooling', default=False, type=bool)
    parser.add_argument('--criterion', default='cross_entropy', type=str)  # cross_entropy, focal_loss

    parser.add_argument('--train_path', default='../dataset/train/removed_paren_train_split.csv')
    parser.add_argument('--dev_path', default='../dataset/train/removed_paren_val_split.csv')
    parser.add_argument('--test_path', default='../dataset/train/removed_paren_val_split.csv')
    parser.add_argument('--predict_path', default='../dataset/test/test_data.csv')
    args = parser.parse_args(args=[])
    
    # try:
    #     wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    # except:
    #     anony = "must"
    #     print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    # wandb.init(project="level2", name= "4-2. baseline + entity_marker(query) + query augmentation")
    # wandb_logger = WandbLogger('level2')

    dataloader = Dataloader(
        args.tokenizer_name,
        args.batch_size,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        args.masking,
        shuffle=True
    )

    model = Model(
        args.model_name, 
        args.learning_rate,
        args.pooling,
        args.criterion
    )

    # tracking special tokens(entity marker 쓸 때만 사용)
    # if dataloader.added_token_num > 0:
    #     model.model.resize_token_embeddings(
    #         dataloader.tokenizer.vocab_size + dataloader.added_token_num
    #     )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="/opt/models/",
        monitor="val_micro_f1",
        save_top_k=-1,
        filename="roberta-large+{epoch}+{val_micro_f1:.3f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch,  
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        # logger=wandb_logger
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    
    # model_name = re.sub(r'[/]', '-', args.model_name)
    # torch.save(model, f'/opt/models/{model_name}.pt')