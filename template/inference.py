import re
import os
import argparse
from models import Model
import torch
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torchmetrics
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from dataloader import *
import wandb
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/project2/level2_klue_nlp-level2-nlp-01/dataset/back_translation/cha_train.csv')
    parser.add_argument('--dev_path', default='/opt/ml/project2/level2_klue_nlp-level2-nlp-01/dataset/back_translation/cha_val.csv')
    parser.add_argument('--test_path', default='/opt/ml/project2/level2_klue_nlp-level2-nlp-01/dataset/back_translation/cha_val.csv')
    parser.add_argument('--predict_path', default='/opt/ml/project2/level2_klue_nlp-level2-nlp-01/dataset/test/removed_paren_test_data.csv')
    args = parser.parse_args(args=[])

    try:
        wandb.login(key='1f2c61c98244a7284f6726d8db5d111bca8111fa')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    wandb.init(project="inference", name= f"{args.model_name}")
    wandb_logger = WandbLogger('inference')

    dataloader = Dataloader(
        args.tokenizer_name,
        args.batch_size,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        shuffle=False
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1,
        logger = wandb_logger
    )

    model_name = re.sub(r'[/]', '-', args.model_name)
    model = Model.load_from_checkpoint(checkpoint_path='/opt/ml/template/models/roberta_large(cha_back_trans)+epoch=4+val_micro_f1=91.992.ckpt')
    
    # model = torch.load(f'{model_name}.pt')


    results = trainer.predict(model=model, datamodule=dataloader)

    preds_all, probs_all = [], []
    for preds, probs in results:
        preds_all.append(preds[0]); probs_all.append(str(list(probs)))

    preds_all = dataloader.num_to_label(preds_all) 

    output = pd.DataFrame({
        'id': [idx for idx in range(len(preds_all))],
        'pred_label': preds_all,
        'probs': probs_all
    })

    output.to_csv('/opt/ml/dataset/for_ensemble/output_cha.csv', index=False)