import re
import os
import argparse

import torch
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torchmetrics
import pytorch_lightning as pl

from dataloader import *


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/dataset/train/train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/dataset/train/val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])

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
        max_epochs=args.max_epoch, 
        log_every_n_steps=1
    )

    model_name = re.sub(r'[/]', '-', args.model_name)
    model = torch.load(f'{model_name}.pt')

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

    output.to_csv('output.csv', index=False)