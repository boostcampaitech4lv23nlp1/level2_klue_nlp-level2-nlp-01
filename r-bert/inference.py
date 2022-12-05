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
from models import *

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


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

    parser.add_argument('--masking', default=True, type=bool)
    parser.add_argument('--pooling', default=False, type=bool)
    parser.add_argument('--criterion', default='focal_loss', type=str)  # cross_entropy, focal_loss

    parser.add_argument('--train_path', default='../dataset/train/new_train_split.csv')
    parser.add_argument('--dev_path', default='../dataset/train/new_val_split.csv')
    parser.add_argument('--test_path', default='../dataset/train/new_val_split.csv')
    parser.add_argument('--predict_path', default='../dataset/test/test_data.csv')
    args = parser.parse_args(args=[])

    dataloader = Dataloader(
        args.tokenizer_name,
        args.batch_size,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        args.masking,
        shuffle=False
    )
    
    model_name = '/opt/ml/models/rbert-crossentropy-new_train/roberta-large+epoch=3+val_micro_f1=86.259.ckpt'
    model = Model(
        args.model_name, 
        args.learning_rate,
        args.pooling,
        args.criterion
    )

    # tracking special tokens
    model.model.resize_token_embeddings(
            dataloader.tokenizer.vocab_size + dataloader.added_token_num
        )

    model = weights_update(
        model, 
        torch.load(model_name)
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

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

    output.to_csv('/opt/ml/models/rbert-crossentropy-new_train/output.csv', index=False)