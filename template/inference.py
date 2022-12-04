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
from models import Model


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    
    parser.add_argument('--masking', default=True, type=bool)
    parser.add_argument('--pooling', default=True, type=bool)
    parser.add_argument('--criterion', default='cross_entropy', type=str, help='cross_entropy, focal_loss')

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

    # model_name = re.sub(r'[/]', '-', args.model_name)
    checkpoint_path = '/opt/models/' + 'roberta-large+epoch=4+val_micro_f1=85.341.ckpt'
    
    # model = torch.load(f'/opt/models/{model_name}.pt')
    main_model = Model(
        args.model_name, 
        args.learning_rate,
        args.pooling,
        args.criterion
    )

    main_model.model.resize_token_embeddings(
            dataloader.tokenizer.vocab_size + dataloader.added_token_num
    )

    main_model = weights_update(
        main_model, 
        torch.load(checkpoint_path)
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1
    )

    results = trainer.predict(model=main_model, datamodule=dataloader)
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