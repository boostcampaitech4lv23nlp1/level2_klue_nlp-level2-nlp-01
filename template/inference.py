import os
import argparse
from models import Model
import torch
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torchmetrics
import pytorch_lightning as pl

from dataloader import Dataloader


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/roberta-large', type=str)
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)

    parser.add_argument('--marker', default=True, type=bool)
    parser.add_argument('--pooling', default=True, type=bool)
    parser.add_argument('--criterion', default='cross_entropy', type=str)  # cross_entropy, focal_loss
    parser.add_argument('--shuffle', default=True, type=bool)
    
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--n_folds', default=5, type=bool)
    parser.add_argument('--split_seed', default=42, type=bool)

    parser.add_argument('--train_path', default='/opt/ml/dataset/train/new_train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/dataset/train/new_val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/dataset/train/new_val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/test/test_data.csv')
    args = parser.parse_args(args=[])
    
    dataloader = Dataloader(args.tokenizer_name, args.batch_size,
                                args.train_path, args.dev_path, args.test_path, args.predict_path, args.marker, args.shuffle,
                                args.k, args.n_folds, args.split_seed)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=args.max_epoch, 
        log_every_n_steps=1
    )

    use_files = []
    PATH = '/opt/ml/template/kfolds/'  # kfold ckpt 파일 경로
    for idx, file_nm in enumerate(os.listdir(PATH)):
        use_files.append(PATH + file_nm)
    use_files
    
    for idx, file_nm in enumerate(use_files):
        print(f'=======================fold {idx}==========================')
        model_name = file_nm
        model = Model.load_from_checkpoint(checkpoint_path=model_name,strict=False)

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

        output.to_csv(f'/opt/ml/template/kfold_result/kfold_output_{idx}.csv', index=False) # 저장 경로
        print(f"=======================fold {idx}th DONE=======================")