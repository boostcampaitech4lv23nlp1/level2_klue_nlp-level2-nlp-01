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
import matplotlib.pyplot as plt
from transformers import pipeline
import itertools

def plot_confusion_matrix(cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', default='klue/bert-base', type=str)
    parser.add_argument('--model_name', default='klue/bert-base', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/dataset/train/new_train_split.csv')
    parser.add_argument('--dev_path', default='/opt/ml/dataset/train/new_val_split.csv')
    parser.add_argument('--test_path', default='/opt/ml/dataset/train/new_val_split.csv')
    parser.add_argument('--predict_path', default='/opt/ml/dataset/train/new_val_split.csv')
    args = parser.parse_args(args=[])


    dataloader = Dataloader(args.tokenizer_name,
                            args.batch_size,
                            args.train_path,
                            args.dev_path,
                            args.test_path,
                            args.predict_path,
                            shuffle=True)
    
    model = Model(args.model_name, args.learning_rate)

    trainer = pl.Trainer(accelerator = 'gpu',
                        devices = 1,
                        max_epochs=args.max_epoch, 
                        log_every_n_steps=1,
                        precision=16
                        )

    model = Model.load_from_checkpoint(checkpoint_path = '/opt/ml/template/models/roberta-large+epoch=0+val_micro_f1=83.147.ckpt')
    results = trainer.predict(model = model, datamodule=dataloader)
    print(results)
    cm = np.zeros([30,30])
    y_predicted, y_true = [], []
    for preds, probs in results:
        cm[preds[0]][probs[0]] += 1
        y_predicted.append(preds[0]); y_true.append(probs[0])
    cm = cm.astype('int')

    label = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion'] # 라벨 설정
   
    plot_confusion_matrix(cm,
                      normalize = False,
                      target_names = label,
                      title = "Confusion Matrix")