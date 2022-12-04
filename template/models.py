import argparse
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import torchmetrics
import transformers
from tqdm.auto import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

import losses
import metrics
from dataloader import *
import losses

class Model(pl.LightningModule):
    def __init__(self, model_name:str, lr: float, pooling: bool, criterion: str) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.pooling = pooling
        
        self.labels_all = []
        self.preds_all = []
        self.probs_all = []

        self.model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )

        self.classification = torch.nn.Linear(1024, 30)

        assert criterion in ['cross_entropy', 'focal_loss'], "criterion not in model"
        if criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion == 'focal_loss':
            self.criterion = losses.FocalLoss()

    # reference : https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a
    def mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output['last_hidden_state']        #First element of model_output contains all token embeddings
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_outputs = self.model(**x)
        
        if self.pooling is True:
            out = self.mean_pooling(model_outputs, x['attention_mask'])
        else:
            out = model_outputs['last_hidden_state'][:, 0, :]
        out = self.classification(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        labels = y.cpu().detach().numpy().tolist()
        probs = logits.cpu().detach().numpy().tolist()
        preds = logits.cpu().detach().numpy().argmax(-1).tolist()
        
        self.labels_all += labels
        self.probs_all += probs
        self.preds_all += preds
        return loss

    def validation_epoch_end(self, output) -> None:
        labels = np.asarray(self.labels_all)
        probs = np.asarray(self.probs_all)
        preds = np.asarray(self.preds_all)

        self.log("val_micro_f1", metrics.klue_re_micro_f1(preds, labels))
        self.log("val_re_auprc", metrics.klue_re_auprc(probs, labels))
        self.log("val_acc", metrics.re_accuracy_score(labels, preds))

        self.labels_all.clear()
        self.probs_all.clear()
        self.preds_all.clear()
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        labels = y.cpu().detach().numpy().tolist()
        probs = logits.cpu().detach().numpy().tolist()
        preds = logits.cpu().detach().numpy().argmax(-1).tolist()

        self.labels_all += labels
        self.probs_all += probs
        self.preds_all += preds

    def test_epoch_end(self, outputs):
        labels = np.asarray(self.labels_all)
        probs = np.asarray(self.probs_all)
        preds = np.asarray(self.preds_all)

        self.log("test_micro_f1", metrics.klue_re_micro_f1(preds, labels))
        self.log("test_re_auprc", metrics.klue_re_auprc(probs, labels))
        self.log("test_acc", metrics.re_accuracy_score(labels, preds))

        self.labels_all.clear()
        self.probs_all.clear()
        self.preds_all.clear()
        return

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
        logits = logits.detach().cpu().numpy()
        preds = logits.argmax(-1).tolist()

        return preds, probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
    
        return optimizer