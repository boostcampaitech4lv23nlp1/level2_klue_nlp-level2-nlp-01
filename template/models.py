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


# https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
class CustomEmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(CustomEmbeddingLayer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.token_type_embeddings = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)  # nn.Embedding(1, 1024)
        self.entity_embeddings = torch.nn.Embedding(self.num_embeddings+1, self.embedding_dim)    # nn.Embedding(2, 1024)
    
    def forward(self, concat_embeddings):
        vector_size = concat_embeddings.size(-1) // 2
        token_type_ids = concat_embeddings[:, :vector_size] # 모든 값이 0이기 때문에 embedding layer가 굳이 있어야 하나 생각
        entity_ids = concat_embeddings[:, vector_size:]

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)

        
        return token_type_embeddings + entity_embeddings


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
        # self.model.embeddings.token_type_embeddings = CustomEmbeddingLayer(1, 1024)     # nn.Embedding(1, 1024)

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_size = 1024
        self.hidden_size = 1024
        self.num_layers = 2
        self.num_dirs = 2
        dropout = 0.1

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dense_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size*self.num_dirs*2, self.hidden_size*self.num_dirs*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.hidden_size*self.num_dirs*2, self.hidden_size),
            torch.nn.ReLU(),
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
            # out = model_outputs['last_hidden_state'][:, 0, :]
            out = model_outputs['last_hidden_state']
            h_0 = torch.zeros(self.num_layers*self.num_dirs, out.size(0), self.hidden_size).to(self.device_)
            c_0 = torch.zeros(self.num_layers*self.num_dirs, out.size(0), self.hidden_size).to(self.device_)
            out, _ = self.lstm(out, (h_0, c_0))
            out = torch.cat((out[:, 0, :], out[:, -1, :]), dim=-1)
            out = self.dense_layer(out)

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
        # lr_scheduler = {
        #     'scheduler': OneCycleLR(
        #         optimizer=optimizer,
        #         max_lr=3e-5,
        #         steps_per_epoch=912,
        #         epochs=5,
        #         pct_start=0.1
        #     ),
        #     'interval': 'step',
        #     'frequency': 1
        # }
        return optimizer