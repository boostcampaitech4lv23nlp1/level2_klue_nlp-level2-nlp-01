import argparse
from copy import deepcopy
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchmetrics
import transformers
from tqdm.auto import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
import losses
import metrics
from dataloader_binary import *

class Model(pl.LightningModule):
    def __init__(self, model_name:str, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.pooling = True
        self.contrastive = True
        self.epsilon = 0.1

        self.model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )
        
        self.classification = torch.nn.Linear(1024,1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        
    # reference : https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a
    def mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output['last_hidden_state']        #First element of model_output contains all token embeddings
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_outputs = self.model(**x)    #model_outputs -> (32,256,1024)
        
        if self.pooling is True:
            hidden_state = self.mean_pooling(model_outputs, x['attention_mask'])
        else:
            hidden_state = model_outputs['last_hidden_state'][:, 0, :] #(32,1,1024)
        out = self.classification(hidden_state)

        return out, hidden_state

    def contrastive_loss(self, embedding, label, temp=0.3):
        """calculate the contrastive loss
        """
        embedding = embedding.cpu().detach().numpy()
        # cosine similarity between embeddings
        cosine_sim = cosine_similarity(embedding, embedding)
        # remove diagonal elements from matrix
        dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
        # apply temprature to elements
        dis = dis / temp
        cosine_sim = cosine_sim / temp
        # apply exp to elements
        dis = np.exp(dis)
        cosine_sim = np.exp(cosine_sim)

        # calculate row sum
        row_sum = []
        for i in range(len(embedding)):
            row_sum.append(sum(dis[i]))
        # calculate outer sum
        contrastive_loss = 0
        for i in range(len(embedding)):
            n_i = label.tolist().count(label[i]) - 1
            inner_sum = 0
            # calculate inner sum
            for j in range(len(embedding)):
                if abs(label[i] - label[j])<=0.1 and i != j: # can fix
                    inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
            if n_i != 0:
                contrastive_loss += (inner_sum / (-n_i))
            else:
                contrastive_loss += 0
        return contrastive_loss
        

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, hidden_state = self(x)
    
        if self.contrastive is True:
            con_loss = self.contrastive_loss(hidden_state, y.float())
            cross_loss = self.criterion(logits.view(-1), y.float())
            lam = 0.9 # can fix
            loss = (lam * con_loss) + (1 - lam) * (cross_loss)
        else:
            loss = self.criterion(logits.view(-1),y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits, hidden_state = self(x)
        if self.contrastive is True:
            con_loss = self.contrastive_loss(hidden_state, y.float())
            cross_loss = self.criterion(logits.view(-1), y.float())
            lam = 0.9 # can fix
            loss = (lam * con_loss) + (1 - lam) * (cross_loss)
        else:
            loss = self.criterion(logits.view(-1),y.float())
        self.log("val_loss", loss)
        
        probs = torch.sigmoid(logits).squeeze(-1)
        labels = y.cpu().detach().numpy().tolist()
        preds = []
        preds.extend( probs.ge(0.5).int().tolist())
        
        self.log('compute_f1', metrics.compute_f1(preds, labels))
        self.log('accuracy', metrics.simple_accuracy(preds, labels))
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits,hidden_state = self(x)
       

        probs = torch.sigmoid(logits).squeeze(-1)
        labels = y.cpu().detach().numpy().tolist()
        preds = []
        preds.extend( probs.ge(0.5).int().tolist())
        
        self.log('compute_f1', metrics.compute_f1(preds, labels))
        self.log('accuracy', metrics.simple_accuracy(preds, labels))
        
        return 

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits,hidden_state = self(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        final_preds = []
        final_preds.extend(probs.ge(0.5).int().tolist())
        
        return final_preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = 0.01)
        lr_scheduler = {'scheduler': OneCycleLR(optimizer=optimizer, max_lr=self.lr, steps_per_epoch=912,epochs=5,pct_start=0.1,anneal_strategy='cos'),
        'interval': 'step','frequency': 1}
        return [optimizer], [lr_scheduler]
