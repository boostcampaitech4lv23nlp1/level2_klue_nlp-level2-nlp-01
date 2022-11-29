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

import losses
import metrics
from dataloader import *

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class Model(pl.LightningModule):
    def __init__(self, model_name:str, lr: float, pooling: bool, criterion: str) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.pooling = pooling
        self.epsilon = 0.1

        self.labels_all = []
        self.preds_all = []
        self.probs_all = []

        self.model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )

        self.entity_fc_layer = FCLayer(input_dim = 1024, output_dim = 1024, dropout_rate = 0.1)

        self.fc_layer = FCLayer(input_dim=1024*3, output_dim=1024, dropout_rate = 0.1)

        self.classification = torch.nn.Linear(1024, 30)

        self.classification_for_rbert = torch.nn.Linear(1024*3, 30)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = losses.FocalLoss()
        

    # reference : https://stackoverflow.com/questions/65083581/how-to-compute-mean-max-of-huggingface-transformers-bert-token-embeddings-with-a
    def mean_pooling(self, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output['last_hidden_state']        #First element of model_output contains all token embeddings
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def entity_average(self, hidden_output: torch.Tensor, e_mask: torch.Tensor) -> torch.Tensor:
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1) # torch.Size([16, 1, 256])
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [16, 1]

        # hidden_unsqueeze = hidden_output.unsqueeze(1) # torch.Size([16, 1, 1024])
        # print(hidden_unsqueeze.size())

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1) # batch matrix multiplication
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector



    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:

        inputs = {
            'input_ids' : x['input_ids'],
            'token_type_ids' : x['token_type_ids'],
            'attention_mask' : x['attention_mask']
        }
        e_mask_s = x['entity_mask_S']
        e_mask_o = x['entity_mask_O']

        model_outputs = self.model(**inputs)
        hidden_output = model_outputs[0] # torch.Size([16, 256, 1024]) 
        pooled_output = model_outputs[1] # [CLS] torch.Size([16, 1024])

        entity_s_h = self.entity_average(hidden_output, e_mask_s) # torch.Size([16, 1024])
        entity_o_h = self.entity_average(hidden_output, e_mask_o) # torch.Size([16, 1024])
        
        if self.pooling is True:
            sentence_out = self.mean_pooling(model_outputs, x['attention_mask']) # torch.Size([16, 1024])
            
        else:
            sentence_out = model_outputs['last_hidden_state'][:, 0, :] # [CLS] torch.Size([16, 1024])
            # out = pooled_output
            # print(out==pooled_output) # False

        entity_s_out = self.entity_fc_layer(entity_s_h)
        entity_o_out = self.entity_fc_layer(entity_o_h)

        # concat
        concat_h = torch.cat([sentence_out, entity_s_out, entity_o_out], dim = -1) # torch.Size([16, 3072])

        
        
        out = self.classification_for_rbert(concat_h)
        
        #TODO: fc layer -> classification 실험
        # concat_h = self.fc_layer(concat_h)
        # out = self.classification(concat_h)
        # 성능 하락...
        return out
    
    def CrossEntropywithLabelSmoothing(self,pred,target):
        K = pred.size(-1) # 전체 클래스의 갯수
        log_probs = F.log_softmax(pred, dim=-1)
        avg_log_probs = (-log_probs).sum(-1).mean()
        ce_loss = F.nll_loss(log_probs, target)
        ce_loss_w_soft_label = (1-self.epsilon) * ce_loss + self.epsilon / K * (avg_log_probs)
        return ce_loss_w_soft_label

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer