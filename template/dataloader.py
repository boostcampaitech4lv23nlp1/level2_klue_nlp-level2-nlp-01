import os
import re
import argparse
from typing import *

import torch
import torchmetrics
import pandas as pd
import transformers
from tqdm.auto import tqdm
import pytorch_lightning as pl
import pickle as pkl

import metrics

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: List[dict], labels: List[int]):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx) -> dict:
        X = {key: torch.tensor(value) for key, value in self.inputs[idx].items()}

        # prediction은 label이 없음
        try:
            y = torch.tensor(self.labels[idx])
        except:
            y = torch.tensor(-1)
        return X, y

    def __len__(self):
        return len(self.inputs)

class Dataloader(pl.LightningDataModule):
    def __init__(self, tokenizer_name, batch_size, train_path, dev_path, test_path, predict_path, shuffle):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.shuffle = shuffle

        # self.using_columns = ['subject_entity', 'object_entity', 'sentence']
        self.using_columns = ['sentence', 'subject_entity', 'subject_start', 'subject_end', 'object_entity', 'object_start', 'object_end']
        self.entity_tokens = ['[ENTITY]', '[/ENTITY]']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[ENTITY]', '[/ENTITY]']
        })

    def num_to_label(self, label):
        origin_label = []
        with open('dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pkl.load(f)
        for v in label:
            origin_label.append(dict_num_to_label[v])
        return origin_label

    def label_to_num(self, label: pd.Series) -> List[int]:
        num_label = []
        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pkl.load(f)
        for v in label:
            num_label.append(dict_label_to_num[v])
        
        return num_label

    def add_entity_token(self, sentence, subject_entity, subject_start, subject_end, object_entity, object_start, object_end):
        if subject_end <= object_start:
            sentence = sentence[:subject_start] + '<sub>' + sentence[subject_start:subject_end] + '</sub>' + sentence[subject_end:]
            sentence = sentence[:object_start+11] + '<obj>' + sentence[object_start+11:object_end+11] + '</obj>' + sentence[object_end+11:]
        else:
            sentence = sentence[:object_start] + '<obj>' + sentence[object_start:object_end] + '</obj>' + sentence[object_end:]
            sentence = sentence[:subject_start+11] + '<sub>' + sentence[subject_start+11:subject_end+11] + '</sub>' + sentence[subject_end+11:]
            

        return sentence

    def tokenizing(self, df: pd.DataFrame) -> List[dict]:
        data = []
        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
            item = [item[column] for column in self.using_columns]
            item = self.add_entity_token(*item)
            # concat_entity = '[SEP]'.join([item[column] for column in self.using_columns])
            # concat_entity = '[SEP]'.join([item[column] for column in self.using_columns[:-1]])
            # concat_entity += '[SEP]' + self.add_entity_token(item['sentence'], item['subject_entity'], item['object_entity'])
            outputs = self.tokenizer(
                item, 
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=256
            )
            data.append(outputs)
        return data

    def preprocessing(self, df: pd.DataFrame):
        subject_entity = []
        object_entity = []

        subject_start = []
        object_start = []

        subject_end = []
        object_end = []

        subject_tag = []
        object_tag = []

        for sub, obj in tqdm(zip(df['subject_entity'], df['object_entity'])):
            sub = eval(sub)
            sub_text = sub['word']
            sub_idx_start = sub['start_idx']
            sub_idx_end = sub['end_idx']+1
            sub_type = sub['type']

            obj = eval(obj)
            obj_text = obj['word']
            obj_idx_start = obj['start_idx']
            obj_idx_end = obj['end_idx']+1
            obj_type = obj['type']
            # sub_tmp = sub[1:-1].split(', ')
            # sub_text = sub_tmp[0].split(':')[1].strip()[1:-1]
            # sub_idx_start = int(sub_tmp[1].split(':')[1].strip())
            # sub_idx_end = int(sub_tmp[2].split(':')[1].strip()) + 1
            # sub_type = sub_tmp[3].split(':')[1].strip().strip("'")

            # obj_tmp = obj[1:-1].split(', ')
            # obj_text = obj_tmp[0].split(':')[1].strip()[1:-1]
            # print(obj)
            # print(obj_tmp)
            # obj_idx_start = int(obj_tmp[1].split(':')[1].strip())
            # obj_idx_end = int(obj_tmp[2].split(':')[1].strip()) + 1
            # obj_type = obj_tmp[3].split(':')[1].strip().strip("'")

            subject_entity.append(sub_text)
            object_entity.append(obj_text)

            subject_start.append(sub_idx_start)
            object_start.append(obj_idx_start)

            subject_end.append(sub_idx_end)
            object_end.append(obj_idx_end)

            subject_tag.append(sub_type)
            object_tag.append(obj_type)
        
        try:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': df['sentence'],
                'subject_entity': subject_entity,
                'subject_start': subject_start,
                'subject_end' : subject_end,
                'subject_tag' : subject_tag,
                'object_entity': object_entity,
                'object_start': object_start,
                'object_end' : object_end,
                'object_tag' : object_tag,
                'label': df['label'],
            })
            
            inputs = self.tokenizing(preprocessed_df)
            targets = self.label_to_num(preprocessed_df['label'])
        except:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': df['sentence'],
                'subject_entity': subject_entity,
                'subject_start': subject_start,
                'subject_end' : subject_end,
                'subject_tag' : subject_tag,
                'object_entity': object_entity,
                'object_start': object_start,
                'object_end' : object_end,
                'object_tag' : object_tag
            })
            inputs = self.tokenizing(preprocessed_df)
            targets = []

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)    

        else:
            test_data = pd.read_csv(self.test_path)    
            predict_data = pd.read_csv(self.predict_path)

            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data.drop(columns=['label'], inplace=True)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
