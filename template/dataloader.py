import os
import re
import argparse
import ast

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
    def __init__(self, tokenizer_name, batch_size, train_path, dev_path, test_path, predict_path, masking, shuffle):
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

        self.masking = masking
        self.shuffle = shuffle

        self.added_token_num = 0
        self.using_columns = ['subject_entity', 'object_entity', 'sentence']

        self.special_tokens = [
            '[O:ORG]', '[/O:ORG]', 
            '[/O:POH]', '[O:NOH]', 
            '[/O:PER]', '[O:LOC]', 
            '[O:PER]', '[/O:LOC]', 
            '[S:PER]', '[/S:PER]', 
            '[O:DAT]', '[/O:DAT]', 
            '[/O:NOH]', '[S:ORG]', 
            '[/S:ORG]', '[O:POH]'
        ]
        self.ner_tokens = {
            'ORG':'단체', 
            'PER':'사람', 
            'DAT':'날짜', 
            'LOC':'위치', 
            'POH':'기타', 
            'NOH':'수량'
        }

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )
        
        if self.masking is True:
            self.added_token_num += self.tokenizer.add_special_tokens({
                'additional_special_tokens': self.special_tokens
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

    def add_entity_token(self, item: pd.Series):
        '''
        ### Add Entity Token
        - "가수 로이킴(김상우·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다."
            - "가수 [S:PER]로이킴[/S:PER]([O:PER]김상우[/O:PER]·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다."
        
        - "김영록 전라남도지사를 비롯해 윤병태 정무부지사, 직원들이 폭력예방 교육을 받고 있다."
            - "#^사람^김영록# 전라남도지사를 비롯해 @*사람*윤병태@ 정무부지사, 직원들이 폭력예방 교육을 받고 있다."
        '''
        sentence = item['sentence']
        ids = item['ids']
        types = item['types']

        if self.masking is False:
            return '[SEP]'.join([item[column] for column in self.using_columns]), None
        else:
            # i = 0 -> subject entity
            # i = 1 -> object entity
            
            slide_size = 0
            so = ['S', 'O']
            attaches = []
            for i, entity in enumerate([item['subject_entity'], item['object_entity']]):
                special_token_pair = f'[{so[i]}:{types[i]}]', f'[/{so[i]}:{types[i]}]'
                attached = special_token_pair[0] + entity + special_token_pair[1]
                attaches.append(attached)
                sentence = sentence[:ids[i]+slide_size] + attached + sentence[ids[i]+len(entity)+slide_size:]

                if ids[0] < ids[1]:
                    slide_size += len(f'[{so[i]}:{types[i]}]' + f'[/{so[i]}:{types[i]}]')
            
            # slide_size = 0
            # so = ['@*', '#^']
            # attaches = []
            # for i, entity in enumerate([item['subject_entity'], item['object_entity']]):
            #     special_token_pair = f'{so[i]}{self.ner_tokens[types[i]]}{so[i][1]}', f'{so[i][0]}'
            #     attached = special_token_pair[0] + entity + special_token_pair[1]
            #     attaches.append(attached)

            #     sentence = sentence[:ids[i]+slide_size] + attached + sentence[ids[i]+len(entity)+slide_size:]
            #     if ids[0] < ids[1]:
            #         slide_size += len(f'{so[i]}{self.ner_tokens[types[i]]}{so[i][1]}' + f'{so[i][0]}')
        return sentence, attaches
    
    def add_entity_embeddings(self, outputs):
        entity_embeddings = []
        flag = 0
        
        for token_idx in outputs['input_ids']:
            entity_embeddings.append(flag)
            if self.tokenizer.decode(token_idx) in self.special_tokens:
                if flag == 0:
                    flag = 1
                else: 
                    flag = 0
                    entity_embeddings[-1] = flag
        outputs['token_type_ids'] += entity_embeddings
        return outputs

    def tokenizing(self, df: pd.DataFrame) -> List[dict]:
        data = []

        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
            # 실험 1. 로이킴[SEP]김상우[SEP]가수 로이킴(김상우·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다.
            # concat_entity, _ = self.add_entity_token(item)

            # 실험 2. "가수 [S:PER]로이킴[/S:PER]([O:PER]김상우[/O:PER]·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다."
            concat_entity, _ = self.add_entity_token(item)

            # 실험 2-1. [S:PER]로이킴[/S:PER],[O:PER]김상우[/O:PER]의 관계[SEP]가수 [S:PER]로이킴[/S:PER]([O:PER]김상우[/O:PER]·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다.
            concat_entity, attaches = self.add_entity_token(item)
            concat_entity = f'{attaches[0]},{attaches[1]}의 관계' + '[SEP]' + concat_entity

            # 실험 3. @*사람*로이킴@[SEP]#^사람^김상우#[SEP]가수 @*사람*로이킴@(#^사람^김상우#·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다.            
            # sentence, attaches = self.add_entity_token(item)
            # concat_entity = '[SEP]'.join(attached for attached in attaches) + '[SEP]'
            # concat_entity += sentence

            # 실험 4. 로이킴[SEP]김상우[SEP]가수 @*사람*로이킴@(#^사람^김상우#·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다.
            # sentence, attaches = self.add_entity_token(item)
            # concat_entity = '[SEP]'.join(item[columns] for columns in self.using_columns[:2]) + '[SEP]'
            # concat_entity += sentence

            outputs = self.tokenizer(
                concat_entity, 
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=256
            )
            
            # entity_embedding -> entity 토큰 있는 위치마다 표시
            # outputs = self.add_entity_embeddings(outputs)
            data.append(outputs)
        return data
    
    def preprocessing(self, df: pd.DataFrame):
        subject_entities = []
        object_entities = []
        ids = []
        types = []

        for sub, obj in tqdm(zip(df['subject_entity'], df['object_entity'])):
            # 보안 검증 : https://docs.python.org/3/library/ast.html
            subject_entity = ast.literal_eval(sub)
            object_entity = ast.literal_eval(obj)

            subject_entities.append(subject_entity['word'])
            object_entities.append(object_entity['word'])
            ids.append((subject_entity['start_idx'], object_entity['start_idx']))
            types.append((subject_entity['type'], object_entity['type']))
        
        try:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': df['sentence'],
                'subject_entity': subject_entities,
                'object_entity': object_entities,
                'ids': ids,
                'types': types,
                'label': df['label'],
            })
            
            inputs = self.tokenizing(preprocessed_df)
            targets = self.label_to_num(preprocessed_df['label'])
        except:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': df['sentence'],
                'subject_entity': subject_entities,
                'object_entity': object_entities,
                'ids': ids,
                'types': types,
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
