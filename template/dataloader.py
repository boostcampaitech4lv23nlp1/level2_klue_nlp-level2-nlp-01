import os
import re
import argparse
import random
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
    def __init__(self, tokenizer_name, batch_size, train_path, dev_path, test_path, predict_path, masking, augmented, shuffle):
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

        self.masking: bool = masking
        self.augmented: bool = augmented
        self.shuffle: bool = shuffle

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
        
        # if self.masking is True:
        #     self.added_token_num += self.tokenizer.add_special_tokens({
        #         'additional_special_tokens': self.special_tokens
        #     })

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
            so = ['@*', '#^']
            attaches = []
            for i, entity in enumerate([item['subject_entity'], item['object_entity']]):
                special_token_pair = f'{so[i]}{self.ner_tokens[types[i]]}{so[i][1]}', f'{so[i][0]}'
                attached = special_token_pair[0] + entity + special_token_pair[1]
                attaches.append(attached)

                sentence = sentence[:ids[i]+slide_size] + attached + sentence[ids[i]+len(entity)+slide_size:]
                if ids[0] < ids[1]:
                    slide_size += len(f'{so[i]}{self.ner_tokens[types[i]]}{so[i][1]}' + f'{so[i][0]}')
        return sentence, attaches
    
    def add_entity_embeddings(self, outputs):
        entity_embeddings = []
        flag = 0
        
        # 32001 ~ 
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
    
    def add_aeda_sentence(self, sentence: str, ratio=0.1) -> str:
        '''
        s : subject word
        o : object word
        sentence : 원래 문장
        '''
        punctuations = ['.', ',', '!', '?', ';', ':']  # len(punctuations) : 6
        count = int(len(sentence) * ratio)
    
        subject_range = [i for i in range(*re.search(r'\@.*\@', sentence).span())]
        object_range = [i for i in range(*re.search(r'\#.*\#', sentence).span())]

        sentence = list(sentence)
        sentence_ids = [i for i in range(len(sentence)) if i not in subject_range+object_range]
        for _ in range(count):
            sentence_idx = sentence_ids.pop(random.randint(0, len(sentence_ids)-1))
            sentence[sentence_idx] += punctuations[random.randint(0, len(punctuations)-1)]
        sentence = ''.join(sentence)
        return sentence

    def tokenizing(self, df: pd.DataFrame, augmented_num) -> List[dict]:
        data = []
        
        if augmented_num > 1:
            print("augmentation start")
        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):     
            concat_entity, attaches = self.add_entity_token(item)

            concat_entities = []
            concat_entities.append(f'{attaches[0]},{attaches[1]}건의 관계는?' + '[SEP]' + concat_entity)

            for _ in range(augmented_num-1):
                concat_entities.append(f'{attaches[0]},{attaches[1]}건의 관계는?' + '[SEP]' + self.add_aeda_sentence(concat_entity))
 
            concat_entity = []
            for s in concat_entities:
                concat_entity.append(s)

            assert isinstance(concat_entity, str) or isinstance(concat_entity, list), "str, list만 지원합니다."
            if isinstance(concat_entity, str):
                outputs = self.tokenizer(
                    concat_entity, 
                    add_special_tokens=True, 
                    padding='max_length',
                    truncation=True,
                    max_length=256
                )
                data.append(outputs)
            else:
                for s in concat_entity:
                    outputs = self.tokenizer(
                        s, 
                        add_special_tokens=True, 
                        padding='max_length',
                        truncation=True,
                        max_length=256
                    )
                    data.append(outputs)
        return data
    
    def preprocessing(self, df: pd.DataFrame, augmented_num=1):
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
            if self.augmented is False:
                augmented_num = 1
            inputs = self.tokenizing(preprocessed_df, augmented_num)

            labels = []
            assert augmented_num == (len(inputs) // len(preprocessed_df)), 'augmentation error'

            for i in range(len(preprocessed_df)):
                labels += [preprocessed_df['label'][i] for _ in range(augmented_num)]

            result_inputs = []
            result_labels = []
            for i in range(0, len(labels)-augmented_num, augmented_num):
                result_inputs.append(inputs[i]); result_labels.append(labels[i])
                if labels[i] != 'no_relation':
                    for j in range(i+1, i+augmented_num):
                        result_inputs.append(inputs[j]); result_labels.append(labels[j])

            inputs = result_inputs
            targets = self.label_to_num(pd.Series(result_labels))        
               
        except BaseException as e:
            print(f"BaseException : {e}\n")
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': df['sentence'],
                'subject_entity': subject_entities,
                'object_entity': object_entities,
                'ids': ids,
                'types': types,
            })
            inputs = self.tokenizing(preprocessed_df, augmented_num)
            targets = []

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data, augmented_num=2)
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
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
