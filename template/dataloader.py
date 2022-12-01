import os
import re
import argparse
import ast

from typing import *
import json
import torch
import torchmetrics
import pandas as pd
import transformers
from tqdm.auto import tqdm
import pytorch_lightning as pl
import pickle as pkl

import metrics

from sklearn.model_selection import StratifiedKFold

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
    def __init__(self, tokenizer_name, batch_size, train_path, dev_path, test_path, predict_path, masking, shuffle, k, n_folds, split_seed):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

        self.train_path = train_path # train
        self.dev_path = dev_path # val
        self.test_path = test_path # val
        self.predict_path = predict_path # test

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.shuffle = shuffle

        self.masking = masking
        self.shuffle = shuffle
        self.k = k
        self.n_folds = n_folds
        self.split_speed = split_seed

        
        self.ner_type = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
        self.using_columns = {'subject_entity':'', 'object_entity':''}
        # self.entity_tokens = ['[ENTITY]', '[/ENTITY]']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )
        # self.tokenizer.add_special_tokens({
        #     'additional_special_tokens': ['[ENTITY]', '[/ENTITY]']
        # })


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


    def tokenizing(self, df: pd.DataFrame) -> List[dict]:
        data = []
    
        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
            sub_word = item['subject_entity']
            obj_word = item['object_entity']
            subject_entity = '@*' + self.ner_type[item['subject_type']] + '*' + item['subject_entity'] + '@'
            object_entity = '#^' + self.ner_type[item['object_type']] + '^' + item['object_entity'] + '#'
            concat_entity = f'문장에서 @{obj_word}@와 #{sub_word}#건의 관계는?' # concat.v4
            # concat_entity = subject_entity+ '[SEP]' + object_entity
            # concat_entity = f'이 문장에서 {subject_entity}와 {object_entity}의 관계'

            outputs = self.tokenizer(
                concat_entity,
                item['sentence'],
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=256
            )
            data.append(outputs)
        return data
    
    def add_entity_marker_punct(self, sentence, sub_word, obj_word, sub_type, obj_type, ss, se, os, oe):
        if ss < os:
            new_sentence = sentence[:ss] + '@'+ '*' + self.ner_type[sub_type] + '*' + sub_word + '@' + sentence[se + 1 : os] + '#' + '^' + self.ner_type[obj_type] + '^' + obj_word + '#' + sentence[oe + 1 :]
        else: 
            new_sentence = sentence[:os] + '#'+ '^' + self.ner_type[obj_type] + '^' + obj_word + '#' + sentence[oe + 1 : ss] + '@' + '*' + self.ner_type[sub_type] + '*' + sub_word + '@' + sentence[se + 1 :]
            #new_sentence = sentence[:os] + '@'+ '*' + self.ner_type[obj_type] + '*' + obj_word + '@' + sentence[oe + 1 : ss] + '#' + '^' + self.ner_type[sub_type] + '^' + sub_word + '#' + sentence[se + 1 :]
    
        return new_sentence
    
    def preprocessing(self, df: pd.DataFrame):
        subject_entities = []
        object_entities = []
        sentence = []
        subject_type = []
        object_type = []  
        
        for sub,obj,sent in zip(df['subject_entity'], df['object_entity'], df['sentence']):

            subject_entity = ast.literal_eval(sub)
            object_entity = ast.literal_eval(obj)

            sub_word = subject_entity['word']
            obj_word = object_entity['word']

            ss = subject_entity['start_idx']
            se = subject_entity['end_idx']
            os = object_entity['start_idx']
            oe = object_entity['end_idx']
            sub_type = subject_entity['type']
            obj_type = object_entity['type']

            prepro_sent = self.add_entity_marker_punct(sent, sub_word, obj_word, sub_type, obj_type, ss, se, os, oe)

            subject_entities.append(sub_word)
            object_entities.append(obj_word)
            sentence.append(prepro_sent)
            subject_type.append(sub_type)
            object_type.append(obj_type)
   
        try:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': sentence,
                'subject_entity': subject_entities,
                'object_entity': object_entities,
                'subject_type' : subject_type,
                'object_type' : object_type,
                'label': df['label'],
            })
            
            inputs = self.tokenizing(preprocessed_df)
            targets = self.label_to_num(preprocessed_df['label'])

        except:
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': sentence,
                'subject_entity': subject_entities,
                'object_entity': object_entities,
                'subject_type' : subject_type,
                'object_type' : object_type,
            })
            inputs = self.tokenizing(preprocessed_df)
            targets = []
    
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

        ####################### << K-Fold >> #######################
            total_data = pd.concat([train_data, val_data])
            total_data.set_index('Unnamed: 0', inplace=True)
            data_label = total_data['label']

            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.split_speed)

            for n_fold, (_, v_idx) in enumerate(skf.split(total_data, total_data['label'])):
                total_data.loc[v_idx, 'fold'] = n_fold
            
            train_data = total_data[total_data['fold'] != self.k]
            val_data = total_data[total_data['fold'] == self.k]
        ###############################################################

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)    

        else:
            # test_data = pd.read_csv(self.test_path)    
            predict_data = pd.read_csv(self.predict_path)

            # test_inputs, test_targets = self.preprocessing(test_data)
            # self.test_dataset = Dataset(test_inputs, test_targets)
            self.test_dataset = self.val_dataset

            predict_data.drop(columns=['label'], inplace=True)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=8)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
