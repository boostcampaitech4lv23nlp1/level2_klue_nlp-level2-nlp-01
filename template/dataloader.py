import os
import re
import ast
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
    
        self.ner_type = {'ORG':'기관', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )


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


    def tokenizing(self, df: pd.DataFrame):
        data = []
    
        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
            sub_word = item['subject_entity']
            obj_word = item['object_entity']
            subject_entity = '@*' + self.ner_type[item['subject_type']] + '*' + item['subject_entity'] + '@'
            object_entity = '#^' + self.ner_type[item['object_type']] + '^' + item['object_entity'] + '#'
            # concat_entity = sub_word + '[SEP]' + obj_word                 # baseline
            # concat_entity = subject_entity + '[SEP]' + object_entity      # baseline(+ typed entity marker)
            # concat_entity = f'이 문장에서 {subject_entity}와 {object_entity}의 관계'   # concat.v1
            concat_entity = f'문장에서 @{obj_word}@와 #{sub_word}#의 관계를 고르세요.'  # concat.v2
            # concat_entity = f'문장에서 {subject_entity}와 {object_entity}의 관계를 고르세요.' # concat.v3
            # concat_entity = f'문장에서 @{obj_word}@와 #{sub_word}#건의 관계는?'

            outputs = self.tokenizer(
                concat_entity,
                item['sentence'],
                add_special_tokens=True, 
                padding='max_length',
                max_length=256,
                truncation=True
            )
            data.append(outputs)
        return data

    
    def add_entity_marker_punct(self, sentence, sub_word, obj_word, sub_type, obj_type, ss, se, os, oe):
        if ss < os:
            new_sentence = sentence[:ss] + '@'+ '*' + self.ner_type[sub_type] + '*' + sub_word + '@' + sentence[se + 1 : os] + '#' + '^' + self.ner_type[obj_type] + '^' + obj_word + '#' + sentence[oe + 1 :]
        else: 
            new_sentence = sentence[:os] + '#'+ '^' + self.ner_type[obj_type] + '^' + obj_word + '#' + sentence[oe + 1 : ss] + '@' + '*' + self.ner_type[sub_type] + '*' + sub_word + '@' + sentence[se + 1 :]
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
            self.predict_dataset = Dataset(predict_inputs, predict_targets)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,num_workers=8)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=8)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)


    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size= 1)