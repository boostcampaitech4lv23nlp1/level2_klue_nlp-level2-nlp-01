import os
import re
import argparse
import ast
import sys

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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: List[dict], labels: List[int]):
        self.inputs = inputs
        self.labels = labels
        
        # if labels :
        #     self.labels = labels
        # else:
        #     self.labels = []

    def __getitem__(self, idx) -> dict:
        
        X = {key: torch.tensor(value) for key, value in self.inputs[idx].items()}

        # prediction은 label이 없음
        try:
            y = torch.tensor(self.labels[idx])
        except:
            y = torch.tensor(-1)
        return X, y

#         X = self.inputs[idx]
#         for k,v in X.items():
#             X[k] = torch.tensor(v, dtype=torch.long)

#         if len(self.labels) == 0 :
#             return X

#         else:
#             y = torch.tensor(self.labels[idx],dtype=torch.long)
#             return X, y

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

<<<<<<< HEAD:r-bert/dataloader.py
        self.added_token_num = 0
        self.using_columns = ['subject_entity', 'object_entity', 'sentence']

        self.special_tokens = [
            '[S:ORG]', '[/S:ORG]', # 0, 1
            '[S:PER]', '[/S:PER]', # 2, 3
            '[S:POH]', '[/S:POH]', # 4, 5
            '[S:LOC]', '[/S:LOC]', # 6, 7
            '[S:DAT]', '[/S:DAT]', # 8, 9
            '[S:NOH]', '[/S:NOH]', # 10, 11
            '[O:ORG]', '[/O:ORG]', # 12, 13
            '[O:PER]', '[/O:PER]', # 14, 15
            '[O:POH]', '[/O:POH]', # 16, 17
            '[O:LOC]', '[/O:LOC]', # 18, 19
            '[O:DAT]', '[/O:DAT]', # 20, 21
            '[O:NOH]', '[/O:NOH]' # 22, 23
        ]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )
        
        if self.masking is True:
            self.added_token_num += self.tokenizer.add_special_tokens({
                'additional_special_tokens': self.special_tokens
            })
=======
        
        self.ner_type = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
        self.using_columns = {'subject_entity':'', 'object_entity':''}
        # self.entity_tokens = ['[ENTITY]', '[/ENTITY]']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
        )
        # self.tokenizer.add_special_tokens({
        #     'additional_special_tokens': ['[ENTITY]', '[/ENTITY]']
        # })

>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py

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

<<<<<<< HEAD:r-bert/dataloader.py
    def add_entity_token(self, item: pd.Series, method='tem'):
        '''
        ### Add Entity Token
        "가수 로이킴(김상우·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다."
        "가수 [S:PER]로이킴[/S:PER]([O:PER]김상우[/O:PER]·26)의 음란물 유포 혐의 '비하인드 스토리'가 공개됐다."
        '''
        sentence = item['sentence']
        ids = item['ids']
        types = item['types']

        if method == 'none':
            return '[SEP]'.join([item[column] for column in self.using_columns])
        else:
            slide_size = 0
            # i = 0 -> subject entity
            # i = 1 -> object entity
            so = ['S', 'O']
            for i, entity in enumerate([item['subject_entity'], item['object_entity']]):
                special_token_pair = f'[{so[i]}:{types[i]}]', f'[/{so[i]}:{types[i]}]'
                attached = special_token_pair[0] + entity + special_token_pair[1]
                sentence = sentence[:ids[i]+slide_size] + attached + sentence[ids[i]+len(entity)+slide_size:]

                if ids[0] < ids[1]:
                    slide_size += len(f'[{so[i]}:{types[i]}]' + f'[/{so[i]}:{types[i]}]')
        return sentence
=======
>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py

    def tokenizing(self, df: pd.DataFrame) -> List[dict]:
        data = []
    
        for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
<<<<<<< HEAD:r-bert/dataloader.py
            concat_entity = self.add_entity_token(item, method='tem')
            # print(self.tokenizer.tokenize(concat_entity, add_special_tokens=True))
            # print(concat_entity)
            # sys.exit()
=======
            
            # self.using_columns['subject_entity'] = '@*' + self.ner_type[item['subject_type']] + '*' + item['subject_entity'] + '@'
            # self.using_columns['object_entity'] = '#^' + self.ner_type[item['object_type']] + '^' + item['object_entity'] + '#'
            # concat_entity = '[SEP]'.join([val for column,val in self.using_columns.items()])
            
            subject_entity = '@*' + self.ner_type[item['subject_type']] + '*' + item['subject_entity'] + '@'
            object_entity = '#^' + self.ner_type[item['object_type']] + '^' + item['object_entity'] + '#'
            # concat_entity = subject_entity+ '[SEP]' + object_entity
            concat_entity = f'이 문장에서 {subject_entity}와 {object_entity}의 관계'
>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py

            outputs = self.tokenizer(
                concat_entity,
                item['sentence'],
                add_special_tokens=True, 
                padding='max_length',
                truncation=True,
                max_length=256
            )
            outputs['entity_mask_S'] = []
            entity_mask_S = 0
            
            for i in range(len(outputs['input_ids'])):
                idx = outputs['input_ids'][i]
                if idx in list(range(32000,32011,2)):
                    # print(list(range(32000,32011,2)))
                    entity_mask_S = 1
                    outputs['entity_mask_S'].append(entity_mask_S)
                elif idx in list(range(32001,32012,2)):
                    outputs['entity_mask_S'].append(entity_mask_S)
                    entity_mask_S = 0
                else:
                    outputs['entity_mask_S'].append(entity_mask_S)  

            outputs['entity_mask_O'] = []
            entity_mask_O = 0
        

            for i in range(len(outputs['input_ids'])):
                idx = outputs['input_ids'][i]
                if idx in list(range(32012,32023,2)):
                    # print(list(range(32000,32011,2)))
                    entity_mask_O = 1
                    outputs['entity_mask_O'].append(entity_mask_O)
                elif idx in list(range(32013,32024,2)):
                    outputs['entity_mask_O'].append(entity_mask_O)
                    entity_mask_O = 0
                else:
                    outputs['entity_mask_O'].append(entity_mask_O) 

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
<<<<<<< HEAD:r-bert/dataloader.py
        
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
=======
        subject_entity = []
        object_entity = []
        sentence = []
        subject_type = []
        object_type = []  

>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py
        
        for sub,obj,sent in zip(df['subject_entity'], df['object_entity'], df['sentence']):
            sub_word = sub[1:-1].split("':")[1].replace("'", '').split(',')[0].strip()
            obj_word = obj[1:-1].split("':")[1].replace("'", '').split(',')[0].strip()

            ss = int(sub.split("':")[2].split(',')[0].strip())   # subject_start
            se = int(sub.split("':")[3].split(',')[0].strip())   # sub_end
            os = int(obj.split("':")[2].split(',')[0].strip())   # obj_start
            oe = int(obj.split("':")[3].split(',')[0].strip())   # obj_end
            sub_type = sub[1:-1].split("':")[4].replace("'", '').strip() # sub_ner
            obj_type = obj[1:-1].split("':")[4].replace("'", '').strip() #obj_ner

            prepro_sent = self.add_entity_marker_punct(sent, sub_word, obj_word, sub_type, obj_type, ss, se, os, oe)
            
            subject_entity.append(sub_word)
            object_entity.append(obj_word)
            sentence.append(prepro_sent)
            subject_type.append(sub_type)
            object_type.append(obj_type)

        try:
            # print('training')
            preprocessed_df = pd.DataFrame({
                'id': df['id'], 
                'sentence': sentence,
                'subject_entity': subject_entity,
                'object_entity': object_entity,
                'subject_type' : subject_type,
                'object_type' : object_type,

                'label': df['label'],
            })
            
            inputs = self.tokenizing(preprocessed_df)
            targets = self.label_to_num(preprocessed_df['label'])

        except:
            # print('testing')
            preprocessed_df = pd.DataFrame({
<<<<<<< HEAD:r-bert/dataloader.py
            'id': df['id'], 
            'sentence': df['sentence'],
            'subject_entity': subject_entities,
            'object_entity': object_entities,
            'ids': ids,
            'types': types,
        })
=======
                'id': df['id'], 
                'sentence': sentence,
                'subject_entity': subject_entity,
                'object_entity': object_entity,
                'subject_type' : subject_type,
                'object_type' : object_type,

            })
>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py
            inputs = self.tokenizing(preprocessed_df)
            targets = []
    
        return inputs, targets

    def preprocessing_test(self, df: pd.DataFrame):
        
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
        
        
        # print('hey')
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

            test_inputs, test_targets = self.preprocessing_test(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data.drop(columns=['label'], inplace=True)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
<<<<<<< HEAD:r-bert/dataloader.py
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
=======
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=8)

>>>>>>> a76cebdf05df928f633155c32e05fb5c909ba707:template/dataloader.py

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)