# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import unicodedata
import argparse
import collections
import json
import logging
import math
import os
import sys
import random

from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from transformers import BertForQuestionAnswering,BertConfig,AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

from torch.optim import Adam
max_len = 512

class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        tokenized_context = tokenizer(context, return_offsets_mapping=True)
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offset_mapping):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)
        if len(ans_token_idx) == 0:
            self.skip = True
            return
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]
        tokenized_question = tokenizer(question,return_offsets_mapping=True)
        input_ids = tokenized_context.input_ids + tokenized_question.input_ids[1:]
        token_type_ids = [0] * len(tokenized_context.input_ids) + [1] * len(tokenized_question.input_ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_len - len(input_ids)
        if padding_length > 0:   
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  
            self.skip = True
            return
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offset_mapping

def create_squad_examples(raw_data_path):
    with open(raw_data_path,'r',encoding = "utf-8") as f:
        raw_data = json.load(f)


    features = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                all_answers = [_["text"] for _ in qa["answers"]]
                start_char_idx = qa["answers"][0]["answer_start"]
                squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
                squad_eg.preprocess()
                if squad_eg.skip is True:
                    continue

                dataset_dict = {
                       "input_ids": None,
                       "token_type_ids": None,
                       "attention_mask": None,
                       "start_token_idx": None,
                       "end_token_idx": None,
                }

                dataset_dict["input_ids"] = squad_eg.input_ids
                dataset_dict["token_type_ids"] = squad_eg.token_type_ids
                dataset_dict["attention_mask"] = squad_eg.attention_mask
                dataset_dict["start_token_idx"] = squad_eg.start_token_idx
                dataset_dict["end_token_idx"] = squad_eg.end_token_idx
                features.append(dataset_dict)

    return features

def prepare_data(train_features):

    # 其实用到的训练特征为下面五个，前面的数据处理都是为了转化为下面的特征
    all_input_ids = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long)
    all_type_ids = torch.tensor([f["token_type_ids"] for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in train_features], dtype=torch.long)
    all_start_token_idx = torch.tensor([f["start_token_idx"] for f in train_features], dtype=torch.long)
    all_end_token_idx = torch.tensor([f["end_token_idx"] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_type_ids, all_attention_mask, all_start_token_idx,all_end_token_idx)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=6)
    return train_dataloader


def main():

    train_features = create_squad_examples("./data/cmrc2018_trial.json")
    train_dataloader = prepare_data(train_features)

    model = BertForQuestionAnswering(BertConfig()).from_pretrained("bert-base-chinese").cuda()
    model.train()
    
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        }]
    optimizer = Adam(optimizer_grouped_parameters,lr=3e-5)

    for epoch in range(20):
        for step, batch in enumerate(train_dataloader):
                input_ids, attention_mask, token_type_ids,start_positions,end_positions = batch
                inputs = {"input_ids":batch[0].cuda(),
                          "attention_mask":batch[1].cuda(),
                          "token_type_ids":batch[2].cuda(),
                          "start_positions":batch[3].cuda(),
                          "end_positions":batch[4].cuda()
                          }
                loss = model(**inputs)[0]
                print(epoch,step,loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    main()
