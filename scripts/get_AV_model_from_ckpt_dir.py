import os
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import Trainer, AutoModelForSequenceClassification


def get_text_encodings(model_name, texts, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return  tokenizer(texts, truncation=True, 
                      padding="max_length", 
                      max_length=max_length)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataset(model_name, texts,
                max_length, labels):
    
    encodings = get_text_encodings(model_name, texts, 
                                   max_length)

    dataset = CustomDataset(encodings, labels)
    return dataset


def get_model_and_trainer(ckpt_dir):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    trainer = Trainer(model=model)
    return trainer