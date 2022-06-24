import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Union
from collections import Counter
from src.utils.create_data import *
import logging

logger = logging.getLogger(__name__)


class CommentCollate:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batchs, **kwargs):
        token_ids = [batch['token_ids'] for batch in batchs]
        attention_masks = [batch['attention_mask'] for batch in batchs]
        labels = [batch['label'] for batch in batchs]
        labels = torch.tensor(labels, dtype=torch.long)
        token_ids = pad_sequence(token_ids, padding_value=self.pad_id, batch_first=True)
        attention_masks = pad_sequence(attention_masks, padding_value=0, batch_first=True)
        return {
            "token_ids": token_ids,
            "attention_masks": attention_masks,
            "labels": labels
        }


class CommentDataset(Dataset):
    def __init__(self, data: Union[DataFrame, str],
                 tokenizer_name: str,
                 text_col: str = 'comment',
                 label_col: str = 'pred_label',
                 max_length: int = 256,
                 **kwargs):
        if isinstance(data, str):
            data = pd.read_csv(data)

        self.data = data
        self.data = self.data[self.data[text_col].notna()].reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_col = text_col
        self.label_col = label_col
        self.token_ids, self.attention_mask = [], []
        self.pad_token_id = self.tokenizer.pad_token_id
        self.list_labels = list(set(self.data[label_col]))
        self.map_label = {label: idx for idx, label in enumerate(self.list_labels)}
        self.data[label_col] = self.data[label_col].map(self.map_label)
        self.labels = self.data[label_col].values.tolist()
        for idx, row in tqdm(self.data.iterrows(), total=len(data)):
            text = row[text_col]
            id, mask = make_input_bert(self.tokenizer, sent=text, max_length=max_length)
            self.token_ids.append(id)
            self.attention_mask.append(mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            'token_ids': torch.tensor(self.token_ids[item], dtype=torch.long),
            'mask': torch.tensor(self.attention_mask[item], dtype=torch.long),
            'label': self.labels[item]
        }
