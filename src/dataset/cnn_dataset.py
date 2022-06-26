import torch
import pandas as pd
import logging

from torch.utils.data import Dataset
from typing import Union, Dict
from tqdm import tqdm
from src.utils.create_data import make_input_cnn

logger = logging.getLogger(__name__)


class CNNTokenizer:
    def __init__(self, padding_idx: int = 0,
                 unk_idx: int = 0,
                 max_len_vocab: int = 10000,
                 min_appearance: int = 5,
                 sequence_length: int = 256,
                 vocab_dict: Dict = None):
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.max_len_vocab = max_len_vocab
        self.min_appearance = min_appearance
        self.sequence_length = sequence_length
        if vocab_dict is None:
            self.vocab_dict = {}
            self.vocab_list = []
            self.vocab_map = {}
        else:
            self.vocab_dict = vocab_dict
            vocab_list = list(set(self.vocab_dict.keys()))
            if self.unk_idx < self.padding_idx:
                vocab_list.insert(self.unk_idx, "<unk>")
                vocab_list.insert(self.padding_idx, "<pad>")
            else:
                vocab_list.insert(self.padding_idx, "<pad>")
                vocab_list.insert(self.unk_idx, "<unk>")
            self.vocab_list = vocab_list
            self.vocab_map = {token: vocab_list.index(token) for token in vocab_list}

    def build_vocab(self, dataset):
        sent_token = [str(sent).strip().split(" ") for sent in dataset]
        vocab_dict = {}
        for sent in sent_token:
            for token in sent:
                if token not in vocab_dict.keys():
                    vocab_dict[token] = 1
                else:
                    vocab_dict[token] += 1

        vocab_dict = {k: v for (k, v) in vocab_dict.items() if v > self.min_appearance}
        vocab_dict = {k: v for (k, v) in sorted(vocab_dict.items(), key=lambda x: -x[1])}
        if len(vocab_dict) > self.max_len_vocab:
            vocab_dict = {k: v for (k, v) in list(vocab_dict.items())[:self.max_len_vocab]}
        self.vocab_dict = vocab_dict
        vocab_list = list(set(vocab_dict.keys()))
        if self.unk_idx < self.padding_idx:
            vocab_list.insert(self.unk_idx, "<unk>")
            vocab_list.insert(self.padding_idx, "<pad>")
        else:
            vocab_list.insert(self.padding_idx, "<pad>")
            vocab_list.insert(self.unk_idx, "<unk>")
        self.vocab_list = vocab_list
        self.vocab_map = {token: vocab_list.index(token) for token in vocab_list}

    def tokenize(self, sent):
        words_list = sent.split(" ")
        idx = []
        for item in words_list:
            if item in self.vocab_map.keys():
                idx.append(self.vocab_map[item])
            else:
                idx.append(self.vocab_map["<unk>"])
        if len(idx) < self.sequence_length:
            for item in range(len(idx), self.sequence_length):
                idx.append(self.vocab_map["<pad>"])
        return torch.tensor(idx, dtype=torch.long)


class CNNCollate:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch, **kwargs):
        token_ids = [each_batch['input_id'].detach().numpy() for each_batch in batch]
        token_ids = torch.tensor(token_ids, dtype=torch.long).squeeze(0)
        labels = [each_batch['label'] for each_batch in batch]
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": token_ids,
            "labels": labels
        }


class CNNDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, str],
                 tokenizer: CNNTokenizer,
                 text_col: str = 'comment',
                 label_col: str = 'pred_label',
                 max_length: int = 256,
                 map_label: dict = None,
                 **kwargs):
        super().__init__()

        if isinstance(data, str):
            data = pd.read_csv(data)

        self.data = data
        self.data = self.data[self.data[text_col].notna()].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label = label_col
        self.token_ids, self.attention_mask = [], []
        self.pad_token_id = self.tokenizer.padding_idx
        if map_label is None:
            self.list_labels = list(set(self.data[label_col]))
            self.map_label = {label: idx for idx, label in enumerate(self.list_labels)}
        else:
            self.map_label = map_label
            self.list_labels = list(self.map_label.keys())

        self.data[label_col] = self.data[label_col].map(self.map_label)
        self.labels = self.data[label_col].values.tolist()
        for idx, row in tqdm(self.data.iterrows(), total=len(data)):
            text = row[text_col]
            id_token = make_input_cnn(self.tokenizer, text)
            self.token_ids.append(id_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            'input_id': torch.tensor(self.token_ids[item], dtype=torch.long),
            'label': self.labels[item]
        }

