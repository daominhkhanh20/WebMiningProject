import logging
import torch
import os
import pandas as pd

from collections import Counter
from torch import Tensor
from src.dataset import CNNDataset, CNNTokenizer

logger = logging.getLogger(__name__)


class CNNDataSource(object):
    def __init__(self,
                 train_dataset: CNNDataset = None,
                 val_dataset: CNNDataset = None,
                 test_dataset: CNNDataset = None,
                 tokenizer: CNNTokenizer = None,
                 map_label: dict = None,
                 weight_contribution: Tensor = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.map_label = map_label
        self.weight_contribution = weight_contribution

    @classmethod
    def init_datasource(cls,
                        path_folder_data: str = 'assets/data',
                        max_length: int = 256,
                        text_col: str = 'comment',
                        label_col: str = 'pred_label'):
        train_dataset, test_dataset, val_dataset = None, None, None
        map_labels = None
        train_file = ""
        for file in os.listdir(path_folder_data):
            if "train" in file:
                train_file = file
                break
        if train_file == "":
            raise Exception(f"Train file not in data folder.\n"
                            f"Need train file to create tokenizer.")

        train_data = pd.read_csv(f"{path_folder_data}/{train_file}")
        data = list(train_data[text_col])
        tokenizer = CNNTokenizer(sequence_length=max_length)
        tokenizer.build_vocab(data)

        for file in os.listdir(path_folder_data):
            logger.info(f"Start convert file {file} to dataset.")
            dataset = CNNDataset(
                data=f"{path_folder_data}/{file}",
                tokenizer=tokenizer,
                text_col=text_col,
                label_col=label_col,
                max_length=max_length,
                map_label=map_labels
            )

            if map_labels is None:
                map_labels = dataset.map_label

            logger.info(dataset.map_label)

            if 'train' in file:
                train_dataset = dataset
            elif 'test' in file:
                test_dataset = dataset
            elif 'val' in file:
                val_dataset = dataset
            else:
                raise Exception(f"File not valid, only accept file for train, test and valid.")
        if train_dataset is not None:
            labels_counter = dict(Counter(train_dataset.data[label_col]))
            labels_counter = {label: labels_counter[train_dataset.map_label[label]] for label in \
                              train_dataset.list_labels}
            norm_count_labels = [len(train_dataset.data) /value for label, value in labels_counter.items()]
            weight_contribution = torch.tensor(
                [value / sum(norm_count_labels) for value in norm_count_labels]
            )
        else:
            weight_contribution = None

        return cls(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            map_label=train_dataset.map_label,
            weight_contribution=weight_contribution
        )