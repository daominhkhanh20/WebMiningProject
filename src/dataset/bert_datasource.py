import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
import os
from collections import Counter

from torch import Tensor

from src.dataset import CommentDataset

logger = logging.getLogger(__name__)


class BertDataSource(object):
    def __init__(self,
                 train_dataset: CommentDataset = None,
                 val_dataset: CommentDataset = None,
                 test_dataset: CommentDataset = None,
                 map_label: dict = None,
                 weight_contribution: Tensor = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.map_label = map_label
        self.weight_contribution = weight_contribution

    @classmethod
    def init_datasource(cls,
                        path_folder_data: str = 'assets/data',
                        pretrained_model_name: str = 'FPTAI/vibert-base-cased',
                        max_length: int = 256,
                        text_col: str = 'comment',
                        label_col: str = 'pred_label',
                        mode_increase_weight_neural: bool = False,
                        neural_weight: int = 10,
                        use_weight_contribution: bool = False):
        train_dataset, test_dataset, val_dataset = None, None, None
        map_labels = None
        for file in os.listdir(path_folder_data):
            logger.info(f"Start convert file {file} to dataset")
            dataset = CommentDataset(
                data=f"{path_folder_data}/{file}",
                tokenizer_name=pretrained_model_name,
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
            elif 'val' in file or 'dev' in file:
                val_dataset = dataset
            else:
                raise Exception(f"Currently, convert file for {file} isn't support"
                                f"Only support for train, test, val")
        weight_contribution = None
        if use_weight_contribution:
            logger.info("Turn on mode use weight contribution")
            if train_dataset is not None and mode_increase_weight_neural is False:
                labels_counter = dict(Counter(train_dataset.data[label_col]))
                labels_counter = {label: labels_counter[train_dataset.map_label[label]] for label in train_dataset.list_labels}
                norm_count_labels = [len(train_dataset.data) / value for label, value in labels_counter.items()]
                weight_contribution = torch.tensor(
                    [value/sum(norm_count_labels) for value in norm_count_labels]
                )
            elif mode_increase_weight_neural:
                weight_contribution = []
                for label in train_dataset.map_label.values():
                    if 'neural' in label:
                        weight_contribution.append(neural_weight)
                    else:
                        weight_contribution.append(1)

                weight_contribution = torch.tensor(weight_contribution, dtype=torch.long)
        else:
            logger.info("Turn off mode use weight contribution")

        return cls(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            map_label=train_dataset.map_label,
            weight_contribution=weight_contribution
        )
