import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
import os
from collections import Counter

from torch import Tensor

from src.dataset import AnnCommentDataset

logger = logging.getLogger(__name__)


class AnnDataSource(object):
    def __init__(self,
                 train_dataset: AnnCommentDataset = None,
                 val_dataset: AnnCommentDataset = None,
                 test_dataset: AnnCommentDataset = None,
                 map_label: dict = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.map_label = map_label

    @classmethod
    def init_datasource(cls,
                        path_folder_data: str = 'assets/data',
                        stopword_path: str = 'assets/stopword/stopword.txt',
                        path_save_tf: str ='assets/utils_weight',
                        text_col: str = 'sentence',
                        label_col: str = 'lb_name',
                        map_labels:dict = {'negative':0,'neutral':1,'positive':2}
                        ):
        train_dataset, test_dataset, val_dataset = None, None, None
        for file in os.listdir(path_folder_data):
            logger.info(f"Start convert file {file} to dataset")
            if 'train' in file:
                train_dataset = AnnCommentDataset(
                    data=f"{path_folder_data}/{file}",
                    text_col=text_col,
                    label_col=label_col,
                    map_label=map_labels,
                    stopword_path= stopword_path,
                    path_save_tf= path_save_tf,
                    is_train=True)
                logger.info(train_dataset.map_label)
                if map_labels is None:
                    map_labels = train_dataset.map_label

        for file in os.listdir(path_folder_data):
            if 'train' not in file:
                dataset = AnnCommentDataset(
                    data=f"{path_folder_data}/{file}",
                    text_col=text_col,
                    label_col=label_col,
                    path_save_tf= path_save_tf,
                    map_label=map_labels
                )

                if 'test' in file:
                    test_dataset = dataset
                elif 'val' in file:
                    val_dataset = dataset
                else:
                    raise Exception(f"Currently, convert file for {file} isn't support"
                                    f"Only support for train, test, val")

        return cls(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            map_label=train_dataset.map_label)
