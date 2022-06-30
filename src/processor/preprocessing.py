import re

import gensim.utils
from tqdm._tqdm_notebook import tqdm_notebook
import pandas as pd
from pandas import DataFrame
from typing import List, Union
import unicodedata
import os
from src.utils.create_data import clean_text


tqdm_notebook().pandas()


class CleanText:
    def __init__(self, path_folder_data: str = 'CommentData',
                 val_split_percentage: float = 0.1,
                 test_split_percentage: float = 0.1,
                 prefix_file: str = 'final',
                 path_save: str = 'assets/data'):
        self.path_folder_data = path_folder_data
        self.val_split_percentage: float = val_split_percentage
        self.test_split_percentage: float = test_split_percentage
        self.data: Union[List[DataFrame], DataFrame] = []
        self.prefix_file = prefix_file
        self.path_save = path_save
        if os.path.exists(self.path_save) is False:
            os.makedirs(self.path_save, exist_ok=True)

    def split(self, df: DataFrame, label_col: str = 'pred_label', n_percentage_split: float = 0.1):
        # print(f"Before: {df.shape}")
        # df_split = df.groupby(label_col).sample(frac=n_percentage_split, random_state=42)
        # df = df.drop(df_split.index)
        # print(len(df_split))
        # print(len(df))
        # print('\n\n')
        # return df.reset_index(drop=True), df_split.reset_index(drop=True)
        df_split = df.sample(frac=n_percentage_split)
        df = df.drop(df_split.index)
        return df.reset_index(drop=True), df_split.reset_index(drop=True)

    def save(self, df: DataFrame, file_name: str):
        df.to_csv(file_name, index=False)

    def fit(self):
        for file in os.listdir(self.path_folder_data):
            if not file.startswith(self.prefix_file):
                df = pd.read_csv(f"{self.path_folder_data}/{file}")
                df['by'] = file[: file.find('.')]
                self.data.append(df)
        self.data = pd.concat(self.data, axis=0)
        self.data.drop_duplicates(subset=['comment'], keep='first', inplace=True)
        self.data = self.data[self.data.notna()].reset_index(drop=True)
        self.data = self.data[['comment', 'rating_star', 'by', 'pred_label']]
        self.data.reset_index(drop=True)
        self.data['comment'] = self.data['comment'].progress_apply(lambda x: clean_text(x))
        df_train, df_test = self.split(self.data, n_percentage_split=self.test_split_percentage)
        df_train, df_val = self.split(df_train, n_percentage_split=self.val_split_percentage)
        self.save(df_train, f"{self.path_save}/{self.prefix_file}_train.csv")
        self.save(df_test, f"{self.path_save}/{self.prefix_file}_test.csv")
        self.save(df_val, f"{self.path_save}/{self.prefix_file}_val.csv")
        print(df_train.shape)
        print(df_test.shape)
        print(df_val.shape)


clean = CleanText()
clean.fit()
