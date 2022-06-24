import re
from tqdm._tqdm_notebook import tqdm_notebook
import pandas as pd
from pandas import DataFrame
from typing import List, Union
import unicodedata
import os

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

    def clean_text(self, sent):
        sent = unicodedata.normalize('NFC', sent)
        emoji = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", re.UNICODE)
        return re.sub(emoji, '', sent)

    def split(self, df: DataFrame, label_col: str = 'pred_label', n_percentage_split: float = 0.1):
        # print(f"Before: {df.shape}")
        # df_split = df.groupby(label_col).sample(frac=n_percentage_split, random_state=42)
        # df = df.drop(df_split.index)
        # print(len(df_split))
        # print(len(df))
        # print('\n\n')
        # return df.reset_index(drop=True), df_split.reset_index(drop=True)
        df_split = df.sample(frac=n_percentage_split)
        df.drop(df_split.index)
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
        self.data = self.data[['comment', 'rating_star', 'by', 'pred_label']]
        self.data.drop_duplicates(subset=['comment'], keep='first', inplace=True)
        self.data.reset_index(drop=True)
        self.data['comment'] = self.data['comment'].progress_apply(lambda x: self.clean_text(x))
        df_train, df_test = self.split(self.data, n_percentage_split=self.test_split_percentage)
        df_train, df_val = self.split(df_train, n_percentage_split=self.val_split_percentage)
        self.save(df_train, f"{self.path_save}/{self.prefix_file}_train.csv")
        self.save(df_test, f"{self.path_save}/{self.prefix_file}_test.csv")
        self.save(df_val, f"{self.path_save}/{self.prefix_file}_val.csv")
        # print(df_train.shape)
        # print(df_test.shape)
        # print(df_val.shape)


clean = CleanText()
clean.fit()
