import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from imblearn.over_sampling import RandomOverSampler

class AnnCommentDataset(Dataset):
    def __init__(self, data: Union[DataFrame, str],
                 text_col: str = 'comment',
                 label_col: str = 'pred_label',
                 map_label: dict = None,
                 is_train: bool = None,
                 stopword_path: str=None,
                 path_save_tf : str=None,
                 **kwargs):
        if isinstance(data, str):
            data = pd.read_csv(data)
            data = data.fillna('')

        self.data = data
        if is_train:
            stopwords = []
            with open(stopword_path,'r',encoding="utf8") as f:
                for word in f:
                    stopwords.append(word.replace('\n',''))
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            self.data_tf = vectorizer.fit_transform(self.data[text_col]).toarray()

            ros = RandomOverSampler(random_state=42,sampling_strategy={'neural':2000})
            self.data_tf, self.y_res = ros.fit_resample(self.data_tf, self.data[label_col])
            with open(f'{path_save_tf}/vectorizer.pk', 'wb') as fin:
                pickle.dump(vectorizer, fin)
        else:
            with open(f'{path_save_tf}/vectorizer.pk', 'rb') as fin:
                vectorizer = pickle.load(fin)
                self.data_tf = vectorizer.transform(self.data[text_col]).toarray()



        self.input_size = self.data_tf.shape[1]
        if map_label is None:
            self.list_labels = list(set(self.data[label_col]))
            self.map_label = {label: idx for idx, label in enumerate(self.list_labels)}
        else:
            self.map_label = map_label
            self.list_labels = list(self.map_label.keys())
        if is_train:
            self.labels = self.y_res.map({'negative':0,'neural':1,'positive':2}).to_list()
        else:
            self.data[label_col] = self.data[label_col].map(self.map_label)
            self.labels = self.data[label_col].values.tolist()

    def __len__(self):
        return len(self.data_tf)

    def __getitem__(self, item):
        return {
            'input': torch.tensor(self.data_tf[item], dtype=torch.float),
            'label': self.labels[item]
        }