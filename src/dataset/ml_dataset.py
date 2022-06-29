import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class MlDataset():
    def __init__(self, path_trainset, path_testset, path_valset, mode_encoding: str = "bow"):
        self.path_trainset = path_trainset
        self.path_testset = path_testset
        self.path_valset = path_valset
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.vectorizer = CountVectorizer(ngram_range=(1, 1),
                                             max_df=0.8,
                                             max_features=None)
        self.label_encoder = LabelEncoder()
        # self.vectorizer_final = None
        self.df_test = None
        self.df_train = None
        self.df_val = None

    def load_all_data(self):
        self.df_train = pd.read_csv(self.path_trainset)
        self.df_test = pd.read_csv(self.path_testset)
        self.df_val = pd.read_csv(self.path_valset)
        data = pd.concat([self.df_train, self.df_test, self.df_val], axis=0)
        input = data['text'].to_list()
        self.vectorizer.fit(input)
        label = data['label'].to_list()
        self.label_encoder.fit(label)
        print(list(self.label_encoder.classes_))

    def load_trainset(self):
        input = self.df_train['text'].to_list()
        # self.vectorizer_final = self.vectorizer.fit(input)
        input = self.vectorizer.transform(input).toarray()

        label = self.df_train['label'].to_list()
        label = self.label_encoder.transform(label)
        self.train_set = {'input': input, 'label': label}
        
    def load_testset(self):
        input = self.df_test['text'].to_list()
        input = self.vectorizer.transform(input).toarray()

        label = self.df_test['label'].to_list()
        label = self.label_encoder.transform(label)
        self.test_set = {'input': input, 'label': label}

    def load_valset(self):
        input = self.df_val['text'].to_list()
        input = self.vectorizer.transform(input).toarray()

        label = self.df_val['label'].to_list()
        label = self.label_encoder.transform(label)
        self.val_set = {'input': input, 'label': label}


    def get_data(self):
        self.load_all_data()
        self.load_trainset()
        self.load_testset()
        self.load_valset()
        return (self.train_set, self.test_set, self.val_set)