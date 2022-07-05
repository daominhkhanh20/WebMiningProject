import pickle
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
 

class KNNClassifier():
    def __init__(self, 
        model_path: str = "assets/model",
        path_save_model: str = "assets/model", 
        max_df: float = 0.8, 
        min_n: int = 1, 
        max_n: int = 1, n_neighbors=3, mode: str = 'uniform'):

        self.mode = mode
        self.model_path = model_path
        self.path_save_model = path_save_model
        self.max_df = max_df
        self.min_n = min_n
        self.max_n = max_n 
        self.n_neighbors = n_neighbors
        self.KNN = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.mode)
        self.vectorizer = CountVectorizer(ngram_range=(self.min_n, self.max_n),
                                             max_df=self.max_df,
                                             max_features=None) 

    def train(self, X_train, y_train):
        self.KNN.fit(X_train, y_train)

    def evaluate(self,X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        y_pred = self.KNN.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        F1_score = f1_score(y_test, y_pred, average='weighted')
        print("Accuracy score vs {} neighbors: {}".format(self.n_neighbors, accuracy))
        print("Precision score vs {} neighbors: {}".format(self.n_neighbors, precision))
        print("Recall score vs {} neighbors: {}".format(self.n_neighbors, recall))
        print("F1 score vs {} neighbors: {}".format(self.n_neighbors, F1_score))
        print(classification_report(y_test, y_pred, target_names=['negative', 'neural', 'positive']))
        return (accuracy, precision, recall, F1_score, y_test, y_pred)

    def predict(self, input):
        input = self.vectorizer.transform(input)
        y_pred = self.KNN.predict(input)
        y_pred
