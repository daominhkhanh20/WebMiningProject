import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
 

class KNNClassifier():
    def __init__(self, 
        model_path: str = "assets/model",
        path_save_model: str = "assets/model", 
        max_df: float = 0.8, 
        min_n: int = 1, 
        max_n: int = 1, n_neighbors=3):

        self.model_path = model_path
        self.path_save_model = path_save_model
        self.max_df = max_df
        self.min_n = min_n
        self.max_n = max_n 
        self.n_neighbors = n_neighbors
        self.KNN = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.vectorizer = CountVectorizer(ngram_range=(self.min_n, self.max_n),
                                             max_df=self.max_df,
                                             max_features=None) 

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)
        self.KNN.fit(X_train, y_train)

    def evaluate(self,X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        X_test = self.vectorizer.fit_transform(X_test)
        y_pred = self.KNN.predict(X_test)
        print("Accuracy score vs {} neighbors: {}".format(self.n_neighbors, accuracy_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred))

    def predict(self, input):
        input = self.vectorizer.fit_transform(input)
        y_pred = self.KNN.predict(input)
        y_pred
