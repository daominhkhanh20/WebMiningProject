import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

 

class NaiveBayesClassifier():
    def __init__(self, 
        model_path: str = "assets/model",
        path_save_model: str = "assets/model", 
        max_df: float = 0.8, 
        min_n: int = 1, 
        max_n: int = 1):

        self.model_path = model_path
        self.path_save_model = path_save_model
        self.max_df = max_df
        self.min_n = min_n
        self.max_n = max_n 
        self.pipeline = Pipeline([ 
                                # ('tfidf', TfidfTransformer()), 
                                ('clf', MultinomialNB()) ])

    def train(self, X_train, y_train): 
        model = self.pipeline.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join(self.path_save_model, "naive_bayes.pkl"), 'wb'))

    def evaluate(self, X_test, y_test):
        model = pickle.load(open(os.path.join(self.model_path, "naive_bayes.pkl"), 'rb'))
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        F1_score = f1_score(y_test, y_pred, average='weighted')
        print(classification_report(y_test, y_pred, target_names=['negative', 'neural', 'positive']))
        return (accuracy, precision, recall, F1_score)

    def predict(self, input):
        model = pickle.load(open(os.path.join(self.model_path, "naive_bayes.pkl"), 'rb'))
        y_pred = model.predict(input)
        return y_pred
