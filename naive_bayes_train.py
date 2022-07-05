import argparse
from importlib.resources import path

from sklearn import naive_bayes

from src.learner.KNN import KNNClassifier
from src.dataset.ml_dataset import MlDataset
import matplotlib.pyplot as plt
import numpy as np

from src.learner.NaiveBayes import NaiveBayesClassifier 
from src.utils.plot import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data')
parser.add_argument('--path_model', type=str, default='assets/models')

args = parser.parse_args()

dataset = MlDataset(
    path_trainset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/train.csv',
    path_testset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/test.csv',
    path_valset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/dev.csv'
)
naive_bayes = NaiveBayesClassifier(model_path='/home/thangnd/Documents/Project/WebMiningProject/assets/models',
 path_save_model='/home/thangnd/Documents/Project/WebMiningProject/assets/models', mode='gaussian')

train_set, test_set, val_set = dataset.get_data()

# naive_bayes.train(train_set['input'], train_set['label'])
acc, pre, recal, f1, y_true, y_pred = naive_bayes.evaluate(test_set['input'], test_set['label'])
print("Accuracy: ", acc)
print("Precision: ", pre)
print("Recall: ", recal)
print("F1 score: ", f1)

# confusion matrix plot
plt.figure()
class_names = ['negative', 'neutral', 'positive']
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()