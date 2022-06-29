import argparse
from importlib.resources import path

from sklearn import naive_bayes

from src.learner.KNN import KNNClassifier
from src.dataset.ml_dataset import MlDataset
import matplotlib.pyplot as plt
import numpy as np

from src.learner.DecisionTree import DecisionTreeClassifier 

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data')
parser.add_argument('--path_model', type=str, default='assets/models')

args = parser.parse_args()

dataset = MlDataset(
    path_trainset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/train.csv',
    path_testset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/test.csv',
    path_valset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/dev.csv'
)
decision_tree = DecisionTreeClassifier(model_path='/home/thangnd/Documents/Project/WebMiningProject/assets/models',
 path_save_model='/home/thangnd/Documents/Project/WebMiningProject/assets/models')

train_set, test_set, val_set = dataset.get_data()

decision_tree.train(train_set['input'], train_set['label'])

# acc, pre, recal, f1 = decision_tree.evaluate(test_set['input'], test_set['label'])
# print("Accuracy: ", acc)
# print("Precision: ", pre)
# print("Recall: ", recal)
# print("F1 score: ", f1)