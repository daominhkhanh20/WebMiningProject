import argparse

from src.learner.KNN import KNNClassifier
from src.dataset.ml_dataset import MlDataset
import matplotlib.pyplot as plt
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data')
parser.add_argument('--path_model', type=str, default='assets/models')

args = parser.parse_args()

dataset = MlDataset(
    path_trainset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/train.csv',
    path_testset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/test.csv',
    path_valset='/home/thangnd/Documents/Project/WebMiningProject/assets/_UIT-VSFC/csv/dev.csv'
)
knn = KNNClassifier()

train_set, test_set, val_set = dataset.get_data()
# print(train_set['input'].shape)
# print(train_set['label'].shape)
# print(test_set['input'].shape)
# print(test_set['label'].shape)
# print(val_set['input'].shape)
# print(val_set['label'].shape)

# knn.evaluate(train_set['input'], train_set['label'], test_set['input'], test_set['label'])

# data_train = np.append([train_set['input']], [val_set['input']], axis=0)
# label_train = np.append([train_set['label'], val_set['label']], axis=0)

k_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
accuracy = []
precision = []
recall = []
f1_score = []
for k in k_neighbors:
    knn = KNNClassifier(n_neighbors=k)
    print("Result classification of k = {} nearest neighborhoods".format(k))
    a,p,r,f = knn.evaluate(train_set['input'], train_set['label'], test_set['input'], test_set['label'])
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f1_score.append(f)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(k_neighbors, accuracy)
axs[0, 0].set_title("Accuracy score")

axs[0, 1].plot(k_neighbors, precision)
axs[0, 1].set_title("Precision score")

axs[1, 0].plot(k_neighbors, recall)
axs[1, 0].set_title("Recall score")

axs[1, 1].plot(k_neighbors, f1_score)
axs[1, 1].set_title("F1-score")

for ax in axs.flat:
    ax.set(xlabel='k_neighbors', ylabel='score')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()


# Choose k = 5