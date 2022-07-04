from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def plot_loss(history: defaultdict, path_save: str):
    range_x = [i for i in range(len(history['train_loss']))]
    plt.plot(range_x, history['train_loss'], label='train_loss')
    plt.plot(range_x, history['val_loss'], label='val_loss')
    plt.legend(loc="upper left")
    plt.savefig(f'{path_save}/loss.png')
    logger.info(f"Save loss.png at {path_save}")
    plt.close()


def plot_acc(history: defaultdict, path_save: str):
    range_x = [i for i in range(len(history['train_acc']))]
    plt.plot(range_x, history['train_acc'], label='train_acc')
    plt.plot(range_x, history['val_acc'], label='val_acc')
    plt.legend(loc="upper left")
    plt.savefig(f'{path_save}/acc.png')
    logger.info(f"Save acc.png at {path_save}")
    plt.close()


def plot_confusion_matrix(final_true: list, final_preds: list, labels: list, path_save: str):
    temp = confusion_matrix(final_true, final_preds, labels=labels)
    df = pd.DataFrame(temp, index=labels, columns=labels).astype(int)
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(df, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    result = accuracy_score(final_true, final_preds)
    plt.title("Accuracy for test:{:0.5f}".format(result))
    plt.savefig(f"{path_save}/confusion_matrix.png")
    logger.info(f"Save confusion_matrix.png at {path_save}")
    plt.close()
