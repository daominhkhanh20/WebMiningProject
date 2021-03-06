import torch.optim as optim
from tqdm import tqdm
import os
import torch
from torch import nn
import logging
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
import time
from tabulate import tabulate
from src.dataset import AnnDataSource
from src.model import ANNModel
from src.learner import BaseLeaner
from src.constants import *
import json
import pickle
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader
from src.utils.create_data import clean_text

logger = logging.getLogger(__name__)


class AnnLearner(BaseLeaner):
    def __init__(self, mode: str = 'training',
                 data_source: AnnDataSource = None,
                 batch_size: int = 64, n_epochs: int = 10,
                 learning_rate: float = 1e-6, path_save_model: str = 'assets/models',
                 path_report: str = 'assets/report',
                 is_save_best_model: bool = True,
                 infer_parm: dict=None,
                 over_sampling: bool=None,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        if mode.lower() not in [INFERENCE_MODE, TRAINING_MODE]:
            NotImplementedError(f"{mode} isn't support")
        if mode == INFERENCE_MODE:
            if path_save_model is not None:
                path_save_model = os.path.abspath(path_save_model)
            else:
                raise Exception(f"{path_save_model} isn't exist")
        self.over_sampling = over_sampling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if mode == INFERENCE_MODE:
            self.data_source = data_source
            self.map_label = infer_parm['map_label']

            self.model = ANNModel(
                input_size=infer_parm['input_size'],
                n_labels=infer_parm['n_labels'],
                drop_out=infer_parm['drop_out']
            ).to(self.device)
            if infer_parm['over_sampling']:
                weight_name = "ann_weight_over_sampling.pth"
            else:
                weight_name = "ann_weight.pth"
            self.model.load_state_dict(torch.load(f"{path_save_model}/{weight_name}",map_location='cpu'))
            print('Load model done...')

        elif mode == TRAINING_MODE:
            self.data_source = data_source
            self.map_label = self.data_source.train_dataset.map_label
            self.n_labels = len(self.map_label)
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.learning_rate = learning_rate
            self.path_report = path_report
            self.path_save_model = path_save_model
            self.is_best = 0

            self.is_save_best_model = is_save_best_model

            self.model = ANNModel(
                input_size=self.data_source.train_dataset.input_size,
                n_labels=len(self.data_source.train_dataset.map_label),
                drop_out=dropout
            ).to(self.device)
            if os.path.exists(path_save_model) is False:
                os.makedirs(self.path_save_model, exist_ok=True)
            if os.path.exists(path_report) is False:
                os.makedirs(self.path_report, exist_ok=True)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate)
            self.loss_fn = nn.CrossEntropyLoss()
            self.train_loader = self.make_loader(
                self.data_source.train_dataset)
            if self.data_source.val_dataset:
                self.val_loader = self.make_loader(
                    self.data_source.val_dataset)
            if self.data_source.test_dataset:
                self.test_loader = self.make_loader(
                    self.data_source.test_dataset)
            self.config_architecture = {}
            self.best_val_loss = sys.maxsize
            self.best_val_acc = 0
            self.info_train = {
                'train_acc': [],
                'train_loss': [],
                'val_acc': [],
                'val_loss': []

            }
            self.config_architecture['best_model'] = {
                "val_loss": self.best_val_loss,
                "val_acc": self.best_val_acc,
                "epoch": 0
            }
            self.config_architecture['model'] = {
                "model_name": 'ann',
                "input_size":self.data_source.train_dataset.input_size,
                "n_labels": len(self.data_source.train_dataset.map_label),
                "drop_out":dropout,
                'map_label':self.map_label,
                'over_sampling' :self.over_sampling
            }

    def make_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    def predict(self,input):
        self.model.eval()
        if input == None:
            input = ''
        input = clean_text(input)
        input = [input]
        print(input)
        path_save_tf = 'assets/utils_weight'
        with open(f'{path_save_tf}/vectorizer.pk', 'rb') as fin:
            vectorizer = pickle.load(fin)
            self.input = vectorizer.transform(input).toarray()
        self.input = torch.tensor(self.input,dtype=torch.float)
        # self.input = torch.unsqueeze(self.input, 0).to(self.device)
        with torch.no_grad():
            outputs = self.model(self.input)
            label_pred = torch.argmax(
                outputs, dim=-1).detach().cpu().numpy()
            res = ''
            for item in self.map_label:
                if self.map_label[item] == label_pred[0]:
                    res = item
            return res
            

    def train_one_epoch(self, **kwargs):
        self.model.train()
        train_loss = 0
        labels_truth, labels_pred = [], []
        for idx, sample in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            self.optimizer.zero_grad()
            input = sample['input'].to(self.device)
            label = sample['label'].to(self.device)

            outputs = self.model(input)
            loss = self.loss_fn(outputs, label)

            label_pred = torch.argmax(
                outputs, dim=-1).detach().cpu().numpy().tolist()
            label_truth = label.detach().cpu().numpy().tolist()
            labels_pred.extend(label_pred)
            labels_truth.extend(label_truth)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        accuracy = accuracy_score(labels_truth, labels_pred)
        return train_loss / len(self.train_loader), accuracy

    def save_json(self, data, file_name):
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)

    def save(self, epoch, **kwargs):
        self.config_architecture["best_model"]["val_loss"] = self.best_val_loss
        self.config_architecture["best_model"]["val_acc"] = self.best_val_acc
        self.config_architecture["best_model"]["epoch"] = epoch
        if self.over_sampling:
            self.save_json(self.config_architecture,
                       f"{self.path_save_model}/ann_config_architecture_over_sampling.json")
        else:
            self.save_json(self.config_architecture,
                        f"{self.path_save_model}/ann_config_architecture.json")
        if self.is_save_best_model:
            if self.over_sampling:
                weight_name = "ann_weight_over_sampling.pth"
            else:
                weight_name = 'ann_weight.pth'
        else:
            if self.over_sampling:
                weight_name = f"ann_weight_over_sampling{epoch}.pth"
            else:
                weight_name =f"ann_weight{epoch}.pth"

        torch.save(self.model.state_dict(),
                   f"{self.path_save_model}/{weight_name}")
        logger.info("Save model done")

    def evaluate(self, loader, **kwargs):
        self.model.eval()
        mode_testing = kwargs.get('mode_testing', False)
        labels_truth, labels_pred = [], []
        val_loss = 0
        for idx, sample in tqdm(enumerate(loader), total=len(loader)):
            input = sample['input'].to(self.device)
            label = sample['label'].to(self.device)

            outs = self.model(input)
            loss = self.loss_fn(outs, label)

            label_pred = torch.argmax(
                outs, dim=-1).detach().cpu().numpy().tolist()
            label_truth = label.detach().cpu().numpy().tolist()
            labels_pred.extend(label_pred)
            labels_truth.extend(label_truth)
            val_loss += loss.item()

        accuracy = accuracy_score(labels_truth, labels_pred)
        if mode_testing:
            report = {
                "f1_score": [f1_score(labels_truth, labels_pred, average='weighted')],
                "recall_score": [recall_score(labels_truth, labels_pred, average='weighted')],
                "accuracy_score": [accuracy]
            }
            logger.info("Detail result for testing")
            if self.is_best:
                print('best',self.k)
                self.config_architecture['best'] = {'epoch':self.k}

                self.config_architecture['best']['labels_truth'] = labels_truth
                self.config_architecture['best']['labels_pred'] = labels_pred
                self.is_best = 0
                self.save(self.k)


            print(tabulate(report, headers="keys", tablefmt="pretty"))

        print(classification_report(labels_truth, labels_pred,
              target_names=list(self.map_label.keys())))
        return val_loss / len(loader), accuracy

    def fit(self, **kwargs):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            logger.info("Start Training")
            train_loss, train_acc = self.train_one_epoch()
            self.info_train['train_acc'].append(train_acc)
            self.info_train['train_loss'].append(train_loss)
            if self.data_source.val_dataset:
                logger.info("Start evaluate")
                val_loss, val_acc = self.evaluate(self.val_loader)
                self.info_train['val_loss'].append(val_loss)
                self.info_train['val_acc'].append(val_acc)
                if val_loss < self.best_val_loss and val_acc > self.best_val_acc:
                    self.is_best = 1
                    self.k = epoch
                    print('best',self.k)
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc
                logger.info(f"Epoch: {epoch} --- Training loss: {train_loss} --- Train acc: {train_acc} --- Val loss: {val_loss} --- Val acc: {val_acc}"
                            f"Time: {time.time() -start_time}s")

            if self.data_source.test_dataset:
                logger.info("Start testing")
                _, test_acc = self.evaluate(
                    self.test_loader, mode_testing=True)
        if self.path_report:
            if self.over_sampling:
                path  = f'{self.path_report}/ann_report_over_sampling.pk'
            else:
                path  = f'{self.path_report}/ann_report.pk'

            with open(path, 'wb') as fin:
                pickle.dump(self.info_train, fin)
