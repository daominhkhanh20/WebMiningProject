import os
import logging
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from tabulate import tabulate

from src.dataset import BertDataSource, CommentCollate
from src.model import BertCommentModel
from src.learner import BaseLeaner
from src.utils.io import *
from src.constants import *

logger = logging.getLogger(__name__)


class BertLearner(BaseLeaner):
    def __init__(self, mode: str = 'training', pretrained_model: str = 'vinai/phobert-base',
                 data_source: BertDataSource = None,
                 batch_size: int = 64, n_epochs: int = 10, use_label_smoothing: bool = False, fine_tune: bool = True,
                 smoothing_value: float = 0.1, learning_rate: float = 1e-6, path_save_model: str = 'models',
                 is_save_best_model: bool = False, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if mode.lower() not in [INFERENCE_MODE, TRAINING_MODE]:
            NotImplementedError(f"{mode} isn't support")
        if mode == INFERENCE_MODE:
            if path_save_model is not None:
                path_save_model = os.path.abspath(path_save_model)
            else:
                raise Exception(f"{path_save_model} isn't exist")

            self.config_architecture = get_config_architecture(path_save_model)
            self.pretrained_model = self.config_architecture['architecture']['encoder']['pretrained_name']
        elif mode.lower() == TRAINING_MODE:
            self.pretrained_model = pretrained_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_encoder = AutoModel.from_pretrained(self.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.pad_id = self.tokenizer.pad_token_id

        if mode == INFERENCE_MODE:
            pass
        elif mode == TRAINING_MODE:
            self.data_source = data_source
            self.map_label = self.data_source.train_dataset.map_label
            self.n_labels = len(self.map_label)
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.learning_rate = learning_rate
            self.use_smoothing_label = use_label_smoothing
            self.smoothing_value = smoothing_value
            self.path_save_model = path_save_model
            self.is_save_best_model = is_save_best_model
            self.model = BertCommentModel(
                bert_encoder=self.bert_encoder,
                n_labels=len(self.data_source.train_dataset.map_label),
                dropout=dropout,
                fine_tune=fine_tune,
                use_label_smoothing=use_label_smoothing,
                weight_contribution=self.data_source.weight_contribution,
                smoothing_value=smoothing_value
            ).to(self.device)

            if os.path.exists(path_save_model) is False:
                os.makedirs(self.path_save_model, exist_ok=True)

            self.config_architecture = {'model': {
                "architecture": {
                    "encoder": {
                        "pretrained_model": self.pretrained_model,
                        "fine_tune": fine_tune
                    },
                    "head": {
                        "linear_1": {
                            "in_features": self.model.linear[0].in_features,
                            "out_feature": self.model.linear[0].out_features
                        },
                        "layer_norm": {
                            "in_features": self.model.linear[1].normalized_shape,
                        },
                        "dropout": self.model.linear[2].p,
                        "linear_2": {
                            "in_features": self.model.linear[3].in_features,
                            "out_feature": self.model.linear[3].out_features
                        },
                    }
                }
            }, 'hyper_parameter': {
                'num_epochs': n_epochs,
                'learning_rate': learning_rate,
                'use_label_smoothing': use_label_smoothing,
                'smoothing_value': smoothing_value
            }}
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            self.train_loader = self.make_loader(self.data_source.train_dataset)
            if self.data_source.val_dataset:
                self.val_loader = self.make_loader(self.data_source.val_dataset)
            if self.data_source.test_dataset:
                self.test_loader = self.make_loader(self.data_source.test_dataset)
            self.num_warmup_steps = int(len(self.train_loader) / 2)
            self.num_training_steps = len(self.train_loader)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_training_steps,
                num_training_steps=self.num_training_steps
            )
            self.config_architecture['hyper_parameter']['num_warmup_steps'] = self.num_warmup_steps
            self.config_architecture['hyper_parameter']['num_training_steps'] = self.num_training_steps
            self.config_architecture['data'] = {
                "map_label": self.map_label,
                "n_label": self.n_labels,
                "weight_contribution": self.data_source.weight_contribution.cpu().numpy()
            }
            self.best_val_loss = sys.maxsize
            self.best_val_acc = 0
            self.config_architecture['best_model'] = {
                "val_loss": self.best_val_loss,
                "val_acc": self.best_val_acc,
                "epoch": 0
            }

    def make_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=CommentCollate(pad_id=self.pad_id))

    def save(self, epoch, **kwargs):
        self.config_architecture["best_model"]["val_loss"] = self.best_val_loss
        self.config_architecture["best_model"]["val_acc"] = self.best_val_acc
        self.config_architecture["best_model"]["epoch"] = epoch
        save_json(self.config_architecture, f"{self.path_save_model}/config_architecture.json")
        if self.is_save_best_model:
            weight_name = 'weight.pth'
        else:
            weight_name = f"weight{epoch}.pth"
        torch.save(self.model.state_dict(), f"{self.path_save_model}/{weight_name}")
        logger.info("Save model done")

    def train_one_epoch(self, **kwargs):
        self.model.train()
        train_loss = 0
        for idx, sample in tqdm(self.train_loader, total=len(self.train_loader)):
            for key in sample.keys():
                sample[key] = sample[key].to(self.device)
            loss, _ = self.model(**sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def evaluate(self, loader, **kwargs):
        mode_testing = kwargs.get('mode_testing', False)
        labels_truth, labels_pred = [], []
        val_loss = 0
        for idx, sample in tqdm(loader, total=len(loader)):
            for key in sample.keys():
                sample[key] = sample[key].to(self.device)
            loss, outs = self.model(**sample)
            label_pred = torch.argmax(outs, dim=-1).detach().cpu().numpy().tolist()
            label_truth = sample['labels'].detach().cpu().numpy().tolist()
            labels_pred.extend(label_pred)
            labels_truth.extend(label_truth)
            val_loss += loss.item()

        accuracy = accuracy_score(labels_truth, labels_pred)
        if mode_testing:
            report = {
                "f1_score": f1_score(labels_truth, labels_pred),
                "recall_score": recall_score(labels_truth, labels_pred),
                "accuracy_score": accuracy
            }
            logger.info("Detail result for testing")
            print(tabulate(report, headers="keys", tablefmt="pretty"))
        print(classification_report(labels_truth, labels_pred, target_names=list(self.map_label.keys())))
        return val_loss / len(loader), accuracy

    def fit(self, **kwargs):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            logger.info("Start Training")
            train_loss = self.train_one_epoch()
            if self.data_source.val_dataset:
                logger.info("Start evaluate")
                val_loss, val_acc = self.evaluate(self.val_loader)
                if val_loss < self.best_val_loss and val_acc > self.best_val_acc:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc
                    self.save(epoch)
                logger.info(f"Epoch: {epoch} --- Training loss: {train_loss} --- Val loss: {val_loss} --- Val acc: {val_acc}"
                            f"Time: {time.time() -start_time}s")

            if self.data_source.test_dataset:
                logger.info("Start testing")
                _, test_acc = self.evaluate(self.test_loader, mode_testing = True)

