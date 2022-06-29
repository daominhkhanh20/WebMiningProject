from transformers import AutoModel, BertPreTrainedModel
import torch
from torch import nn, Tensor
import logging
from src.module import LabelSmoothingCrossEntropyLoss

logger = logging.getLogger(__name__)

class BertCommentModel(nn.Module):
    def __init__(self,
                 bert_encoder: BertPreTrainedModel,
                 n_labels: int,
                 dropout: float = 0.2,
                 fine_tune: bool = True,
                 use_label_smoothing: bool = False,
                 weight_contribution: Tensor = None,
                 smoothing_value: float = 0.1):
        super(BertCommentModel, self).__init__()
        self.bert_encoder = bert_encoder
        self.hidden_size = self.bert_encoder.config.hidden_size
        self.n_labels = n_labels
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, n_labels)
        )
        if use_label_smoothing:
            self.loss_fn = LabelSmoothingCrossEntropyLoss(
                weight=weight_contribution,
                smoothing_value=smoothing_value
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        if fine_tune:
            logger.info("Turn on fine tune mode")

        for child in self.bert_encoder.children():
            for param in child.parameters():
                param.requires_grad = fine_tune

    def forward(self, input_ids: Tensor,
                attention_masks: Tensor,
                labels: Tensor = None):
        outputs = self.bert_encoder(
            input_ids,
            attention_masks
        )
        pooling_outputs = outputs.pooler_output
        out_logits = self.linear(pooling_outputs)
        loss = 0
        if labels is not None:
            loss += self.loss_fn(out_logits, labels)
        return loss, out_logits
