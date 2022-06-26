import torch

from torch import nn, Tensor
from typing import List
from src.module import LabelSmoothingCrossEntropyLoss


class CnnCls(nn.Module):
    def __init__(self,
                 n_cnn: int,
                 kernel_size: List,
                 pooling_kernel_size: int,
                 n_filter: List,
                 n_dense: int,
                 n_tensor_dense: List,
                 n_labels: int,
                 n_vocab: int,
                 embedding_dim: int,
                 dropout: float,
                 use_label_smoothing: bool = False,
                 weight_contribution: Tensor = None,
                 smoothing_value: float = 0.1):
        super().__init__()
        assert len(kernel_size) == len(n_filter) == n_cnn, "Invalid input"
        assert len(n_tensor_dense) == n_dense, "Invalid input"

        self.n_cnn = n_cnn
        self.kernel_size = kernel_size
        self.pooling_kernel_size = pooling_kernel_size
        self.n_dense = n_dense
        self.n_tensor_dense = n_tensor_dense
        self.n_labels = n_labels
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.use_label_smoothing = use_label_smoothing
        self.weight_contribution = weight_contribution
        self.smoothing_value = smoothing_value

        cnn_layers = [nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim, padding_idx=0)]
        base_channel = 1
        for i in range(n_cnn):
            cnn_layers.append(nn.Conv2d(in_channels=base_channel, out_channels=base_channel * n_filter[i],
                                        kernel_size=kernel_size[i], padding='same'))
            cnn_layers.append(nn.SiLU())
            cnn_layers.append(nn.Dropout2d(p=dropout))
            cnn_layers.append(nn.MaxPool2d(kernel_size=pooling_kernel_size))
            base_channel *= n_filter[i]

        cnn_layers.append(nn.Flatten())

        cnn_layers.append(nn.LazyLinear(out_features=n_tensor_dense[0]))
        cnn_layers.append(nn.SiLU())
        cnn_layers.append(nn.Dropout(p=dropout))
        for i in range(n_dense-1):
            cnn_layers.append(nn.Linear(in_features=n_tensor_dense[i], out_features=n_tensor_dense[i+1]))
            cnn_layers.append(nn.SiLU())
            cnn_layers.append(nn.Dropout(p=dropout))

        cnn_layers.append(nn.Linear(in_features=n_tensor_dense[-1], out_features=n_labels))
        cnn_layers.append(nn.Softmax())

        self.model = nn.Sequential(*cnn_layers)

        if use_label_smoothing:
            self.loss_fn = LabelSmoothingCrossEntropyLoss(
                weight=weight_contribution,
                smoothing_value=smoothing_value
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        token_ids = inputs['input_ids']
        labels = inputs['labels']
        output = self.model(token_ids)

        loss = 0
        if labels is not None:
            loss += self.loss_fn(output, labels)
        return loss, output
