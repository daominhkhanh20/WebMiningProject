import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn

class ANNModel(nn.Module):

    def __init__(self,
                input_size,
                n_labels,
                drop_out: float = 0.2):
        super(ANNModel, self).__init__()
        self.input_size = input_size
        self.n_labels = n_labels
        self.drop_out = drop_out

        self.linear = nn.Sequential(
                nn.Linear(self.input_size, 512),
                nn.ReLU(),
                nn.Dropout(p=self.drop_out),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=self.drop_out),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=self.drop_out),

                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=self.drop_out),

                nn.Linear(128, 10),
                nn.ReLU(),
                nn.Dropout(p=self.drop_out),

                nn.Linear(10, self.n_labels),
                nn.Softmax()
        )

    def forward(self, x):
        x = self.linear(x)
        return x