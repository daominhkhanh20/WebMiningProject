from torch import nn
from torch import Tensor
import torch
from torch.nn import functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Tensor, smoothing_value=0.1, reduction='mean', ignore_index: int = -100):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weigth = weight.to(device)
        self.smoothing_value = smoothing_value
        self.reduction = reduction
        self.ignore_index = ignore_index

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, outputs, targets):
        n_classes = outputs.size(1)
        log_preds = F.log_softmax(outputs, dim=1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, targets, weight=self.weigth, reduction=self.reduction)
        return (1 - self.smoothing_value) * nll + self.smoothing_value * (loss / n_classes)
