import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss


class NYCMetric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """

    def __init__(self):
        super(NYCMetric, self).__init__()


    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred.to(pred.device) - gt.to(pred.device)).view(pred.size()[0], -1)
        mae = abs_err.sum(-1) / abs_err.size()[1]
        self.record.append(mae.cpu().numpy())

    def score_fun(self):
        r"""
        """
        records = np.concatenate(self.record)
        return [records.mean()]
