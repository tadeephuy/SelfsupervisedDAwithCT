import torch
from torch import nn
from torch.nn import functional as F

class KLD(nn.KLDivLoss):
    def forward(self, pred, target):
        super().forward(F.logsigmoid(pred), target.detach().sigmoid())

def entropy_minimization(x):
    return torch.sum(-x * torch.log(x)).mean()