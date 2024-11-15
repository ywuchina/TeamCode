import torch
import torch.nn as nn
import torch.nn.functional as F


def frobeniusNormLoss(predicted, igt):
    """ |predicted*igt - I| (should be 0) """
    # predicted:[B,4,4]
    # igt;[B,4,4]
    error = predicted.matmul(igt)
    I = torch.eye(4).to(error).view(1, 4, 4).expand(error.size(0), 4, 4)
    return F.mse_loss(error, I, size_average=True) * 16


class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, predicted, igt):
        return frobeniusNormLoss(predicted, igt)


