import torch.nn.functional as F
from model.sdr import pairwise_neg_sisdr
from model.pit_wrapper import PITLossWrapper
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)


