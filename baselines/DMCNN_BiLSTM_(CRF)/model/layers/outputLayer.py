import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.global_variables import Global
from utils.evaluation import Evaluation

class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.num_class = config.getint("runtime", "num_class")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, prediction, labels):
        loss = self.criterion(prediction, labels)   # ([B, N], [B,])
        return loss