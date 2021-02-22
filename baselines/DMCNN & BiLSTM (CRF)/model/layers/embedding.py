import torch
import torch.nn as nn
import numpy as np
from utils.global_variables import Global

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        if Global.word2vec_mat is None:
            weight = None
            self.vocab_size = config.getint("runtime", "vocab_size")
            self.embedding_size = config.getint("runtime", "embedding_size")
        else:
            weight = torch.from_numpy(Global.word2vec_mat).float()
            self.vocab_size, self.embedding_size = weight.size()
        self.embedding = nn.Embedding.from_pretrained(weight)
        
    def forward(self, input):
        return self.embedding(input)
