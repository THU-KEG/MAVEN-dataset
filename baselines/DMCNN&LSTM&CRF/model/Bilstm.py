import torch
import torch.nn as nn
from utils.global_variables import Global
from model.layers import embedding, outputLayer

class Bilstm(nn.Module):
    def __init__(self, config):
        super(Bilstm, self).__init__()
        self.config = config
        self.embedding = embedding.Embedding(config)
        self.rnn = DynamicRNN(config)
        
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.fc = nn.Linear(in_features=config.getint("model", "hidden_size"),
                            out_features=config.getint("runtime", "num_class"),
                            bias=True)
        self.out = outputLayer.OutputLayer(config)
        print(self)


    def forward(self, data, **params):
        """
        :param data: 这一轮输入的数据
        :param params: 存放任何其它需要的信息
        """
        mode = params["mode"]
        tokens = data["tokens"]     # [B, L]
        if mode != "test":
            labels = data["labels"]     # [B, ]
        lengths = data["lengths"]   # [B, ]
        indices = data["indices"]   # [B, ]
        
        prediction = self.embedding(tokens)     # [B, L, E]
        prediction = self.dropout(prediction)
        prediction = self.rnn(prediction, lengths, indices)    # [B, H]
        prediction = self.fc(prediction)    # [B, N]

        if mode != "test":
            loss = self.out(prediction, labels)
        prediction = torch.argmax(prediction, dim=1)

        return {"loss": loss, 
                "prediction": prediction, 
                "labels": labels} if mode != "test" else {
                    "prediction": prediction
                }


class DynamicRNN(nn.Module):
    def __init__(self, config):
        super(DynamicRNN, self).__init__()
        self.embedding_size = config.getint("runtime", "embedding_size")
        self.sequence_length = config.getint("runtime", "sequence_length")
        self.num_layers = config.getint("model", "num_layers")
        self.hidden_size = config.getint("model", "hidden_size")
        self.rnn = nn.LSTM(input_size=self.embedding_size,
                           hidden_size=self.hidden_size // 2,
                           num_layers=self.num_layers,
                           bias=True,
                           batch_first=True,
                           dropout=0,
                           bidirectional=True)

    def forward(self, inputs, lengths, indices):
        embedding_packed = nn.utils.rnn.pack_padded_sequence(input=inputs,
                                                             lengths=lengths,
                                                             batch_first=True,
                                                             enforce_sorted=False)
        outputs, _ = self.rnn(embedding_packed, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs,
                                                      batch_first=True,
                                                      padding_value=0.0,
                                                      total_length=self.sequence_length)
        outputs = outputs[torch.arange(inputs.shape[0]), indices]
        return outputs
                                
        

        