import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils.global_variables import Global
from model.layers import embedding, crf, outputLayer

class Crf(nn.Module):
    def __init__(self, config):
        super(Crf, self).__init__()
        self.config = config
        self.embedding = embedding.Embedding(config)
        self.rnn = DynamicRNN(config)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.hidden2tag = nn.Linear(in_features=config.getint("model", "hidden_size"),
                                    out_features=config.getint("runtime", "num_class") + 2,
                                    bias=True)
        self.pad_label_id = config.getint("data", "pad_label_id")
        self.crf = crf.CRF(tagset_size=config.getint("runtime", "num_class"), use_gpu=Global.device)
        print(self)

    def forward(self, data, **params):
        """
        :param data: 这一轮输入的数据
        :param params: 存放任何其它需要的信息
        """
        mode = params["mode"]
        tokens = data["tokens"]         # [B, L]
        labels = data["labels"]         # [B, L]
        lengths = data["lengths"]       # [B, ]
        flags = data["flags"]
        attention_masks = data["masks"] # [B, L]

        prediction = self.embedding(tokens)     # [B, L, E]
        prediction = self.dropout(prediction)
        prediction = self.rnn(prediction, lengths)  # [B, L, H]
        prediction = self.hidden2tag(prediction)    # [B, L, N+2]

        pad_masks = (labels != self.pad_label_id)
        loss_masks = ((attention_masks == 1) & pad_masks)

        if params["crf_mode"] == "train":
            crf_labels, crf_masks = self.to_crf_pad(labels, loss_masks)
            crf_logits, _ = self.to_crf_pad(prediction, loss_masks)
            loss = self.crf.neg_log_likelihood(crf_logits, crf_masks, crf_labels)
            return {"loss": loss,
                    "prediction": None,
                    "labels": None}

        elif params["crf_mode"] == "test":
            masks = (attention_masks == 1)
            crf_logits, crf_masks = self.to_crf_pad(prediction, masks)
            crf_masks = crf_masks.sum(axis=2) == crf_masks.shape[2]
            best_path = self.crf(crf_logits, crf_masks)
            temp_labels = (torch.ones(loss_masks.shape) * self.pad_label_id).to(torch.long)
            prediction = self.unpad_crf(best_path, crf_masks, temp_labels, masks)
            return {"loss": None,
                    "prediction": self.normalize(prediction, flags, lengths),
                    "labels": self.normalize(labels, flags, lengths)} if mode != "test" else {
                        "prediction": self.normalize(prediction, flags, lengths)
                    }

        else:
            raise NotImplementedError

    def normalize(self, logits, flags, lengths):
        results = []
        logits = logits.tolist()
        lengths = lengths.tolist()
        for logit, flag, length in zip(logits, flags, lengths):
            result = []
            for i in range(length):
                if flag[i] == 1:
                    assert logit[i] != self.pad_label_id
                    result.append(Global.id2label[str(logit[i])])
            results.append(result)
        return results

    def to_crf_pad(self, org_array, org_mask):
        crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
        crf_array = pad_sequence(crf_array, batch_first=True, padding_value=self.pad_label_id)
        crf_pad = (crf_array != self.pad_label_id)
        crf_array[~crf_pad] = 0
        return crf_array, crf_pad

    def unpad_crf(self, returned_array, returned_mask, org_array, org_mask):
        out_array = org_array.clone().detach().to(Global.device)
        out_array[org_mask] = returned_array[returned_mask]
        return out_array


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

    def forward(self, inputs, lengths):
        embedding_packed = nn.utils.rnn.pack_padded_sequence(input=inputs,
                                                             lengths=lengths,
                                                             batch_first=True,
                                                             enforce_sorted=False)
        outputs, _ = self.rnn(embedding_packed, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs,
                                                      batch_first=True,
                                                      padding_value=0.0,
                                                      total_length=self.sequence_length)
        return outputs
