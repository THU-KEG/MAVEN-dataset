import torch
from utils.global_variables import Global

class BilstmFormatter(object):
    def __init__(self, config):
        self.config = config

    def process(self, data, mode):
        """
        :param data: [{"tokens": list(int), "labels": int}, ...]
        :param mode: train/valid/test
        :return: {"tokens": LongTensor,
                  "lables": LongTensor,
                  "lengths": LongTensor,
                  "indices": LongTensor}
        """
        tokens, canids, labels, lengths, indices, docids = [], [], [], [], [], []

        sequence_length = self.config.getint("runtime", "sequence_length")

        for item in data:
            length = len(item["tokens"])
            docids.append(item["docids"])
            tokens.append(item["tokens"] + [Global.word2id["<PAD>"]] * (sequence_length - length))
            canids.append(item["canids"])
            if mode != "test":
                labels.append(item["labels"])
            lengths.append(length)
            indices.append(item["index"])

        tlt = lambda t: torch.LongTensor(t)
        tt = lambda t: torch.Tensor(t)
        tokens, lengths, indices = tlt(tokens), tlt(lengths), tlt(indices)
        if mode != "test":
            labels = tlt(labels)

        return {"tokens": tokens,
                "labels": labels,
                "lengths": lengths,
                "indices": indices,
                "canids": canids,
                "docids": docids} if mode != "test" else {
                    "tokens": tokens,
                    "lengths": lengths,
                    "indices": indices,
                    "canids": canids,
                    "docids": docids
                }
