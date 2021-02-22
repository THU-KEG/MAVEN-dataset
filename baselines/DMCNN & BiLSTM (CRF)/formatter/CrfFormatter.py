import torch
from utils.global_variables import Global

class CrfFormatter(object):
    def __init__(self, config):
        self.config = config
        self.pad_label_id = config.getint("data", "pad_label_id")

    def process(self, data, mode):
        """
        :param data: [{"tokens": list(int), "labels": list(int)}, ...]
        :param mode: train/valid/test
        :return: {"tokens": LongTensor,
                  "lables": LongTensor,
                  "masks": LongTensor,
                  "lengths": LongTensor}
        """
        tokens, canids, labels, flags, masks, lengths, docids = [], [], [], [], [], [], []

        sequence_length = self.config.getint("runtime", "sequence_length")

        for item in data:
            docid = item["docids"]
            token = item["tokens"]
            canid_ = item["canids"]
            if mode != "test":
                label = item["labels"]
            else:
                label = [0] * len(token)
            flag = item["flags"] if "flags" in item else [1] * len(token)
            if len(token) > sequence_length:
                token = token[:sequence_length]
                canid_ = canid_[:sequence_length]
                label = label[:sequence_length]
                flag = flag[:sequence_length]
            length = len(token)
            token += [Global.word2id["<PAD>"]] * (sequence_length - length)
            label += [self.pad_label_id] * (sequence_length - length)
            canid = []
            for i in range(len(flag)):
                if flag[i] == 1:
                    canid.append(canid_[i])
            flag += [0] * (sequence_length - length)
            for i in range(sequence_length):
                if i < length and flag[i] == 1:
                    assert label[i] != self.pad_label_id
            docids.append(docid)
            tokens.append(token)
            canids.append(canid)
            labels.append(label)
            flags.append(flag)
            masks.append([1] * length + [0] * (sequence_length - length))
            lengths.append(length)
            for i in range(length):
                assert labels[-1][i] != self.pad_label_id

        tlt = lambda t: torch.LongTensor(t)
        tt = lambda t: torch.Tensor(t)
        
        tokens, labels, masks, lengths = tlt(tokens), tlt(labels), tlt(masks), tlt(lengths)

        return {"tokens": tokens,
                "labels": labels,
                "flags": flags,
                "masks": masks,
                "lengths": lengths,
                "canids": canids,
                "docids": docids}

