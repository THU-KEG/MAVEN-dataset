import torch
from utils.global_variables import Global


class DmcnnFormatter(object):
    def __init__(self, config):
        self.config = config

    def process(self, data, mode):
        """
        :param data: [{"tokens": list(int), "labels": int}, ...]
        :param mode: train/valid/test
        :return: {"tokens": LongTensor,
                  "lables": LongTensor,
                  "pfs": LongTensor,
                  "llfs": LongTensor,
                  "masks": Tensor}
        """
        tokens, canids, labels, masks, pfs, llfs, docids = [], [], [], [], [], [], []

        sequence_length = self.config.getint("runtime", "sequence_length")

        for item in data:
            length = len(item["tokens"])
            docids.append(item["docids"])
            tokens.append(item["tokens"] + [Global.word2id["<PAD>"]] * (sequence_length - length))
            canids.append(item["canids"])
            if mode != "test":
                labels.append(item["labels"])
            mask = []
            for i in range(sequence_length):
                if 0 <= i <= item["index"]:
                    mask.append([100, 0])
                elif i < length:
                    mask.append([0, 100])
                else:
                    mask.append([0, 0])
            masks.append(mask)
            pfs.append([abs(item["index"] - x) for x in range(sequence_length)])
            if item["index"] == 0:
                llfs.append([0] + tokens[-1][item["index"]: item["index"] + 2])
            elif item["index"] == sequence_length - 1:
                llfs.append(tokens[-1][item["index"] - 1: item["index"] + 1] + [0])
            else:
                llfs.append(tokens[-1][item["index"] - 1: item["index"] + 2])
            assert len(llfs[-1]) == 3

        tlt = lambda t: torch.LongTensor(t)
        tt = lambda t: torch.Tensor(t)
        tokens, pfs, llfs = tlt(tokens), tlt(pfs), tlt(llfs)
        masks = tt(masks)
        if mode != "test":
            labels = tlt(labels)

        return {"tokens": tokens, 
                "labels": labels,
                "pfs": pfs,
                "llfs": llfs,
                "masks": masks,
                "canids": canids,
                "docids": docids} if mode != "test" else {
                    "tokens": tokens, 
                    "pfs": pfs,
                    "llfs": llfs,
                    "masks": masks,
                    "canids": canids,
                    "docids": docids
                }
