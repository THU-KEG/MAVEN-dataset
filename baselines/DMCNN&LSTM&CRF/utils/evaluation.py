import torch
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import precision_score, recall_score, f1_score
from utils.global_variables import Global

import warnings
warnings.filterwarnings('always')

class Evaluation(object):
    def __init__(self, config):
        super(Evaluation).__init__()
        self.config = config
        self.y_pred = []
        self.y_true = []
        self.labels = [v for (k, v) in Global.label2id.items() if k != "None"]

    def get_metric(self, mode, batch_pred=None, batch_true=None):
        average = ["micro", "macro"]
        metrics = ["precision", "recall", "f1"]
        ret = {"{}_{}".format(t1, t2) : 0.0 for t1 in average for t2 in metrics}
        if mode == "batch":
            assert batch_pred is not None
            assert batch_true is not None
            batch_pred = torch.argmax(batch_pred, dim=1)
            y_pred = self.normalize(batch_pred)
            y_true = self.normalize(batch_true)
        elif mode == "all":
            y_pred = self.y_pred
            y_true = self.y_true
        else:
            raise NotImplementedError
        assert len(y_pred) == len(y_true)
        for av in average:
            if self.config.has_option("data", "BIO"):
                ret["{}_precision".format(av)] = precision_score(y_true=y_true, y_pred=y_pred)
                ret["{}_recall".format(av)] = recall_score(y_true=y_true, y_pred=y_pred)
                ret["{}_f1".format(av)] = f1_score(y_true=y_true, y_pred=y_pred)
            else:
                ret["{}_precision".format(av)], ret["{}_recall".format(av)], ret["{}_f1".format(av)], _ = precision_recall_fscore_support(y_true=y_true, 
                                                                                                                                          y_pred=y_pred, 
                                                                                                                                          labels=self.labels, 
                                                                                                                                          average=av,
                                                                                                                                          zero_division=0)
        return {key : ('%.4f' % value) for key, value in ret.items() if key.startswith("micro") or key.endswith("f1")}

    def expand(self, batch_pred, batch_true):
        y_pred = batch_pred if isinstance(batch_pred, list) else self.normalize(batch_pred)
        y_true = batch_true if isinstance(batch_true, list) else self.normalize(batch_true)
        self.y_pred += y_pred
        self.y_true += y_true

    def normalize(self, x):
        return x.cpu().numpy().tolist()
