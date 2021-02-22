import os
import json
import copy
import codecs
import numpy as np
from tqdm import tqdm
from utils.global_variables import Global

class MavenReader(object):
    def __init__(self, config):
        self.config = config
        self.data = []
        self.raw_dir = "./raw"
        self.data_dir = "./data"
        self.flag_dir = "{}{}".format(config.get("data", "reader_name")[:-6], "crf" if config.has_option("data", "BIO") else "")
        self.word2vec_source_file = config.get("data", "word2vec_file")
        self.word2vec_file = "word2vec.npy"
        self.modes = ["train", "valid", "test"]

    def read(self, mode):
        """
        :param mode: train/valid/test
        :return: [{"tokens": list(int), "labels": list(int)}, ...]
        """
        self.data.clear()
        if not os.path.exists(os.path.join(self.data_dir, self.flag_dir, 'flag')):
            os.makedirs(os.path.join(self.data_dir, self.flag_dir))
            self.preprocess()
        with open(os.path.join(self.data_dir, self.flag_dir, "{}_processed.json".format(mode)), "r+", encoding="utf-8") as f:
            data = json.load(f)
        if Global.word2vec_mat is None:
            Global.word2vec_mat = np.load(os.path.join(self.data_dir, self.word2vec_file))
            Global.word2id = data["word2id"]
            Global.id2word = data["id2word"]
            Global.label2id = data["label2id"]
            Global.id2label = data["id2label"]
            if self.config.has_option("data", "BIO"):
                Global.type2id = data["type2id"]
        for item in data["info"]:
            tokens = [data["word2id"][x]  if x in data["word2id"] else data["word2id"]["<UNK>"] for x in item["tokens"]]
            if mode != "test":
                labels = [data["label2id"][x] for x in item["labels"]]
            canids = item["canids"]     
            docids = item["docids"]
            if self.config.has_option("data", "split_labels"):
                for i in range(len(canids)):
                    if item["flags"][i]:
                        if mode != "test":
                            temp = {"tokens": tokens,
                                    "labels": labels[i],
                                    "canids": canids[i],
                                    "docids": docids,
                                    "index": i}
                        else:
                            temp = {"tokens": tokens,
                                    "canids": canids[i],
                                    "docids": docids,
                                    "index": i}
                        self.data.append(temp)
            else:
                if mode != "test":
                    temp = {"tokens": tokens,
                            "labels": labels,
                            "canids": canids,
                            "docids": docids,
                            "flags": item["flags"]}
                else:
                    temp = {"tokens": tokens,
                            "canids": canids,
                            "docids": docids,
                            "flags": item["flags"]}
                self.data.append(temp)

        self.config.set("runtime", "vocab_size", Global.word2vec_mat.shape[0])
        self.config.set("runtime", "embedding_size", Global.word2vec_mat.shape[1])
        self.config.set("runtime", "num_class", len(data["label2id"]))
        self.config.set("runtime", "sequence_length", data["sequence_length"])

        print("Mode: {} | Dataset Size = {}".format(mode, len(self.data)))
        return copy.deepcopy(self.data)

    def preprocess(self):
        """
        :return: 输出文件、整合数据以及词向量矩阵
        整合数据格式：{
            "info":[{"tokens": list(str), "labels": list(str), "flags": list(bool)}, ...],
            "word2id": {"<PAD>": 0, "<UNK>": 1},
            "id2word": {0: "<PAD>", 1: "<UNK>"},
            "label2id": {"None": 0},
            "id2label": {0: "None"},
            "sequence_length": int
        }
        """

        embedding_dict = self.load_embedding_dict(os.path.join(self.raw_dir, self.word2vec_source_file))

        processed_data = {"info_train": [],
                          "info_valid": [],
                          "info_test": [],
                          "word2id": {},
                          "id2word": {},
                          "label2id": {},
                          "id2label": {},
                          "sequence_length": 0}

        if self.config.has_option("data", "BIO"):
            processed_data["label2id"]["O"] = 0
            processed_data["id2label"][0] = "O"
            processed_data["type2id"] = {"O": 0}
        else:
            processed_data["label2id"]["None"] = 0
            processed_data["id2label"][0] = "None"
            
        for mode in self.modes:
            with codecs.open(os.path.join(self.raw_dir, "{}.jsonl".format(mode)), 'r', encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip()
                    doc = json.loads(line)
                    docids = doc["id"]
                    doc_tokens, doc_labels, doc_canids, doc_flags = [], [], [], []
                    for item in doc["content"]:
                        doc_tokens.append(item["tokens"])

                    if self.config.has_option("data", "BIO"):

                        for tokens in doc_tokens:
                            if mode != "test":
                                doc_labels.append(["O"] * len(tokens))
                            doc_canids.append([""] * len(tokens))
                            doc_flags.append([0] * len(tokens))
                        
                        if mode == "test":
                            for candi in doc["candidates"]:
                                for i in range(candi["offset"][0], candi["offset"][1]):
                                    doc_canids[candi["sent_id"]][i] = candi["id"]
                                    doc_flags[candi["sent_id"]][i] = 1
                        else:
                            for event in doc["events"]:
                                tp = event["type"].replace("-", "_")
                                if tp not in processed_data["type2id"]:
                                    processed_data["type2id"][tp] = event["type_id"]
                                for mention in event["mention"]:
                                    for i in range(mention["offset"][0], mention["offset"][1]):
                                        doc_labels[mention["sent_id"]][i] = ("B-" + tp) if (i == mention["offset"][0]) else ("I-" + tp)
                                        doc_canids[mention["sent_id"]][i] = mention["id"]
                                        doc_flags[mention["sent_id"]][i] = 1

                    else:

                        for tokens in doc_tokens:
                            if mode != "test":
                                doc_labels.append(["None"] * len(tokens))
                            doc_canids.append([""] * len(tokens))
                            doc_flags.append([0] * len(tokens))
                            processed_data["sequence_length"] = max(processed_data["sequence_length"], len(tokens))
                        
                        if mode == "test":
                            for candi in doc["candidates"]:
                                for i in range(candi["offset"][0], candi["offset"][1]):
                                    doc_canids[candi["sent_id"]][i] = candi["id"]
                                    doc_flags[candi["sent_id"]][i] = 1
                        else:
                            for event in doc["events"]:
                                if event["type"] not in processed_data["label2id"]:
                                    processed_data["label2id"][event["type"]] = event["type_id"]
                                    processed_data["id2label"][event["type_id"]] = event["type"]
                                for mention in event["mention"]:
                                    for i in range(mention["offset"][0], mention["offset"][1]):
                                        doc_labels[mention["sent_id"]][i] = event["type"]
                                        doc_canids[mention["sent_id"]][i] = mention["id"]
                                        doc_flags[mention["sent_id"]][i] = 1

                    if mode != "test":
                        for mention in doc["negative_triggers"]:
                            for i in range(mention["offset"][0], mention["offset"][1]):
                                doc_canids[mention["sent_id"]][i] = mention["id"]
                                doc_flags[mention["sent_id"]][i] = 1

                        for tokens, labels, canids, flags in zip(doc_tokens, doc_labels, doc_canids, doc_flags):
                            processed_data["info_{}".format(mode)].append({"tokens": tokens,
                                                                           "labels": labels,
                                                                           "canids": canids,
                                                                           "flags": flags,
                                                                           "docids": docids})
                            if self.config.has_option("data", "BIO"):
                                for label in labels:
                                    if label not in processed_data["label2id"]:
                                        id = len(processed_data["label2id"])
                                        processed_data["label2id"][label] = id
                                        processed_data["id2label"][id] = label
                    else:
                        for tokens, canids, flags in zip(doc_tokens, doc_canids, doc_flags):
                            processed_data["info_{}".format(mode)].append({"tokens": tokens,
                                                                           "canids": canids,
                                                                           "flags": flags,
                                                                           "docids": docids})
                    

        if self.config.has_option("data", "BIO"):
            processed_data["sequence_length"] = self.config.getint("data", "sequence_length")
        
        word2vec_mat = []
        for (k, v) in embedding_dict.items():
            id = len(processed_data["word2id"])
            processed_data["word2id"][k] = id
            processed_data["id2word"][id] = k
            word2vec_mat.append(v)
        word2vec_mat = np.array(word2vec_mat, dtype=np.float32)
        if not os.path.exists(os.path.join(self.data_dir, self.word2vec_file)):
            np.save(os.path.join(self.data_dir, self.word2vec_file), word2vec_mat)

        for mode in self.modes:
            with open(os.path.join(self.data_dir, self.flag_dir, "{}_processed.json".format(mode)), "w", encoding="utf-8") as f:
                temp_data = {"info": processed_data["info_{}".format(mode)],
                             "word2id": processed_data["word2id"],
                             "id2word": processed_data["id2word"],
                             "label2id": processed_data["label2id"],
                             "id2label": processed_data["id2label"],
                             "sequence_length": processed_data["sequence_length"]}
                if self.config.has_option("data", "BIO"):
                    temp_data["type2id"] = processed_data["type2id"]
                json.dump(temp_data, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.data_dir, self.flag_dir, 'flag'), "w+") as f:
            f.write("")

        

    def load_embedding_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        embedding_dict = {}
        for line in lines:
            split = line.split(" ")
            embedding_dict[split[0]] = np.array(list(map(float, split[1:])))
        return embedding_dict