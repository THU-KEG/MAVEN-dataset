import os
import sys
import json
import copy
import torch
from utils.global_variables import Global
from utils.evaluation import Evaluation

def run(parameters, config, device):
    trained_epoch = -1
    max_epoch = config.getint("train", "epoch")
    valid_interval = config.getint("train", "valid_interval")
    saver = {}
    for epoch in range(trained_epoch + 1, max_epoch):
        run_one_epoch(parameters, config, device, epoch, "train")
        if epoch % valid_interval == 0:
            with torch.no_grad():
                valid_metric = run_one_epoch(parameters, config, device, epoch, "valid")
                test_metric = run_one_epoch(parameters, config, device, epoch, "test")
                print()
                if saver == {} or valid_metric["micro_f1"] > saver["valid"]["micro_f1"]:
                    saver["epoch"] = epoch
                    saver["valid"] = valid_metric
                    saver["test"] = test_metric
                    with open("./data/results_{}.jsonl".format(config.get("data", "formatter_name")[:-9]), "w", encoding="utf-8") as f:
                        for (k, v) in test_metric.items():
                            f.write(json.dumps({"id": k,
                                                "predictions": v}))
                            f.write('\n')
                        
    print("Best Epoch {}\nValid Metric: {}".format(saver["epoch"], saver["valid"]))


def run_one_epoch(parameters, config, device, epoch, mode):
    model = parameters["model"]

    if mode == "train":
        model.train()
        optimizer = parameters["optimizer"]
    elif mode == "valid" or mode == "test":
        model.eval()
    else:
        raise NotImplementedError

    dataset = copy.deepcopy(parameters["dataset_{}".format(mode)])
    pred = {}
    total_loss = 0
    evaluation = Evaluation(config)
    for step, data in enumerate(dataset):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        if mode == "train":
            optimizer.zero_grad()
        
        if config.get("model", "model_name") == "Crf":
            if mode != "test":
                results = model(data=data, mode=mode, crf_mode="train")
                loss = results["loss"]
                total_loss += loss.item()
                results = model(data=data, mode=mode, crf_mode="test")
                evaluation.expand(results["prediction"], results["labels"])
            else:
                results = model(data=data, mode=mode, crf_mode="test")
                prediction = results["prediction"]
                if not isinstance(prediction, list):
                    prediction = prediction.cpu().numpy().tolist()
                docids = data["docids"]
                canids = data["canids"]
                for doc, can, pre in zip(docids, canids, prediction):
                    if doc not in pred.keys():
                        pred[doc] = []
                    assert (len(can) == len(pre))
                    for c, p in zip(can, pre):
                        if p != "O":
                            p = p[2:]
                        assert p in Global.type2id.keys()
                        pred[doc].append({"id": c,
                                          "type_id": Global.type2id[p]})
        else:
            results = model(data=data, mode=mode)
            if mode != "test":
                loss = results["loss"]
                total_loss += loss.item()
                evaluation.expand(results["prediction"], results["labels"])
            else:
                prediction = results["prediction"].cpu().numpy().tolist()
                docids = data["docids"]
                canids = data["canids"]
                for did, cid, pre in zip(docids, canids, prediction):
                    if did not in pred.keys():
                        pred[did] = []
                    pred[did].append({"id": cid,
                                      "type_id": pre})
        if mode != "test":
            print("\r{}: Epoch {} Step {:0>4d}/{} | Loss = {:.4f}".format(mode, epoch, step + 1, len(dataset), round(total_loss / (step + 1), 4)), end="")
        else:
            print("\r{}: Epoch {} Step {:0>4d}/{}".format(mode, epoch, step + 1, len(dataset)), end="")

        if mode == "train":
            loss.backward()
            optimizer.step()

    if mode != "test":
        metric = evaluation.get_metric("all")
        sys.stdout.write("\r")
        print("\r{}: Epoch {} | Metric: {}".format(mode, epoch, metric))
        return metric
    else:
        return pred