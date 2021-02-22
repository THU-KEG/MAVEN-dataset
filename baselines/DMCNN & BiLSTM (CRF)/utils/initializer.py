import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

get_class = lambda attr, name: getattr(__import__("{}.{}".format(attr, name), fromlist=["dummy"]), name)

def initialize(config, device):
    parameters = {}
    
    reader = get_class("reader", config.get("data", "reader_name"))(config)
    formatter = get_class("formatter", config.get("data", "formatter_name"))(config)
    batch_size = config.getint("train" ,"batch_size")
    shuffle = config.getboolean("train", "shuffle")

    collate_fn_decr = lambda mode: (lambda data, mode=mode: formatter.process(data, mode))
    
    dataset_train = reader.read("train")
    dataset_valid = reader.read("valid")
    dataset_test = reader.read("test")
    parameters["dataset_train"] = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_decr("train"))
    parameters["dataset_valid"] = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_decr("valid"))
    parameters["dataset_test"] = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_decr("test"))
    
    parameters["model"] = get_class("model", config.get("model", "model_name"))(config)
    parameters["model"] = parameters["model"].to(device)
    
    parameters["optimizer"] = get_optim(parameters["model"], config)

    return parameters

def get_optim(model, config):
    hyper_params = {key: value for key, value in config["optimizer"].items() if key != "optimizer_name"}
    optimizer_name = config.get("optimizer", "optimizer_name")
    optimizer = getattr(optim, optimizer_name)
    command = "optim(params, {})".format(", ".join(["{}={}".format(key, value) for key, value in hyper_params.items()]))
    return eval(command, {"optim": optimizer, "params": model.parameters()})
