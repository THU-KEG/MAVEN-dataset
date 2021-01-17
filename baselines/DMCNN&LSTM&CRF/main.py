import torch
import argparse
from utils.configparser_hook import get_config
from utils.global_variables import Global
from utils.initializer import initialize
from utils.runner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = ["config"]
    normal_args = ["gpu"]
    for arg in required_args + normal_args:
        parser.add_argument("--{}".format(arg), required=arg in required_args)
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu) if args.gpu and torch.cuda.is_available() else "cpu")
    Global.device = device
    print("Device:", device)

    config = get_config(args.config)
    config.add_section("runtime")

    parameters = initialize(config, device)

    run(parameters, config, device)    
    