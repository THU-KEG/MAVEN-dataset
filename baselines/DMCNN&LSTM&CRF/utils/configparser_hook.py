import functools
import configparser

class ConfigParserHook(object):
    def __init__(self):
        self.config = configparser.RawConfigParser()

    def read(self, config_file):
        self.config.read(config_file, encoding="utf-8")

def set_hook(func_name):
    @functools.wraps(getattr(configparser.RawConfigParser, func_name))
    def wrapper(self, *args, **kwargs):
        return getattr(self.config, func_name)(*args, **kwargs)
    
    return wrapper

def get_config(config_file):
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith("_") and func_name != "read":
            setattr(ConfigParserHook, func_name, set_hook(func_name))
    setattr(ConfigParserHook, "__getitem__", set_hook("__getitem__"))
    
    config = ConfigParserHook()
    config.read(config_file)

    return config