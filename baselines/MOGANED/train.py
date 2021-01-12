import tensorflow as tf
import utils
from models import Trigger_Model
import os
from constant import *

flags = tf.flags
flags.DEFINE_string("gpu", "1", "The GPU to run on")
flags.DEFINE_string("mode", "MOGANED", "DMCNN or MOGANED")
flags.DEFINE_bool('eval', False, "Eval or Train")

def main(_):
    config = flags.FLAGS
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    extractor = utils.Extractor()
    extractor.extract()
    loader = utils.Loader(cut_len)
    t_data = loader.load_trigger()
    trigger = Trigger_Model(t_data,loader.maxlen,loader.wordemb,config.mode)
    if not config.eval:
        trigger.train_trigger()
    else:
        trigger.eval_trigger()

if __name__=="__main__":
    tf.app.run()