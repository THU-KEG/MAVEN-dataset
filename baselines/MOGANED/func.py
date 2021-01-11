import numpy as np
import random
import tensorflow as tf
import constant
from tqdm import tqdm

class Cudnn_RNN:

    def __init__(self, num_layers, num_units, mode="lstm",keep_prob=1.0, is_train=None, scope="cudnn_rnn", gpu=True):
        self.num_layers = num_layers
        self.rnns = []
        self.mode = mode
        if mode == "gru":
            if gpu:
                rnn = tf.contrib.cudnn_rnn.CudnnGRU
            else:
                rnn = tf.contrib.rnn.GRUCell
        elif mode == "lstm":
            if gpu:
                rnn = tf.contrib.cudnn_rnn.CudnnLSTM
            else:
                rnn = tf.contrib.rnn.BasicLSTM
        else:
            raise Exception("Unknown mode for rnn")
        for layer in range(num_layers):
            if gpu:
                rnn_fw = rnn(1, num_units)
                rnn_bw = rnn(1, num_units)
            else:
                rnn_fw = rnn(num_units)
                rnn_bw = rnn(num_units)
            self.rnns.append((rnn_fw, rnn_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            rnn_fw, rnn_bw = self.rnns[layer]
            output = dropout(outputs[-1], keep_prob=keep_prob, is_train=is_train)
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, state_fw = rnn_fw(output)
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(output, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                out_bw, state_bw = rnn_bw(inputs_bw)
                out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers is True:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        state_fw = tf.squeeze(state_fw[0], [0])
        state_bw = tf.squeeze(state_bw[0], [0])
        state = tf.concat([state_fw, state_bw], axis=1)
        return res, state

def dropout(args, keep_prob, is_train, mode=None):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], shape[1], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape), lambda: args)
    return args

def f_score(predict,golden,mode='f'):
    assert len(predict)==len(golden)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predict)):
        if predict[i]==golden[i] and predict[i] != 0:
            TP+=1
        elif predict[i]!=golden[i]:
            if predict[i]==0:
                FN+=1
            elif golden[i]==0:
                FP+=1
            else:
                FN+=1
                FP+=1
        else:
            TN+=1
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F = 2*P*R/(P+R)
    except:
        P=R=F=0

    if mode=='f':
        return P,R,F
    else:
        return TP,FN,FP,TN

def get_batch(data_all,batch_size,shuffle=True):
    data,data_subg = data_all
    assert len(list(set([np.shape(d)[0] for d in data]))) == 1
    num_data = np.shape(data[0])[0]
    indices = list(np.arange(0,num_data))
    if shuffle:
        random.shuffle(indices)
    for i in tqdm(range((num_data // batch_size)+1)):
        select_indices = indices[i*batch_size:(i+1)*batch_size]
        select_subg_indices = [[idx]+indice for idx,select_indice in enumerate(select_indices) for indice in data_subg[select_indice]]
        yield [np.take(d,select_indices,axis=0) for d in data]+[select_subg_indices]

def get_trigger_feeddict(model,batch,stage,maxlen,is_train=True):
    if stage=='DMCNN':
        posis,sents,maskls,maskrs,event_types,lexical,_,_,_,_ = batch
        return {model.posis:posis,model.sents:sents,model.maskls:maskls,model.maskrs:maskrs,
                model._labels:event_types,model.lexical:lexical,model.is_train:is_train}
    else:
        posis,sents,maskls,maskrs,event_types,lexical,pos,ner,trigger_idxs,subg_indices = batch
        subg_vals = [1.0]*len(subg_indices)
        subg_shape = [sents.shape[0],maxlen,maxlen]
        subg = (subg_indices,subg_vals,subg_shape)

        gather_idxs = np.stack([np.array(np.arange(posis.shape[0])),trigger_idxs],axis=1)
        return {model.posis:posis,model.sents:sents,model.maskls:maskls,model.maskrs:maskrs,
                model._labels:event_types,model.lexical:lexical,model.is_train:is_train,
                model.pos_idx:pos,model.ner_idx:ner,model.subg_a:subg,model.gather_idxs:gather_idxs}


#GAT util function

def u_compute(ps,subg,maxlen):
    with tf.variable_scope("e_compute",reuse=tf.AUTO_REUSE):
        att = tf.layers.dense(ps,constant.Watt_dim,name='Watt')
        left_comb = tf.layers.dense(att,1,name='comb_left')
        right_comb = tf.layers.dense(att,1,name='comb_right')

        tile_left = tf.tile(left_comb,[1,1,maxlen],name='tile_1')
        tile_right = tf.tile(tf.transpose(left_comb,[0,2,1],name='transpose_1'),[1,maxlen,1],name='tile_2')
        tiles_concat = tile_left+tile_right

        e_mat = tf.nn.leaky_relu(tiles_concat,alpha=constant.leaky_alpha,name='lrelu_1')
    with tf.variable_scope('u_compute',reuse=tf.AUTO_REUSE):
        u_raw = tf.multiply(e_mat,subg,name='mul_1')-(1-subg)*1e8
        u_mat = tf.nn.softmax(u_raw,axis=2,name='soft_1')
    return u_mat

def GAC_func(ps,subg,maxlen,a,k):
    with tf.variable_scope("GAC_compute",reuse=tf.AUTO_REUSE):
        u_mat = u_compute(ps,subg,maxlen)
        weight_name = a+'_'+str(k)
        dense = tf.layers.dense(ps,constant.graph_dim,name=weight_name)
        # dense_expand = tf.tile(tf.expand_dims(dense,2,name='expand_1'),[1,1,maxlen,1],name='tile_1')
        # u_mat_expand = tf.tile(tf.expand_dims(u_mat,3,name='expand_2'),[1,1,1,constant.graph_dim],name='tile_2')
        dense_expand = tf.expand_dims(dense,2,name='expand_1')
        u_mat_expand = tf.expand_dims(u_mat,3,name='expand_2')
        sums = tf.reduce_sum(tf.multiply(u_mat_expand,dense_expand,name='mul1'),axis=2,name='sum_1')
        graph_emb = tf.nn.elu(sums,name='elu_1')
    return graph_emb
        
def matmuls(a,times):
    with tf.variable_scope('matmuls_'):
        res = a
        for i in range(times-1):
            res = tf.matmul(res,a)
    return res