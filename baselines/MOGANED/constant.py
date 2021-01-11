maven_path = '../GAT/data/newdataset'
GloVe_file = '../GAT/glove/glove.6B.100d.txt'
corenlp_path = '../GAT/stanford-corenlp-full-2018-10-05'

EVENT_TYPE_TO_ID = {i:i for i in range(169)}
ROLE_TO_ID = {'None': 0, 'Person': 1, 'Place': 2, 'Buyer': 3, 'Seller': 4, 'Beneficiary': 5, 'Price': 6, 'Artifact': 7, 'Origin': 8, 'Destination': 9, 'Giver': 10, 'Recipient': 11, 'Money': 12, 'Org': 13, 'Agent': 14, 'Victim': 15, 'Instrument': 16, 'Entity': 17, 'Attacker': 18, 'Target': 19, 'Defendant': 20, 'Adjudicator': 21, 'Prosecutor': 22, 'Plaintiff': 23, 'Crime': 24, 'Position': 25, 'Sentence': 26, 'Vehicle': 27, 'Time-Within': 28, 'Time-Starting': 29, 'Time-Ending': 30, 'Time-Before': 31, 'Time-After': 32, 'Time-Holds': 33, 'Time-At-Beginning': 34, 'Time-At-End': 35}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6,
             'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22,
             '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}



INF = 1e8

#general hyperparams
embedding_dim = 100
posi_embedding_dim = 50
event_type_embedding_dim = 5
cut_len = 50                      #set None to not cut length


#trigger hyperparameters
t_filters = 200
t_batch_size = 30
t_lr = 0.001
t_epoch = 10
t_keepprob = 0.7
t_bias_lambda = 1

#GAT hypers
pos_dim = 50
ner_dim = 50
hidden_dim  = 100

Watt_dim = 100
s_dim = 100

leaky_alpha = 0.2
graph_dim = 150

K=3

