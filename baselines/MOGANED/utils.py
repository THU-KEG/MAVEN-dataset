import os
import constant
from xml.dom.minidom import parse
from tqdm import tqdm
import re
import random
import json
import numpy as np
import copy
from stanfordcorenlp import StanfordCoreNLP

class Extractor():
    def __init__(self):
        pass

    def preprocess(self):
        splits = {'train':'train','valid':'dev','test':'test'}
        path = constant.maven_path
        nlp = StanfordCoreNLP(constant.corenlp_path)
        mention_ids = []
        for split in tqdm(splits):
            split_data = []
            with open(path+'/'+split+'.jsonl','r') as f:
                line = f.readline().rstrip()
                while line:
                    doc = json.loads(line)
                    content = doc['content']
                    for sent_tuple in content:
                        origin_sent,origin_tokens = sent_tuple['sentence'],sent_tuple['tokens']

                        parse_sent = ' '.join(sent_tuple['tokens'])
                        nlp_words,nlp_span = nlp.word_tokenize(parse_sent,True)
                        nlp_span_dict = {e[0]:i for i,e in enumerate(nlp_span)}
                        origin_span = {i:len(' '.join(origin_tokens[:i]))+1 for i in range(1,len(origin_tokens))}
                        origin_span[0] = 0
                        sent_tuple['origin_span'] = origin_span
                        sent_tuple['nlp_span_dict'] = nlp_span_dict
                        sent_tuple['nlp_words'] = nlp_words
                        
                        dependency_parsing =nlp.dependency_parse(parse_sent)
                        pos_tags = [e[1] for e in nlp.pos_tag(parse_sent)]
                        ner_tags = [e[1] for e in nlp.ner(parse_sent)]
                        sent_tuple['ner'] = ner_tags
                        sent_tuple['pos'] = pos_tags
                        sent_tuple['dependency'] = dependency_parsing
                    if split!='test':
                        for event in doc['events']:
                            event_type = event['type_id']
                            assert isinstance(event_type,int)
                            assert event_type<169
                            # if event_type==207:
                                # continue
                            for mention in event['mention']:
                                trigger = mention['trigger_word'].lower()
                                offset = mention['offset']
                                tokens = content[mention['sent_id']]['tokens']
                                
                                origin_span = content[mention['sent_id']]['origin_span']
                                nlp_span_dict = content[mention['sent_id']]['nlp_span_dict']
                                nlp_words = content[mention['sent_id']]['nlp_words']
                                if origin_span[offset[0]] not in nlp_span_dict:
                                    real_offset = offset[0]
                                else:
                                    real_offset = nlp_span_dict[origin_span[offset[0]]]

                                mention_ids.append((mention['id'],event_type))

                                info = {'tokens':nlp_words,
                                        'trigger_tokens':[nlp_words[real_offset]],
                                        'ner_tags':content[mention['sent_id']]['ner'],
                                        'pos_tags':content[mention['sent_id']]['pos'],
                                        'dependency_parsing':content[mention['sent_id']]['dependency'],
                                        'trigger_start':real_offset,
                                        'trigger_end':real_offset,
                                        'event_type':event_type}
                                split_data.append(info)
                    negative_triggers = 'negative_triggers'
                    if split=='test':
                        negative_triggers = 'candidates'
                    for mention in doc[negative_triggers]:
                        trigger = mention['trigger_word'].lower()
                        offset = mention['offset']
                        tokens = content[mention['sent_id']]['tokens']
                        mention_ids.append((mention['id'],0))
                        origin_span = content[mention['sent_id']]['origin_span']
                        nlp_span_dict = content[mention['sent_id']]['nlp_span_dict']
                        nlp_words = content[mention['sent_id']]['nlp_words']
                        if origin_span[offset[0]] not in nlp_span_dict:
                            real_offset = offset[0]
                        else:
                            real_offset = nlp_span_dict[origin_span[offset[0]]]

                        info = {'tokens':nlp_words,
                                'trigger_tokens':[nlp_words[real_offset]],
                                'ner_tags':content[mention['sent_id']]['ner'],
                                'pos_tags':content[mention['sent_id']]['pos'],
                                'dependency_parsing':content[mention['sent_id']]['dependency'],
                                'trigger_start':real_offset,
                                'trigger_end':real_offset,
                                'event_type':0}
                        split_data.append(info)

                    line = f.readline().rstrip()
            with open(path+'/'+splits[split]+'.json','w') as f:
                json.dump(split_data,f)

        nlp.close()
        
    def id_align(self):
        ids = []

        with open('{}/test.jsonl'.format(constant.maven_path),'r') as f:
            line = f.readline().rstrip()
            while line:
                doc = json.loads(line)
                doc_id = doc['id']
                for mention in doc['candidates']:
                    trigger_id = mention['id']
                    ids.append((doc_id,trigger_id))
                line = f.readline().rstrip()

        with open('{}/id_align.json'.format(constant.maven_path),'w') as f:
            json.dump(ids,f)
    
    def extract(self):
        if not os.path.exists(constant.maven_path+'/train.json'):
            print('----Preprocessing----')
            self.preprocess()
        else:
            print("--Preprocessed files exist--")
        if not os.path.exists(constant.maven_path+'/id_align.json'):
            print('----Id Aligning----')
            self.id_align()

class Loader():
    def __init__(self,cut_len):
        self.train_path = constant.maven_path+'/train.json'
        self.dev_path = constant.maven_path+'/dev.json'
        self.test_path = constant.maven_path+'/test.json'
        self.glove_path = constant.GloVe_file
        self.cut_len = cut_len

    def load_embedding(self):
        word2idx = {}
        wordemb = []
        with open(self.glove_path,'r',encoding='utf-8') as f:
            for line in f:
                splt = line.split()
                assert len(splt)==constant.embedding_dim+1
                vector = list(map(float, splt[-constant.embedding_dim:]))
                word = splt[0]
                word2idx[word] = len(word2idx)+2
                wordemb.append(vector)
        return word2idx,np.asarray(wordemb,np.float32)

    def get_maxlen(self):
        if self.cut_len!=None:
            self.maxlen = self.cut_len
            return self.maxlen
        paths = [self.train_path,self.dev_path,self.test_path]
        maxlens = []
        for path in paths:
            with open(path,'r') as f:
                data = json.load(f)
            _maxlen = max([len(d['tokens']) for d in data])
            maxlens.append(_maxlen)
        self.maxlen = max(maxlens)
        return self.maxlen
    
    def get_max_argument_len(self):
        paths = [self.train_path,self.dev_path,self.test_path]
        maxlens = []
        for path in paths:
            with open(path,'r') as f:
                data = json.load(f)
            for instance in data:
                if len(instance['entities'])==0:
                    continue
                _maxlen = max([entity['idx_end']+1-entity['idx_start'] for entity in instance['entities']])
                maxlens.append(_maxlen)
        self.max_argument_len = max(maxlens)
        return self.max_argument_len

    def get_positions(self,start_idx,sent_len,maxlen):
        return list(range(maxlen-start_idx, maxlen)) + [maxlen]  + \
               list(range(maxlen+1, maxlen+sent_len - start_idx))+[0]*(maxlen-sent_len)

    def get_word(self,tokens,word2idx,pad_lenth):
        idx = []
        for word in tokens:
            if word.lower() in word2idx:
                idx.append(word2idx[word.lower()])
            else:
                idx.append(1)
        idx += [0]*(pad_lenth-len(idx))
        return idx

    def get_trigger_mask(self,posi,sent_len,maxlen,direction):
        assert direction in ['left','right']
        mask = [0.]*maxlen
        if direction=='left':
            mask[:posi] = [1.]*posi
        else:
            mask[posi:sent_len] = [1.]*(sent_len-posi)
        return mask

    def load_one_trigger(self,path,maxlen,word2idx):
        trigger_posis,sents,trigger_maskls,trigger_maskrs,event_types,trigger_lexical= [], [], [], [], [], []
        with open(path,'r') as f:
            data = json.load(f)
        
        indices_s,pos,ner = [],[],[]
        trigger_idxs = []

        test_idxs = []
        

        for test_idx,instance in enumerate(data):
            tokens = instance['tokens'][:maxlen]
            event_type = instance['event_type']
            trigger_posi = instance['trigger_start']
            if trigger_posi>maxlen-1:
                continue
            ner_tags = [constant.NER_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['ner_tags']][:maxlen]+[0]*(maxlen-len(instance['ner_tags']))
            pos_tags = [constant.POS_TO_ID[e] if e in constant.NER_TO_ID else 1 for e in instance['pos_tags']][:maxlen]+[0]*(maxlen-len(instance['pos_tags']))
            ner.append(ner_tags)
            pos.append(pos_tags)

            words = self.get_word(tokens,word2idx,maxlen)
            dependency_parsing = instance['dependency_parsing']

            start_word = 0
            current_max = 0
            indices = []
            for edge in dependency_parsing:
                if edge[0]=="ROOT":
                    start_word = max(start_word,current_max)
                else:
                    if edge[1]-1+start_word>maxlen-1 or edge[2]-1+start_word>maxlen-1:
                        continue
                    indices.append([edge[1]-1+start_word,edge[2]-1+start_word])
                current_max = max([current_max,edge[1]+start_word,edge[2]+start_word])
            indices_s.append(indices)

            trigger_posis.append(self.get_positions(trigger_posi,len(tokens),maxlen))
            trigger_idxs.append(trigger_posi)
            sents.append(words)
            trigger_maskls.append(self.get_trigger_mask(trigger_posi,len(tokens),maxlen,'left'))
            trigger_maskrs.append(self.get_trigger_mask(trigger_posi, len(tokens),maxlen, 'right'))
            event_types.append(constant.EVENT_TYPE_TO_ID[event_type])

            _trigger_lexical = []
            if trigger_posi==0:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi-1])

            _trigger_lexical.append(words[trigger_posi])

            if trigger_posi==len(tokens)-1:
                _trigger_lexical.append(0)
            else:
                _trigger_lexical.append(words[trigger_posi+1])

            trigger_lexical.append(_trigger_lexical)
            test_idxs.append(test_idx)
        if path.endswith('test.json'):
            with open('test_idxs.json','w') as f:
                json.dump(test_idxs,f)
        return (np.array(trigger_posis,np.int32),np.array(sents,np.int32),np.array(trigger_maskls,np.int32),\
               np.array(trigger_maskrs,np.int32),np.array(event_types,np.int32),np.array(trigger_lexical,np.int32),\
               np.array(pos,np.int32),np.array(ner,np.int32),np.array(trigger_idxs,np.int32)),indices_s

    def load_trigger(self):
        print('--Loading Trigger--')
        word2idx,self.wordemb = self.load_embedding()
        maxlen = self.get_maxlen()
        paths = [self.train_path, self.dev_path, self.test_path]
        results = []
        for path in paths:
            result = self.load_one_trigger(path,maxlen,word2idx)
            results.append(result)
        return results