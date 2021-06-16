# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Event detection CRF finetuning: utilities to work with ACE 2005 """

from __future__ import absolute_import, division, print_function
import json
import logging
import os
from io import open
from transformers import XLMRobertaTokenizer, BertTokenizer, RobertaTokenizer

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    data=json.load(open(file_path, "r"))
    words=[]
    labels=[]
    def getLabel(eT,is_start=False):
        if eT=='None':
            return 'O'
        if is_start:
            return "B-"+eT
        return "I-"+eT
    words=data[0]['tokens']
    labels=['X' for i in range(0,len(words))]
    labels[data[0]['trigger_start']]=getLabel(data[0]['event_type'],True)
    for j in range(data[0]['trigger_start']+1,data[0]['trigger_end']+1):
        labels[j]=getLabel(data[0]['event_type'])
    for i in range(1,len(data)):
        if data[i]['tokens']!=data[i-1]['tokens']:
            examples.append(InputExample(guid="%s-%d"%(mode,i),
                                        words=words,
                                        labels=labels))
            words=data[i]['tokens']
            labels=['X' for i in range(0,len(words))]
        labels[data[i]['trigger_start']]=getLabel(data[i]['event_type'],True)
        for j in range(data[i]['trigger_start']+1,data[i]['trigger_end']+1):
            labels[j]=getLabel(data[i]['event_type'])
    examples.append(InputExample(guid="%s-%d"%(mode,len(data)),words=words,labels=labels))
    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-100,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 model_name=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    # my logic in crf_padding requires this check. I create mask for crf by labels==pad_token_label_id to not include it
    # in loss and decoding
    assert pad_token_label_id not in label_map.values()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("###############")
            logger.info("Writing example %d of %d", ex_index, len(examples))
            print("###############")

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label!='X':
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            else:
                label_ids.extend([pad_token_label_id] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]  # [label_map["X"]]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        if model_name:
            if model_name == 'xlm-roberta-base':
                tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name.startswith('bert'):
                tokenizer = BertTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
    return features


def get_labels(path):
    return ["O", "B-Attack", "I-Attack", "B-Transport", "I-Transport", "B-Die", "I-Die", "B-End-Position", "I-End-Position", "B-Meet", "I-Meet", "B-Phone-Write", "I-Phone-Write", "B-Elect", "I-Elect", "B-Injure", "I-Injure", "B-Transfer-Ownership", "I-Transfer-Ownership", "B-Start-Org", "I-Start-Org", "B-Transfer-Money", "I-Transfer-Money", "B-Sue", "I-Sue", "B-Demonstrate", "I-Demonstrate", "B-Arrest-Jail", "I-Arrest-Jail", "B-Start-Position", "I-Start-Position", "B-Be-Born", "I-Be-Born", "B-End-Org", "I-End-Org", "B-Execute", "I-Execute", "B-Nominate", "I-Nominate", "B-Fine", "I-Fine", "B-Trial-Hearing", "I-Trial-Hearing", "B-Marry", "I-Marry", "B-Charge-Indict", "I-Charge-Indict", "B-Sentence", "I-Sentence", "B-Convict", "I-Convict", "B-Appeal", "I-Appeal", "B-Declare-Bankruptcy", "I-Declare-Bankruptcy", "B-Merge-Org", "I-Merge-Org", "B-Release-Parole", "I-Release-Parole", "B-Pardon", "I-Pardon", "B-Extradite", "I-Extradite", "B-Divorce", "I-Divorce", "B-Acquit", "I-Acquit"]

def to_crf_pad(org_array, org_mask, pad_label_id):
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    crf_pad = (crf_array != pad_label_id)
    # the viterbi decoder function in CRF makes use of multiplicative property of 0, then pads wrong numbers out.
    # Need a*0 = 0 for CRF to work.
    crf_array[~crf_pad] = 0
    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    return out_array