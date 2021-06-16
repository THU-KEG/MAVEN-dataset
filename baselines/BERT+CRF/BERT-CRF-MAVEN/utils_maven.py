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
""" BERT-CRF fine-tuning: utilities to work with MAVEN. """

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
    file_path = os.path.join(data_dir, "{}.jsonl".format(mode))
    examples = []
    with open(file_path, "r") as fin:
        lines=fin.readlines()
        for line in lines:
            doc=json.loads(line)
            words=[]
            labels=[]
            for sent in doc['content']:
                words.append(sent['tokens'])
                labels.append(['O' for i in range(0,len(sent['tokens']))])#TBD
            if mode!='test':
                for event in doc['events']:
                    for mention in event['mention']:
                        labels[mention['sent_id']][mention['offset'][0]]="B-"+event['type']
                        for i in range(mention['offset'][0]+1,mention['offset'][1]):
                            labels[mention['sent_id']][i]="I-"+event['type']
                for mention in doc['negative_triggers']:
                    labels[mention['sent_id']][mention['offset'][0]]="O"
                    for i in range(mention['offset'][0]+1,mention['offset'][1]):
                        labels[mention['sent_id']][i]="O"
            for i in range(0,len(words)):
                examples.append(InputExample(guid="%s-%s-%d"%(mode,doc['id'],i),
                                            words=words[i],
                                            labels=labels[i]))
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
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

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
    return ["O", "B-Know", "I-Know", "B-Warning", "I-Warning", "B-Catastrophe", "I-Catastrophe", "B-Placing", "I-Placing", "B-Causation", "I-Causation", "B-Arriving", "I-Arriving", "B-Sending", "I-Sending", "B-Protest", "I-Protest", "B-Preventing_or_letting", "I-Preventing_or_letting", "B-Motion", "I-Motion", "B-Damaging", "I-Damaging", "B-Destroying", "I-Destroying", "B-Death", "I-Death", "B-Perception_active", "I-Perception_active", "B-Presence", "I-Presence", "B-Influence", "I-Influence", "B-Receiving", "I-Receiving", "B-Check", "I-Check", "B-Hostile_encounter", "I-Hostile_encounter", "B-Killing", "I-Killing", "B-Conquering", "I-Conquering", "B-Releasing", "I-Releasing", "B-Attack", "I-Attack", "B-Earnings_and_losses", "I-Earnings_and_losses", "B-Choosing", "I-Choosing", "B-Traveling", "I-Traveling", "B-Recovering", "I-Recovering", "B-Using", "I-Using", "B-Coming_to_be", "I-Coming_to_be", "B-Cause_to_be_included", "I-Cause_to_be_included", "B-Process_start", "I-Process_start", "B-Change_event_time", "I-Change_event_time", "B-Reporting", "I-Reporting", "B-Bodily_harm", "I-Bodily_harm", "B-Suspicion", "I-Suspicion", "B-Statement", "I-Statement", "B-Cause_change_of_position_on_a_scale", "I-Cause_change_of_position_on_a_scale", "B-Coming_to_believe", "I-Coming_to_believe", "B-Expressing_publicly", "I-Expressing_publicly", "B-Request", "I-Request", "B-Control", "I-Control", "B-Supporting", "I-Supporting", "B-Defending", "I-Defending", "B-Building", "I-Building", "B-Military_operation", "I-Military_operation", "B-Self_motion", "I-Self_motion", "B-GetReady", "I-GetReady", "B-Forming_relationships", "I-Forming_relationships", "B-Becoming_a_member", "I-Becoming_a_member", "B-Action", "I-Action", "B-Removing", "I-Removing", "B-Surrendering", "I-Surrendering", "B-Agree_or_refuse_to_act", "I-Agree_or_refuse_to_act", "B-Participation", "I-Participation", "B-Deciding", "I-Deciding", "B-Education_teaching", "I-Education_teaching", "B-Emptying", "I-Emptying", "B-Getting", "I-Getting", "B-Besieging", "I-Besieging", "B-Creating", "I-Creating", "B-Process_end", "I-Process_end", "B-Body_movement", "I-Body_movement", "B-Expansion", "I-Expansion", "B-Telling", "I-Telling", "B-Change", "I-Change", "B-Legal_rulings", "I-Legal_rulings", "B-Bearing_arms", "I-Bearing_arms", "B-Giving", "I-Giving", "B-Name_conferral", "I-Name_conferral", "B-Arranging", "I-Arranging", "B-Use_firearm", "I-Use_firearm", "B-Committing_crime", "I-Committing_crime", "B-Assistance", "I-Assistance", "B-Surrounding", "I-Surrounding", "B-Quarreling", "I-Quarreling", "B-Expend_resource", "I-Expend_resource", "B-Motion_directional", "I-Motion_directional", "B-Bringing", "I-Bringing", "B-Communication", "I-Communication", "B-Containing", "I-Containing", "B-Manufacturing", "I-Manufacturing", "B-Social_event", "I-Social_event", "B-Robbery", "I-Robbery", "B-Competition", "I-Competition", "B-Writing", "I-Writing", "B-Rescuing", "I-Rescuing", "B-Judgment_communication", "I-Judgment_communication", "B-Change_tool", "I-Change_tool", "B-Hold", "I-Hold", "B-Being_in_operation", "I-Being_in_operation", "B-Recording", "I-Recording", "B-Carry_goods", "I-Carry_goods", "B-Cost", "I-Cost", "B-Departing", "I-Departing", "B-GiveUp", "I-GiveUp", "B-Change_of_leadership", "I-Change_of_leadership", "B-Escaping", "I-Escaping", "B-Aiming", "I-Aiming", "B-Hindering", "I-Hindering", "B-Preserving", "I-Preserving", "B-Create_artwork", "I-Create_artwork", "B-Openness", "I-Openness", "B-Connect", "I-Connect", "B-Reveal_secret", "I-Reveal_secret", "B-Response", "I-Response", "B-Scrutiny", "I-Scrutiny", "B-Lighting", "I-Lighting", "B-Criminal_investigation", "I-Criminal_investigation", "B-Hiding_objects", "I-Hiding_objects", "B-Confronting_problem", "I-Confronting_problem", "B-Renting", "I-Renting", "B-Breathing", "I-Breathing", "B-Patrolling", "I-Patrolling", "B-Arrest", "I-Arrest", "B-Convincing", "I-Convincing", "B-Commerce_sell", "I-Commerce_sell", "B-Cure", "I-Cure", "B-Temporary_stay", "I-Temporary_stay", "B-Dispersal", "I-Dispersal", "B-Collaboration", "I-Collaboration", "B-Extradition", "I-Extradition", "B-Change_sentiment", "I-Change_sentiment", "B-Commitment", "I-Commitment", "B-Commerce_pay", "I-Commerce_pay", "B-Filling", "I-Filling", "B-Becoming", "I-Becoming", "B-Achieve", "I-Achieve", "B-Practice", "I-Practice", "B-Cause_change_of_strength", "I-Cause_change_of_strength", "B-Supply", "I-Supply", "B-Cause_to_amalgamate", "I-Cause_to_amalgamate", "B-Scouring", "I-Scouring", "B-Violence", "I-Violence", "B-Reforming_a_system", "I-Reforming_a_system", "B-Come_together", "I-Come_together", "B-Wearing", "I-Wearing", "B-Cause_to_make_progress", "I-Cause_to_make_progress", "B-Legality", "I-Legality", "B-Employment", "I-Employment", "B-Rite", "I-Rite", "B-Publishing", "I-Publishing", "B-Adducing", "I-Adducing", "B-Exchange", "I-Exchange", "B-Ratification", "I-Ratification", "B-Sign_agreement", "I-Sign_agreement", "B-Commerce_buy", "I-Commerce_buy", "B-Imposing_obligation", "I-Imposing_obligation", "B-Rewards_and_punishments", "I-Rewards_and_punishments", "B-Institutionalization", "I-Institutionalization", "B-Testing", "I-Testing", "B-Ingestion", "I-Ingestion", "B-Labeling", "I-Labeling", "B-Kidnapping", "I-Kidnapping", "B-Submitting_documents", "I-Submitting_documents", "B-Prison", "I-Prison", "B-Justifying", "I-Justifying", "B-Emergency", "I-Emergency", "B-Terrorism", "I-Terrorism", "B-Vocalizations", "I-Vocalizations", "B-Risk", "I-Risk", "B-Resolve_problem", "I-Resolve_problem", "B-Revenge", "I-Revenge", "B-Limiting", "I-Limiting", "B-Research", "I-Research", "B-Having_or_lacking_access", "I-Having_or_lacking_access", "B-Theft", "I-Theft", "B-Incident", "I-Incident", "B-Award", "I-Award"]


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