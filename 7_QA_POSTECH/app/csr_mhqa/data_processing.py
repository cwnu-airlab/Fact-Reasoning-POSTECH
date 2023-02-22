import gzip
import pickle
import json
import torch
import numpy as np
import os

from os.path import join
from tqdm import tqdm
from numpy.random import shuffle

from envs import DATASET_FOLDER

IGNORE_INDEX = -100

def get_cached_filename(f_type, config):
    assert f_type in ['examples', 'features']

    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 sent_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 ctx_text,
                 ctx_word_to_char_idx,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sent_start_end_position = sent_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 sent_token_idx,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids
        self.sent_spans = sent_spans

        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type

        self.token_to_orig_map = token_to_orig_map
        self.sent_token_idx = sent_token_idx
        self.orig_answer_text = orig_answer_text

        self.start_position = start_position
        self.end_position = end_position


class DataIteratorPack(object):
    def __init__(self,
                 features, example_dict,
                 bsz, device,
                 sent_limit,
                 sequential=False):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        self.example_ptr = 0
        self.max_seq_length = 512
        if not sequential:
            shuffle(self.features)

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, self.max_seq_length)
        context_mask = torch.LongTensor(self.bsz, self.max_seq_length)
        segment_idxs = torch.LongTensor(self.bsz, self.max_seq_length)

        # Mapping
        query_mapping = torch.Tensor(self.bsz, self.max_seq_length).cuda(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids=[]
            is_support.fill_(IGNORE_INDEX)

            sent_token_idx = []

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                if len(case.sent_spans) > 0:
                    for j in range(case.sent_spans[0][0] - 1):
                        query_mapping[i, j] = 1

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start <= end:
                        end = min(end, self.max_seq_length-1)
                        is_support[i, j] = int(is_sp_flag)

                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < self.max_seq_length and context_mask[i][case.end_position[0]+1] == 1: # "[SEP]" is the last token
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2

                ids.append(case.qas_id)
                sent_token_idx.append(case.sent_token_idx)

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous().to(self.device),
                'context_lens': input_lengths.contiguous().to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'sent_token_idx': sent_token_idx
            }

class DataHelper:
    def __init__(self, gz=True, config=None):
        self.DataIterator = DataIteratorPack
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')

        self.__train_features__ = None
        self.__dev_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config

    def get_feature_file(self, tag):
        cached_filename = get_cached_filename('features', self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag):
        cached_filename = get_cached_filename('examples', self.config)

        return join(self.data_dir, tag, cached_filename)

    @property
    def train_feature_file(self):
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        return self.get_feature_file('dev_distractor')

    @property
    def train_example_file(self):
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor')

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Features
    @property
    def train_features(self):
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Feature dict
    @property
    def train_feature_dict(self):
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict

    def load_train(self):
        return self.train_features, self.train_example_dict

    @property
    def dev_loader(self):
        return self.DataIterator(*self.load_dev(),
                                 bsz=self.config.eval_batch_size,
                                 device=self.config.device,
                                 sent_limit=self.config.max_sent_num,
                                 sequential=True)

    @property
    def train_loader(self):
        return self.DataIterator(*self.load_train(),
                                 bsz=self.config.batch_size,
                                 device=self.config.device,
                                 sent_limit=self.config.max_sent_num,
                                 sequential=False)
