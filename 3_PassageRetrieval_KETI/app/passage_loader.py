# Copyright 2021 san kim
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

import os
import bz2
import abc
import json
import copy
import time
import logging
from typing import List

import numpy as np
import torch
import faiss

from data_utils import collate_tokens

def read_bz2(bz2_path):
    with bz2.open(bz2_path) as fp:
        return [json.loads(line) for line in fp.readlines()]

def get_line(bz2_path, line_idx):
    with bz2.open(bz2_path) as fp:
        for i, line in enumerate(fp):
            if i == line_idx:
                return line
            elif i > line_idx:
                break
    return None

def get_line_using_title(bz2_path, qtitle, is_lower=True):
    if is_lower:
        qtitle = qtitle.lower()
    with bz2.open(bz2_path) as fp:
        for i, line in enumerate(fp):
            doc = json.loads(line)
            title = doc['title']
            if is_lower:
                title = title.lower()
            if title == qtitle:
                return doc

    return None

def get_line_using_title_memory(doc_list, qtitle, is_lower=True, start_idx=0, search_len=10000):
    if is_lower:
        qtitle = qtitle.lower()

    for i, doc in enumerate(doc_list[start_idx:start_idx+search_len]):
        title = doc['title']
        if is_lower:
            title = title.lower()
        if title == qtitle:
            return copy.deepcopy(doc)

    return None

def get_doc_id_using_title(bz2_path, qtitle, is_lower=True):
    if is_lower:
        qtitle = qtitle.lower()
    with bz2.open(bz2_path) as fp:
        for i, line in enumerate(fp):
            doc = json.loads(line)
            title = doc['title']
            if is_lower:
                title = title.lower()
            if title == qtitle:
                return i

    return None

def get_doc_id_using_title_memory(doc_list, qtitle, is_lower=True, start_idx=0, search_len=10000):
    if is_lower:
        qtitle = qtitle.lower()
    for i, doc in enumerate(doc_list[start_idx:start_idx+search_len]):
        title = doc['title']
        if is_lower:
            title = title.lower()
        if title == qtitle:
            return i

    return None

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def get_fvec_using_doc_id(fname, index):
    fvecs = fvecs_read(fname)
    size = fvecs.shape[0]
    if index >= size:
        return None
    else:
        return fvecs[index]


class _DefaultDocumentLoader(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def retrieve_doc(self, doc_idx):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def retrieve_docs(self, doc_idx_list):
        raise NotImplementedError


class DocumentLoader(_DefaultDocumentLoader):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(
        self,
        page_info_path) -> None:

        with open(page_info_path, "r") as f:
            self.page_info = json.load(f)
        
        self.data_root = os.path.dirname(page_info_path)
        self.page_table = self.page_info.get('page_table', None)
        self.page_index = self.page_info.get('page_index', None)
        self.is_lower = self.page_info.get('is_lower', False)
        self.max_page_per_shard = self.page_info.get('max_page_per_shard', None)
        self.num_pages = self.page_info.get('num_pages', None)
        self.rel_page_root = self.page_info.get('rel_page_root', None)
        assert self.page_table is not None, "can't find page_table from page_info"
        assert self.max_page_per_shard is not None, "can't find max_page_per_shard from page_info"
        assert self.num_pages is not None, "can't find num_pages from page_info"
        assert self.rel_page_root is not None, "can't find rel_page_root from page_info"

        self.page_root = os.path.join(self.data_root, self.rel_page_root)

    def retrieve_doc(self, doc_idx, passage_base=True):
        if doc_idx > self.num_pages:
            return None
        shard_id, line_id = divmod(doc_idx, self.max_page_per_shard)
        doc = json.loads(get_line(os.path.join(self.data_root, self.rel_page_root, self.page_table[shard_id]), line_id))
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc

    def retrieve_docs(self, doc_idx_list, passage_base=True):
        return [self.retrieve_doc(x, passage_base) for x in doc_idx_list]
    
    def retrieve_docs_batch(self, batch, passage_base=True):
        return [self.retrieve_docs(x, passage_base) for x in batch]
    
    def _find_page_index(self, title):
        if self.is_lower:
            title = title.lower()

        for idx, pindex in enumerate(self.page_index):
            if pindex[0] <= title and title <= pindex[1]:
                return idx
        return -1
    
    def _retrieve_doc_using_title_with_index(self, title, passage_base=True):
        pidx = self._find_page_index(title)
        doc = None
        if pidx > -1:
            fname = self.page_table[pidx]

            doc = get_line_using_title(os.path.join(self.page_root, fname), title, self.is_lower)
        
        if doc is None:
            doc = {
                'title': 'NA',
                'text': ['NA']
            }
        
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc
    
    def _retrieve_doc_using_title(self, title, passage_base=True):
        doc = None
        for fname in self.page_table:
            doc = get_line_using_title(os.path.join(self.page_root, fname), title, self.is_lower)
        
        if doc is  None:
            doc = {
                'title': 'NA',
                'text': 'NA'
            }
        
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc
    
    def retrieve_doc_using_title(self, title, passage_base=True):
        if self.page_index is not None:
            return self._retrieve_doc_using_title_with_index(title, passage_base=passage_base)
        else:
            return self._retrieve_doc_using_title(title, passage_base=passage_base)
    
    def _retrieve_doc_id_using_title_with_index(self, title):
        pidx = self._find_page_index(title)
        doc_id = None
        if pidx > -1:
            fname = self.page_table[pidx]
            doc_id = get_doc_id_using_title(os.path.join(self.page_root, fname), title, self.is_lower)
            if doc_id is not None:
                doc_id += self.max_page_per_shard*pidx

        if doc_id is None:
            doc_id = -1
        return doc_id
    
    def _retrieve_doc_id_using_title(self, title):
        doc_id = None
        for pidx, fname in enumerate(self.page_table):
            doc_id = get_doc_id_using_title(os.path.join(self.page_root, fname), title, self.is_lower)
            if doc_id is not None:
                doc_id += self.max_page_per_shard*pidx
                break

        if doc_id is None:
            doc_id = -1
        return doc_id
    
    def retrieve_doc_id_using_title(self, title):
        if self.page_index is not None:
            return self._retrieve_doc_id_using_title_with_index(title)
        else:
            return self._retrieve_doc_id_using_title(title)
    

    ########################################################################
    # def get_fvec_using_doc_id(self, doc_idx):
    #     pidx, iidx = divmod(doc_idx, self.max_page_per_shard)
    #     page_path = self.page_table[pidx]
    #     fvec_root = os.path.join(self.data_root, self.page_info['rel_fvec_root'])
    #     base_name, _ = os.path.splitext(page_path)
    #     return get_fvec_using_doc_id(os.path.join(fvec_root, base_name+".fvecs"), iidx)
    ########################################################################

#get_doc_id_using_title


class DocumentLoaderOnMemory(_DefaultDocumentLoader):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(
        self,
        page_info_path) -> None:

        with open(page_info_path, "r") as f:
            self.page_info = json.load(f)
        
        self.data_root = os.path.dirname(page_info_path)
        self.page_table = self.page_info.get('page_table', None)
        self.page_index = self.page_info.get('page_index', None)
        self.is_lower = self.page_info.get('is_lower', False)
        self.max_page_per_shard = self.page_info.get('max_page_per_shard', None)
        self.num_pages = self.page_info.get('num_pages', None)
        self.rel_page_root = self.page_info.get('rel_page_root', None)
        assert self.page_table is not None, "can't find page_table from page_info"
        assert self.max_page_per_shard is not None, "can't find max_page_per_shard from page_info"
        assert self.num_pages is not None, "can't find num_pages from page_info"
        assert self.rel_page_root is not None, "can't find rel_page_root from page_info"

        self.page_root = os.path.join(self.data_root, self.rel_page_root)

        self._load_docs()

    def _load_docs(self):
        logging.info("load documents...")
        self.data = []
        for fname in self.page_table:
            docs = read_bz2(os.path.join(self.data_root, self.rel_page_root, fname))
            self.data.extend(docs)

    def retrieve_doc(self, doc_idx, passage_base=True):
        if doc_idx > self.num_pages:
            return None
        doc = copy.deepcopy(self.data[doc_idx])
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc

    def retrieve_docs(self, doc_idx_list, passage_base=True):
        return [self.retrieve_doc(x, passage_base) for x in doc_idx_list]
    
    def retrieve_docs_batch(self, batch, passage_base=True):
        return [self.retrieve_docs(x, passage_base) for x in batch]
    
    def _find_page_index(self, title):
        if self.is_lower:
            title = title.lower()

        for idx, pindex in enumerate(self.page_index):
            if pindex[0] <= title and title <= pindex[1]:
                return idx
        return -1
    
    def _retrieve_doc_using_title_with_index(self, title, passage_base=True):
        pidx = self._find_page_index(title)
        doc = None
        if pidx > -1:
            doc = get_line_using_title_memory(self.data, title, self.is_lower, start_idx=self.max_page_per_shard*pidx, search_len=self.max_page_per_shard)
        
        if doc is None:
            doc = {
                'title': 'NA',
                'text': ['NA']
            }
        
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc
    
    def _retrieve_doc_using_title(self, title, passage_base=True):
        doc = get_line_using_title_memory(self.data, title, self.is_lower, start_idx=0, search_len=self.num_pages)
        
        if doc is  None:
            doc = {
                'title': 'NA',
                'text': 'NA'
            }
        
        if passage_base:
            doc['text'] = ''.join(doc['text'])
        return doc
    
    def retrieve_doc_using_title(self, title, passage_base=True):
        if self.page_index is not None:
            return self._retrieve_doc_using_title_with_index(title, passage_base=passage_base)
        else:
            return self._retrieve_doc_using_title(title, passage_base=passage_base)
    
    def _retrieve_doc_id_using_title_with_index(self, title):
        pidx = self._find_page_index(title)
        doc_id = None
        if pidx > -1:
            doc_id = get_doc_id_using_title_memory(self.data, title, self.is_lower, start_idx=self.max_page_per_shard*pidx, search_len=self.max_page_per_shard)
            if doc_id is not None:
                doc_id += self.max_page_per_shard*pidx

        if doc_id is None:
            doc_id = -1
        return doc_id
    
    def _retrieve_doc_id_using_title(self, title):
        doc_id = get_doc_id_using_title_memory(self.data, title, self.is_lower, start_idx=0, search_len=self.num_pages)

        if doc_id is None:
            doc_id = -1
        return doc_id
    
    def retrieve_doc_id_using_title(self, title):
        if self.page_index is not None:
            return self._retrieve_doc_id_using_title_with_index(title)
        else:
            return self._retrieve_doc_id_using_title(title)



class DocumentLoaderWithCandidates(_DefaultDocumentLoader):
    _NEED_TO_SET_CANDIDATES=True

    def __init__(
        self) -> None:
        self._candidates = None
        self._len = 0

    @property
    def candidates(self):
        return self._candidates
    
    @candidates.setter
    def candidates(self, candidates):
        self._candidates = candidates
        self._len = len(candidates)

    def retrieve_doc(self, doc_idx:int, passage_base=True):
        # doc_idx: int
        if doc_idx > self._len:
            return None
        doc = self._candidates[doc_idx]
        if passage_base and isinstance(doc['text'], list):
            doc['text'] = ''.join(doc['text'])
        return doc
    
    def retrieve_docs(self, doc_idx_list:List[int], passage_base=True):
        # doc_idx_list: List[int]
        return [self.retrieve_doc(x, passage_base) for x in doc_idx_list]
    
    def retrieve_docs_batch(self, batch:List[List[int]], passage_base=True):
        # batch: List[List[int]]
        return [self.retrieve_docs(x, passage_base) for x in batch]


class _SocrerBase(object):
    def __init__(self, normalize=False):
        self.norm = normalize
    
    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm


class ScorerWithCandidates(_SocrerBase):
    _NEED_TO_SET_CANDIDATES=True

    def __init__(self, normalize=False) -> None:
        super().__init__(normalize)

        self._candidates = None
        self._len = 0
    
    @property
    def candidates(self):
        return self._candidates
    
    @candidates.setter
    def candidates(self, candidates):
        self._candidates = candidates
        self._len = len(candidates)
    
    def get_topk(self, query_vec, k=4, normalize=None):
        if self._candidates is None:
            return None
        
        k = min(k, self._len)

        if normalize is not None:
            if normalize:
                query_vec = self.normalize(query_vec)
        elif self.norm:
            query_vec = self.normalize(query_vec)

        # query_vec: (num_q, model_dim)
        # candidates: (num_cand, model_dim)
        # scores: (num_q, num_cand)
        scores = np.einsum("qd,cd->qc", query_vec, self._candidates)
        # topk_idx, topk_score: (num_q, top_k)
        topk_idx = np.flip(np.split(np.argsort(scores, axis=-1), [-k], axis=-1)[-1], axis=-1)
        topk_iidx = np.expand_dims(np.arange(len(query_vec))*self._len, axis=-1) + topk_idx
        topk_score = np.take(scores, topk_iidx)

        return (topk_score, topk_idx)


class FaissScorerBase(_SocrerBase):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            page_info_path,
            normalize=False
            ) -> None:
        super().__init__(normalize)

        self.page_info_path = page_info_path
        self.data_root = os.path.dirname(self.page_info_path)
        self.page_info = self.load_page_info(page_info_path)
    
    def load_page_info(self, page_info_path=None):
        if page_info_path is not None:
            with open(page_info_path, 'r') as f:
                return json.load(f)
        else:
            with open(self.page_info_path, 'r') as f:
                return json.load(f)
    
    def save_page_info(self, page_info=None, page_info_path=None):
        if page_info is not None and page_info_path is not None:
            with open(page_info_path, "w") as f:
                json.dump(page_info, f, indent=4)
        else:
            with open(self.page_info_path, "w") as f:
                json.dump(self.page_info, f, indent=4)
            
    
    def load_data(self, proportion_for_training=1.0):
        
        fvec_root = os.path.join(self.data_root, self.page_info['rel_fvec_root'])
        pages = self.page_info['page_table']

        prob = max(0, min(1.0, proportion_for_training))
        sampling = prob < 1.0
        
        fvec_list = []
        for page_path in pages:
            base_name, _ = os.path.splitext(page_path)
            fvecs = fvecs_read(os.path.join(fvec_root, base_name+".fvecs"))
            if sampling:
                num_samples = fvecs.shape[0]
                pick_num = int(num_samples*prob)
                fvecs_indice = np.random.choice(num_samples, pick_num, replace=False)
                fvecs = np.take(fvecs, fvecs_indice, axis=0)
            fvec_list.append(fvecs)
        fvec_list = np.concatenate(tuple(fvec_list), axis=0)
        return fvec_list
    
    def load_data_with_index(self):
        
        fvec_root = os.path.join(self.data_root, self.page_info['rel_fvec_root'])
        pages = self.page_info['page_table']
        
        fvec_list = []
        index_list = []
        offset = 0
        for page_path in pages:
            base_name, _ = os.path.splitext(page_path)
            fvecs = fvecs_read(os.path.join(fvec_root, base_name+".fvecs"))
            flen = fvecs.shape[0]
            fvec_list.append(fvecs)
            index_list.append(np.arange(offset, offset+flen))
            offset+=flen
        return fvec_list, index_list


class FaissScorer(FaissScorerBase):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            page_info_path,
            proportion_for_training=1.0,
            index_str="IVF65536,Flat",
            nprobe=4,
            normalize=False
            ) -> None:
        super(FaissScorer, self).__init__(page_info_path, normalize)
        
        self.proportion_for_training = proportion_for_training
        
        self.index = self.load_index(index_str)
        self.index.nprobe = nprobe

    def load_index(self, index_str="IVF65536,Flat"):
        index_fname = self.page_info.get("index_fname", None)

        if index_fname is None:
            index_fname = "wiki.index"
            index_path = os.path.join(self.data_root, index_fname)
            data = self.load_data(self.proportion_for_training)
            d = data.shape[-1]
            index = faiss.index_factory(d, index_str, faiss.METRIC_INNER_PRODUCT)
            logging.info('training index...')
            index.train(data)
            logging.info('loading fvecs...')
            data = self.load_data()
            logging.info('adding index...')
            index.add(data)
            faiss.write_index(index, index_path)

            self.page_info["index_fname"] = index_fname
            self.save_page_info()
        
        index_path = os.path.join(self.data_root, index_fname)
        return faiss.read_index(index_path)
    
    def get_topk(self, query_vec, k=4, normalize=False):
        if normalize is not None:
            if normalize:
                query_vec = self.normalize(query_vec)
        elif self.norm:
            query_vec = self.normalize(query_vec)

        return self.index.search(query_vec, k)
    

class FaissScorerExhaustive(FaissScorerBase):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            page_info_path,
            nprobe=1,
            normalize=False
            ) -> None:
        super(FaissScorerExhaustive, self).__init__(page_info_path, normalize)
        
        self.index = self.load_index()
        self.index.nprobe = nprobe

    def load_index(self):
        index_fname = self.page_info.get("index_exhaustive_fname", None)
        
        if index_fname is None:
            index_fname = "wiki_exhaustive.index"
            index_path = os.path.join(self.data_root, index_fname)

            logging.info('loading fvecs...')
            data = self.load_data()
            d = data.shape[-1]
            logging.info('vector dim: {}'.format(d))
            index = faiss.IndexFlatIP(d)
            logging.info('adding index...')
            index.add(data)
            
            faiss.write_index(index, index_path)

            self.page_info["index_exhaustive_fname"] = index_fname
            self.save_page_info()
        
        index_path = os.path.join(self.data_root, index_fname)
        return faiss.read_index(index_path)
    
    def get_topk(self, query_vec, k=4, normalize=False):
        if normalize is not None:
            if normalize:
                query_vec = self.normalize(query_vec)
        elif self.norm:
            query_vec = self.normalize(query_vec)

        return self.index.search(query_vec, k)

class FaissScorerExhaustiveGPU(FaissScorerBase):
    _NEED_TO_SET_CANDIDATES=False

    def __init__(self, 
            page_info_path,
            nprobe=1,
            gpu=0,
            normalize=False
            ) -> None:
        super(FaissScorerExhaustiveGPU, self).__init__(page_info_path, normalize)
        self.gpu = gpu
        self.index = self.load_index(gpu)
        self.index.nprobe = nprobe

    def load_index(self, gpu=0):
        # gpu resources
        res = faiss.StandardGpuResources()

        logging.info('loading fvecs...')
        data = self.load_data()
        d = data.shape[-1]
        logging.info('vector dim: {}'.format(d))
        index_flat = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(res, gpu, index_flat)
        logging.info('adding index...')
        index.add(data)
        
        return index
    
    def get_topk(self, query_vec, k=4, normalize=False):
        if normalize is not None:
            if normalize:
                query_vec = self.normalize(query_vec)
        elif self.norm:
            query_vec = self.normalize(query_vec)

        return self.index.search(query_vec, k)


# from huggingface transformers beamscorer 
# (https://github.com/huggingface/transformers/blob/master/src/transformers/generation_beam_search.py)
class BeamHypotheses(object):
    def __init__(self, num_beams: int):
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)
    
    def add(self, hyp, score):
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    
    def __iter__(self):
        sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)], reverse=True)
        for s, idx in sorted_next_scores:
            yield self.beams[idx]

    def is_done(self, best_cur_score):
        if len(self) < self.num_beams:
            return False
        else:
            return self.worst_score >= best_cur_score


def to_cuda_dict(batch):
    return {
        k: v.cuda() for k, v in batch.items()
    }

class SearchManager(object):
    def __init__(self,
            qp_formatter,
            encoder,
            max_batch_size=32,
            page_info_path=None,
            exhaustive_scorer=False,
            nprobe=4,
            normalize=False,
            use_gpu=False,
            model_gpu=0,
            scorer_gpu=None,
            ) -> None:
        super().__init__()

        self.qp_formatter = qp_formatter
        self.encoder = encoder

        self.max_batch_size = max_batch_size

        self.pad_token_id = self.qp_formatter.pad_token_id

        self.model_gpu = model_gpu
        self.scorer_gpu = scorer_gpu

        self.proc_in = lambda x: x

        if page_info_path is None:
            self.doc_loader = DocumentLoaderWithCandidates()
            self.scorer = ScorerWithCandidates(normalize=normalize)
        else:
            self.doc_loader = DocumentLoaderOnMemory(page_info_path)
            if exhaustive_scorer:
                if use_gpu and scorer_gpu is not None:
                    self.scorer = FaissScorerExhaustiveGPU(page_info_path, 
                        nprobe=nprobe, 
                        normalize=normalize,
                        gpu=scorer_gpu)
                else:
                    self.scorer = FaissScorerExhaustive(page_info_path, 
                        nprobe=nprobe, 
                        normalize=normalize)
            else:
                self.scorer = FaissScorer(page_info_path, nprobe=nprobe, normalize=normalize)
    
    def cuda(self, gpu=None):
        if gpu is None:
            gpu = self.model_gpu
        self.encoder.to(torch.device("cuda:{}".format(gpu)))
        self.proc_in = to_cuda_dict
    
    def cpu(self):
        self.encoder.cpu()
        self.proc_in = lambda x: x

    def search(
            self, 
            sample, 
            top_k=4, 
            n_hop=2,
            num_cands_per_topk=2,
            ):
        # (1)
        question = sample['question']

        # (1)
        # context_title = sample.get('context_title', None)
        # answer = sample.get('answer', None)
        context = sample.get('context', None)

        # prepare question and context
        tokenized_question, tokenized_context = self.prepare_query(question, context)

        # encode context
        encoded_context = None
        if tokenized_context is not None:
            tokenized_context = self.prepare_inputs(tokenized_context)
            # encoded_context: (num_cands, model_dim)
            encoded_context = self.encode_data_context(tokenized_context)
        self.set_candidates(context, encoded_context)

        # encode question
        tokenized_question = self.prepare_inputs(tokenized_question)
        # encoded_question: (1, model_dim)
        encoded_question = self.encode_data_query(tokenized_question)

        num_cands_per_topk = max(num_cands_per_topk, 1)
        cand_size = top_k*num_cands_per_topk
        # topk_score, topk_index: (1, top_k)
        topk_score, topk_index = self.scorer.get_topk(encoded_question, k=cand_size)

        topk_score = np.log2(topk_score)

        # top_k_hypo_cands, top_k_hypo_scores: (cand_size, 1)
        top_k_hypo_cands = np.transpose(topk_index)
        top_k_hypo_scores = np.transpose(topk_score)

        top_k_hypo_scores, top_k_hypo_cands = self.validate_topk(top_k_hypo_scores, top_k_hypo_cands)

        for current_hop in range(1, n_hop):

            # context: (top_k, )
            context = self.get_context(top_k_hypo_cands)
            q_sp = self.prepare_q_sp(question, context)
            q_sp_tokenized = self.prepare_inputs(q_sp)
            q_sp_encoded = self.encode_data_query(q_sp_tokenized)

            # topk_score, topk_index: (top_k, num_cands_per_beam)
            topk_score, topk_index = self.scorer.get_topk(q_sp_encoded, k=(num_cands_per_topk + current_hop))
            topk_score = np.log2(topk_score)

            top_k_hypo_scores,top_k_hypo_cands = self.merge_and_retrieve_top_candsize(
                                                    top_k_hypo_cands,
                                                    top_k_hypo_scores,
                                                    topk_index,
                                                    topk_score,
                                                    cand_size,
                                                    num_cands_per_topk
                                                )
            
            top_k_hypo_scores, top_k_hypo_cands = self.validate_topk(top_k_hypo_scores, top_k_hypo_cands)

        retrieved_docs = self.doc_loader.retrieve_docs_batch(top_k_hypo_cands[:top_k])
        return retrieved_docs
    
    def validate_topk(self, top_k_hypo_scores, top_k_hypo_cands):
        valid_hypo = np.argwhere(np.all(top_k_hypo_cands>-1, axis=-1)).flatten()
        return np.take(top_k_hypo_scores, valid_hypo, axis=0), np.take(top_k_hypo_cands, valid_hypo, axis=0)


    def get_context(self, top_k_hypo_cands):
        # top_k_hypo_cands: (cand_size, 1)
        # get last index of documents
        return [self.doc_loader.retrieve_doc(x[-1]) for x in top_k_hypo_cands]
    
    def merge_and_retrieve_top_candsize(
        self, 
        top_k_hypo_cands, 
        top_k_hypo_scores, 
        topk_index, 
        topk_score,
        cand_size,
        num_cands_per_topk):
        # top_k_hypo_cands: (cand_size, list_of_hop)
        # top_k_hypo_scores: (cand_size, )
        # topk_index, topk_score: (cand_size, num_cands_per_topk+current_hop)
        cur_cand_size, additional_cand_size = topk_index.shape[:2]
        topk_hypo_cands_extended = np.repeat(top_k_hypo_cands, additional_cand_size, axis=0)
        topk_hypo_scores_extended = np.repeat(top_k_hypo_scores, additional_cand_size, axis=0)
        topk_index_view = np.reshape(topk_index, (-1, 1))
        topk_score_view = np.reshape(topk_score, (-1, 1))

        topk_hypo_cands_m = np.concatenate((topk_hypo_cands_extended, topk_index_view), -1)
        topk_hypo_scores_m = topk_hypo_scores_extended + topk_score_view

        # filter duplicated elements
        filtered_idx = []
        for cand_idx in range(cur_cand_size):
            cand_cnt = 0
            for add_cand_idx in range(additional_cand_size):
                idx = cand_idx * additional_cand_size + add_cand_idx
                if cand_cnt < num_cands_per_topk and topk_index_view[idx][0] not in topk_hypo_cands_extended[idx].tolist():
                    filtered_idx.append(idx)
                    cand_cnt += 1

        # for idx in range(topk_index_view.shape[0]):
        #     if topk_index_view[idx][0] not in topk_hypo_cands_extended[idx].tolist():
        #         filtered_idx.append(idx)
        
        filtered_idx = np.array(filtered_idx)
        topk_hypo_cands_filtered = np.take(topk_hypo_cands_m, filtered_idx, axis=0)
        topk_hypo_scores_filtered = np.take(topk_hypo_scores_m, filtered_idx, axis=0)

        top_n = np.squeeze(np.argsort(topk_hypo_scores_filtered, axis=0)[-cand_size:][::-1])
        topk_hypo_cands_f= np.take(topk_hypo_cands_filtered, top_n, axis=0)
        topk_hypo_scores_f = np.take(topk_hypo_scores_filtered, top_n, axis=0)
        return topk_hypo_scores_f, topk_hypo_cands_f
        

    
    def set_candidates(self, context=None, encoded_context=None):
        if self.doc_loader._NEED_TO_SET_CANDIDATES:
            if context is None:
                raise ValueError("document reader({}) needs passage candidates. but candidates are None".format(self.doc_loader.__class__.__name__))
            self.doc_loader.candidates = context
        if self.scorer._NEED_TO_SET_CANDIDATES:
            if encoded_context is None:
                raise ValueError("scorer({}) needs passage candidates. but candidates are None".format(self.scorer.__class__.__name__))
            self.scorer.candidates = encoded_context
    
    def encode_data(self, inputs_batch):
        # list of (max_batch_size, model_dim)
        encoded_vectors = [
            self.encoder.encode_seq(self.proc_in(x)) for x in inputs_batch]
        return torch.cat(tuple([x for x in encoded_vectors]), 0).cpu().numpy()
    
    def encode_data_query(self, inputs_batch):
        # list of (max_batch_size, model_dim)
        encoded_vectors = [
            self.encoder.encode_query(self.proc_in(x)) for x in inputs_batch]
        return torch.cat(tuple([x for x in encoded_vectors]), 0).cpu().numpy()
    
    def encode_data_context(self, inputs_batch):
        # list of (max_batch_size, model_dim)
        encoded_vectors = [
            self.encoder.encode_context(self.proc_in(x)) for x in inputs_batch]
        return torch.cat(tuple([x for x in encoded_vectors]), 0).cpu().numpy()
    
    def prepare_q_sp(self, question, context):
        # question: (1), context: (top_k, )
        q_sp = []
        for nq in context:
            q_sp.append(self.qp_formatter.encode_q_sp(question, nq))
        
        return q_sp

    def prepare_query(self, question, context=None):
        # question: (1)
        # context: (num_context,)
        tokenized_question = [self.qp_formatter.encode_question(question)]

        tokenized_context = context
        if tokenized_context is not None:
            tokenized_context = [self.qp_formatter.encode_context(x) for x in context]

        # tokenized_question: (1, {input_ids, attention_mask})
        # tokenized_context: (num_cand, {input_ids, attention_mask})
        return tokenized_question, tokenized_context

    def prepare_inputs(self, samples):
        batched = self._batchfy(samples, max_batch_size=self.max_batch_size)
        # (sample_bsz, max_batch_size, {input_ids, attention_mask})
        return [
            self._collate_candidates(batched_sample, pad_id=self.pad_token_id) 
            for batched_sample in batched]
    
    def _batchfy(self, samples, max_batch_size=4):
        samples_batched = []
        _len, _mod = divmod(len(samples), max_batch_size)
        for idx in range(_len):
            samples_batched.append(samples[idx*max_batch_size:(idx+1)*max_batch_size])
        if _mod > 0:
            samples_batched.append(samples[_len*max_batch_size:])

        # (sample_bsz, max_batch_size, seq_len)
        return samples_batched
    
    def _collate_candidates(self, samples, pad_id=0):
        if len(samples) == 0:
            return {}
        item = samples[0]
        return {
            k: collate_tokens([torch.LongTensor(x[k]) for x in samples], pad_id if k.endswith('input_ids') else 0) for k in item.keys()
        }




