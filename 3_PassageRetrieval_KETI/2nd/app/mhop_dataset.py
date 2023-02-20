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

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_convert, default_collate
import json
import os
import bz2
import functools
import glob
import tqdm
import random
import multiprocessing

from data_utils import collate_tokens, collate_tokens_new_dim, collate_new_dim

import logging


def get_line(bz2_path, line_idx):
    with bz2.open(bz2_path) as fp:
        for i, line in enumerate(fp):
            if i == line_idx:
                return line
            elif i > line_idx:
                break
    return None


class QueryPassageFormatter(object):
    def __init__(
        self,
        tokenizer,
        max_q_len=50,
        max_q_sp_len=512,
        max_c_len=512,
        max_len=512,
        remove_question_mark=False,
        add_cls_token=False,
        add_sep_token=False,
        add_token_type_ids=True,
        cls_token_id=-1,
        sep_token_id=-1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_q_sp_len = max_q_sp_len
        self.max_c_len = max_c_len
        self.max_len = max_len
        self.remove_question_mark = remove_question_mark

        self.add_cls_token = add_cls_token
        self.add_sep_token = add_sep_token
        self.add_token_type_ids = add_token_type_ids

        self.set_special_tokens(tokenizer, cls_token_id, sep_token_id)

    def set_special_tokens(self, tokenizer, cls_token_id, sep_token_id):
        if cls_token_id >= 0:
            self.cls_token_id = cls_token_id
        elif tokenizer.cls_token_id is not None:
            self.cls_token_id = tokenizer.cls_token_id
        elif len(tokenizer.additional_special_tokens_ids) > 1:
            self.cls_token_id = tokenizer.additional_special_tokens_ids[0]
        else:
            self.cls_token_id = tokenizer.pad_token_id

        if sep_token_id >= 0:
            self.sep_token_id = sep_token_id
        elif tokenizer.sep_token_id is not None:
            self.sep_token_id = tokenizer.sep_token_id
        elif len(tokenizer.additional_special_tokens_ids) > 1:
            self.sep_token_id = tokenizer.additional_special_tokens_ids[1]
        else:
            self.sep_token_id = tokenizer.pad_token_id

        self.pad_token_id = tokenizer.pad_token_id

        # special tokens for spans
        if len(tokenizer.additional_special_tokens_ids) > 2:
            self.sep_token_id_lv2 = tokenizer.additional_special_tokens_ids[2]
        else:
            self.sep_token_id_lv2 = self.sep_token_id

    def encode_para(self, para, max_len=512, add_cls_token=False, add_sep_token=False):
        return self.encode_pair(para["title"].strip(), para["text"].strip(), max_len=max_len, add_cls_token=add_cls_token, add_sep_token=add_sep_token)

    def encode_pair(self, text1, text2=None, max_len=512, add_cls_token=False, add_sep_token=False, convert2tensor=True):
        seq1 = self.tokenizer.encode_plus(
            text1,
            max_length=max_len,
            truncation=True,
            add_special_tokens=False)
        input_ids_s1 = seq1.input_ids
        attention_mask_s1 = seq1.attention_mask
        if add_cls_token:
            input_ids_s1 = [self.cls_token_id] + input_ids_s1
            attention_mask_s1 = [1] + attention_mask_s1

        input_ids_s2 = []
        attention_mask_s2 = []

        if text2 is not None:
            seq2 = self.tokenizer.encode_plus(
                text2,
                max_length=max_len,
                truncation=True,
                add_special_tokens=False)
            input_ids_s2 = seq2.input_ids
            attention_mask_s2 = seq2.attention_mask

        if add_sep_token:
            input_ids_s1 += [self.sep_token_id]
            attention_mask_s1 += [1]

            if text2 is not None:
                input_ids_s2 += [self.sep_token_id]
                attention_mask_s2 += [1]

        input_ids = input_ids_s1 + input_ids_s2
        attention_mask = attention_mask_s1 + attention_mask_s2

        return_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if 'token_type_ids' in seq1 and self.add_token_type_ids:
            return_val['token_type_ids'] = [0] * \
                len(input_ids_s1) + [1] * len(input_ids_s2)

        # truncate
        for k, v in return_val.items():
            if len(v) > max_len:
                return_val[k] = return_val[k][:max_len]

        # convert to tensor
        if convert2tensor:
            for k, v in return_val.items():
                return_val[k] = torch.LongTensor(return_val[k])

        return return_val

    def prepare_question(self, question):
        if self.remove_question_mark:
            if question.endswith("?"):
                question = question[:-1]
        return question

    def encode_question(self, question):
        question = self.prepare_question(question)

        return self.encode_pair(
            question,
            max_len=self.max_q_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)

    def encode_q_sp(self, question, passage):
        question = self.prepare_question(question)
        return self.encode_pair(
            question,
            passage["text"].strip(),
            max_len=self.max_q_sp_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)

    def encode_context(self, passage):
        return self.encode_para(
            passage,
            self.max_c_len,
            add_cls_token=self.add_cls_token,
            add_sep_token=self.add_sep_token)


class MhopDatasetForRetriever(Dataset):
    def __init__(self,
                 qp_formatter,
                 data_path=None,
                 negative_sample="negative_ctxs",
                 ):
        super().__init__()
        self.qp_formatter = qp_formatter
        self.negative_sample = negative_sample

        if data_path is not None:
            logging.info(f"Loading data from {data_path}")
            self.data = [json.loads(line)
                         for line in open(data_path).readlines()]
            if negative_sample != "negative_ctxs":
                for idx in range(len(self.data)):
                    self.data[idx]["negative_ctxs"] = self.data[idx][negative_sample]
            self.data = [_ for _ in self.data if len(_["negative_ctxs"]) >= 2]

            for idx in tqdm.tqdm(range(len(self.data)), desc="split start, end para"):
                if self.data[idx]['type'] == 'bridge':
                    self.data[idx]["positive_ctxs_start"] = [v for v in self.data[idx]
                                                             ["positive_ctxs"] if v['title'] not in self.data[idx]["final_supporting_facts"]]
                    self.data[idx]["positive_ctxs_end"] = [v for v in self.data[idx]
                                                           ["positive_ctxs"] if v['title'] in self.data[idx]["final_supporting_facts"]]
                    self.data[idx]["random"] = False
                    if len(self.data[idx]["positive_ctxs_start"]) == 0 or len(self.data[idx]["positive_ctxs_end"]) == 0:
                        self.data[idx]["random"] = True

            logging.info(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']

        if sample["type"] == "bridge":
            if sample["random"]:
                random.shuffle(sample["positive_ctxs"])
                start_para = sample["positive_ctxs"][0]
                end_para = sample["positive_ctxs"][-1]
            else:
                random.shuffle(sample["positive_ctxs_start"])
                random.shuffle(sample["positive_ctxs_end"])
                start_para = sample["positive_ctxs_start"][0]
                end_para = sample["positive_ctxs_end"][0]

        else:
            random.shuffle(sample["positive_ctxs"])
            start_para = sample["positive_ctxs"][0]
            end_para = sample["positive_ctxs"][-1]

        random.shuffle(sample["negative_ctxs"])

        start_para_codes = self.qp_formatter.encode_context(start_para)
        end_para_codes = self.qp_formatter.encode_context(end_para)
        neg_codes_1 = self.qp_formatter.encode_context(
            sample["negative_ctxs"][0])
        neg_codes_2 = self.qp_formatter.encode_context(
            sample["negative_ctxs"][1])

        q_codes = self.qp_formatter.encode_question(question)
        q_sp_codes = self.qp_formatter.encode_q_sp(question, start_para)

        return {
            "q_codes": q_codes,
            "q_sp_codes": q_sp_codes,
            "start_para_codes": start_para_codes,
            "end_para_codes": end_para_codes,
            "neg_codes_1": neg_codes_1,
            "neg_codes_2": neg_codes_2,
        }

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        return functools.partial(mhop_retriever_collate, pad_id=self.qp_formatter.pad_token_id)


def chunkify(lst, n=1):
    return [lst[i::n] for i in range(n)]


def create_pages(cmd_list):
    for cmd_obj in tqdm.tqdm(cmd_list, desc="write shards..."):
        is_lower = cmd_obj.get('is_lower', False)
        with bz2.open(os.path.join(cmd_obj["page_root"], cmd_obj["output_path"]), 'w') as fpage:
            for item in tqdm.tqdm(cmd_obj["elems"], desc="write pages..."):
                line = get_line(item[1], item[2])
                if line is None:
                    raise IndexError(
                        "fail to get line {} from file {}".format(item[2], item[1]))

                abstract = json.loads(line)
                abstract_title = abstract['title'].lower(
                ) if is_lower else abstract['title']
                item_title = item[0].lower() if is_lower else item[0]

                assert item_title == abstract_title, "assert item({}) != title({}). from file {}".format(
                    item[0], abstract['title'], item[1])

                fpage.write((json.dumps({
                    "title": abstract['title'],
                    "text": abstract['text']
                })+'\n').encode('utf-8'))


class FullWikiDatasetForRetriever(MhopDatasetForRetriever):
    def __init__(
        self,
        qp_formatter,
        target_bz2,
        is_doc_level=True
    ):

        super(FullWikiDatasetForRetriever, self).__init__(
            qp_formatter,
            data_path=None,
        )

        self.data = [json.loads(line)
                     for line in bz2.open(target_bz2, "r").readlines()]

        if is_doc_level:
            for idx in range(len(self.data)):
                self.data[idx]['text'] = ''.join(self.data[idx]['text'])

    def __getitem__(self, index):
        sample = self.data[index]
        return self.qp_formatter.encode_context(sample)

    def __len__(self):
        return len(self.data)

    @classmethod
    def create_page_id_map_shard(
            cls,
            input_data_pattern,
            repr_output_root,
            page_dir='pages',
            max_page_per_shard=100000):

        page_root = os.path.join(repr_output_root, page_dir)
        if not os.path.isdir(page_root):
            os.makedirs(page_root)

        in_files = glob.glob(input_data_pattern)

        shard_id = 0
        num_pages = 0
        page_table = {}
        from_id, to_id = shard_id * \
            max_page_per_shard, (shard_id + 1)*max_page_per_shard
        output_path = "{:05d}_{:05d}.pages".format(from_id, to_id)

        page_list = []
        for in_file in tqdm.tqdm(in_files):
            with bz2.open(in_file) as f:
                for line in f.readlines():
                    abstract = json.loads(line)

                    page_list.append((json.dumps({
                        "title": abstract['title'],
                        "text": abstract['text']
                    })+'\n').encode('utf-8'))

                    if len(page_list) == max_page_per_shard:
                        with bz2.open(os.path.join(page_root, output_path), 'w') as fpage:
                            fpage.writelines(page_list)
                        page_table[shard_id] = output_path
                        num_pages += max_page_per_shard
                        shard_id += 1
                        from_id, to_id = shard_id * \
                            max_page_per_shard, (shard_id + 1) * \
                            max_page_per_shard - 1
                        output_path = "{:05d}_{:05d}.pages".format(
                            from_id, to_id)
                        page_list = []
        if len(page_list) > 0:
            with bz2.open(os.path.join(page_root, output_path), 'w') as fpage:
                fpage.writelines(page_list)
            page_table[shard_id] = output_path
            num_pages += len(page_list)
            shard_id += 1
            from_id, to_id = shard_id * \
                max_page_per_shard, (shard_id + 1)*max_page_per_shard - 1
            output_path = "{:05d}_{:05d}.pages".format(from_id, to_id)
            page_list = []

        output_path = os.path.join(repr_output_root, "page_info.json")
        with open(output_path, 'w') as f:
            json.dump({
                "page_table": page_table,
                "max_page_per_shard": max_page_per_shard,
                "num_pages": num_pages,
                "num_shards": shard_id,
                "rel_page_root": page_dir,
                "is_lower": False,
            }, f, indent=4)

    @classmethod
    def create_page_id_map_shard_with_index(
            cls,
            input_data_pattern,
            repr_output_root,
            page_dir='pages',
            max_page_per_shard=100000,
            is_lower=True):

        page_root = os.path.join(repr_output_root, page_dir)
        if not os.path.isdir(page_root):
            os.makedirs(page_root)

        in_files = glob.glob(input_data_pattern)

        # create index
        title_path_pairs = []
        for in_file in tqdm.tqdm(in_files, desc="create index..."):
            with bz2.open(in_file) as f:
                for line_idx, line in enumerate(f.readlines()):
                    abstract = json.loads(line)
                    title_path_pairs.append((
                        abstract['title'].lower(
                        ) if is_lower else abstract['title'],
                        in_file,
                        line_idx
                    ))
        title_path_pairs = sorted(title_path_pairs, key=lambda item: item[0])

        job_cmd = []
        num_pages = len(title_path_pairs)
        n_parts = num_pages//max_page_per_shard
        for shard_id in range(n_parts):
            from_id, to_id = shard_id * \
                max_page_per_shard, (shard_id + 1)*max_page_per_shard - 1
            output_path = "{:05d}_{:05d}.pages".format(from_id, to_id)
            job_cmd.append({
                "output_path": output_path,
                "page_root": page_root,
                "index": (title_path_pairs[from_id][0], title_path_pairs[to_id][0]),
                "elems": title_path_pairs[from_id:to_id + 1],
                "is_lower": is_lower,
            })

        if num_pages % max_page_per_shard != 0:
            from_id, to_id = n_parts*max_page_per_shard, num_pages - 1
            output_path = "{:05d}_{:05d}.pages".format(from_id, to_id)
            job_cmd.append({
                "output_path": output_path,
                "page_root": page_root,
                "index": (title_path_pairs[from_id][0], title_path_pairs[to_id][0]),
                "elems": title_path_pairs[from_id:to_id + 1],
                "is_lower": is_lower,
            })
            n_parts += 1

        page_info = {
            "page_table": [item["output_path"] for item in job_cmd],
            "page_index": [item["index"] for item in job_cmd],
            "max_page_per_shard": max_page_per_shard,
            "num_pages": num_pages,
            "num_shards": n_parts,
            "rel_page_root": page_dir,
            "is_lower": is_lower,
        }

        pageinfo_path = os.path.join(repr_output_root, "page_info.json")
        with open(pageinfo_path, 'w') as f:
            json.dump(page_info, f, indent=4)

        num_proc = min(max(multiprocessing.cpu_count(), 1), len(job_cmd))

        chunks = chunkify(job_cmd, num_proc)
        jobs = []
        for i in range(num_proc):
            job = multiprocessing.Process(
                target=create_pages, args=(chunks[i],))
            jobs.append(job)
            job.start()

        for job in jobs:
            job.join()

    @classmethod
    def get_shard_pairs(
            cls,
            repr_output_root,
            repr_dir='fvecs',
            local_rank=0,
            world_size=1):

        page_info_path = os.path.join(repr_output_root, 'page_info.json')
        with open(page_info_path, 'r') as f:
            p_info = json.load(f)
            page_dir = p_info["rel_page_root"]
            p_info["rel_fvec_root"] = repr_dir

        if local_rank == 0 or world_size == 1:
            with open(page_info_path, 'w') as f:
                json.dump(p_info, f, indent=4)

        page_root = os.path.join(repr_output_root, page_dir)
        repr_root = os.path.join(repr_output_root, repr_dir)
        if local_rank == 0 and not os.path.isdir(repr_root):
            os.makedirs(repr_root)
        page_files = glob.glob(os.path.join(page_root, "*.pages"))
        page_files = sorted(page_files)

        target_pages = [x for idx, x in enumerate(
            page_files) if (idx % world_size) == local_rank]

        target_output_pair = []
        for x in target_pages:
            base_name, _ = os.path.splitext(os.path.basename(x))
            target_output_pair.append(
                (x, os.path.join(repr_root, base_name+'.fvecs')))
        return target_output_pair

    def get_collate_fn(self):
        return functools.partial(fullwiki_collate, pad_id=self.qp_formatter.pad_token_id)


class HotpotQADatasetForRetriever(Dataset):
    def __init__(self,
                 qp_formatter=None,
                 in_file=None,
                 split_info=None,
                 ):

        super(HotpotQADatasetForRetriever, self).__init__()

        if split_info is not None:
            self.split_info = split_info
        elif in_file.find("dev") > -1:
            self.split_info = "dev"
        elif in_file.find("test") > -1:
            self.split_info = "test"
        else:
            if self.split_info not in ['dev', 'test']:
                raise ValueError(
                    "split_info must be a 'test' or 'dev' but {}".format(self.split_info))

        self.data = self._preproc_data(in_file)

    def _preproc_data(self, in_file):

        data_list = json.load(open(in_file, 'r'))

        if self.split_info == "dev":
            t_keys = ['_id', 'answer', 'question',
                      'supporting_facts', 'context', 'type', 'level']
        elif self.split_info == "test":
            t_keys = ['_id', 'question', 'context']
        else:
            raise ValueError("invalid split info: {}".format(self.split_info))

        data = []
        for sample in data_list:
            sample_new = {}

            for t_key in t_keys:
                sample_new[t_key] = sample[t_key]
                if t_key == 'context':
                    sample_new[t_key] = [
                        {'title': item[0], 'text':''.join(item[1])} for item in sample[t_key]]
                    sample_new['context_title'] = [item[0]
                                                   for item in sample[t_key]]
                    #sample_new['context_text'] = [''.join(item[1]) for item in sample[t_key]]

            if self.split_info == "dev":
                sample_new["supporting_facts"] = [x[0]
                                                  for x in sample["supporting_facts"]]
            data.append(sample_new)

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        return simple_collate


class HotpotRerankerDataset(Dataset):
    def __init__(self,
                 qp_formatter,
                 in_file,
                 split_info=None,
                 rerank_mode=False,
                 cand_topk=16,
                 use_cache=True,
                 preprocessing=False,
                 max_ex_num=-1
                 ):

        super(HotpotRerankerDataset, self).__init__()

        if split_info is not None:
            self.split_info = split_info
        elif in_file.find("train") > -1:
            self.split_info = "train"
        elif in_file.find("dev") > -1:
            self.split_info = "dev"
        elif in_file.find("test") > -1:
            self.split_info = "test"
        else:
            if self.split_info not in ['dev', 'train', 'test']:
                raise ValueError(
                    "split_info must be a 'train' or 'dev' or 'test' but {}".format(self.split_info))

        self.qp_formatter = qp_formatter
        self.rerank_mode = rerank_mode and self.split_info != "train"
        self.cand_topk = cand_topk
        self.max_ex_num = max_ex_num

        self.data = self.load_data(in_file, use_cache, preprocessing)

    def load_data(self, in_file, use_cache=True, preprocessing=False):
        if use_cache:
            # from hashlib import blake2s
            hash_str = '.'.join([str(self.rerank_mode), str(self.cand_topk), str(self.max_ex_num)])
            # hash_str = str(blake2s(hash_str.encode('utf-8'), digest_size=8, key=b'HotpotRerankerDataset').hexdigest())
            cache_path = in_file + "." + hash_str
            if not preprocessing and os.path.isfile(cache_path):
                return torch.load(cache_path)

        data = self._preproc_data(in_file)
        if use_cache:
            torch.save(data, cache_path)
        return data

    def _preproc_data(self, in_file):

        data_list = json.load(open(in_file, 'r'))

        if self.split_info not in ["dev", "test", "train"]:
            raise ValueError("invalid split info: {}".format(self.split_info))

        max_len = self.qp_formatter.max_len
        pad_id = self.qp_formatter.pad_token_id
        sep_id = self.qp_formatter.sep_token_id_lv2
        add_token_type_ids = self.qp_formatter.add_token_type_ids

        max_ex_num = self.max_ex_num
        if max_ex_num == -1:
            max_ex_num = len(data_list)

        data = []

        if self.rerank_mode:
            for sample in tqdm.tqdm(data_list[:max_ex_num], desc="preproc..."):
                _id = sample['_id']
                question = sample['question']
                q_t = self.qp_formatter.encode_pair(
                    question, text2=None, max_len=max_len, 
                    add_cls_token=True, add_sep_token=True, convert2tensor=False)

                sample_new = {
                    "_id": _id,
                    "input_ids": [],
                    "attention_mask": []
                }

                for cand in sample['candidates']:
                    cand_t_1 = self.qp_formatter.encode_pair(
                        cand[0]['title'], text2=cand[0]['text'], 
                        max_len=max_len, add_cls_token=False, 
                        add_sep_token=True, convert2tensor=False)
                    cand_t_2 = self.qp_formatter.encode_pair(
                        cand[1]['title'], text2=cand[1]['text'], 
                        max_len=max_len, add_cls_token=False, 
                        add_sep_token=True, convert2tensor=False)
                    input_ids = q_t['input_ids'] + \
                        cand_t_1['input_ids'] + \
                        cand_t_2['input_ids']
                    attention_mask = [1] * len(input_ids)

                    sample_new["input_ids"].append(torch.LongTensor(input_ids))
                    sample_new["attention_mask"].append(
                        torch.LongTensor(attention_mask))

                    if 'token_type_ids' in cand_t_1 and add_token_type_ids:
                        if "token_type_ids" not in sample_new:
                            sample_new["token_type_ids"] = []
                        sample_new['token_type_ids'].append(torch.LongTensor(
                            [0]*len(q_t['input_ids']) + [1]*(len(input_ids) - len(q_t['input_ids']))))

                sample_new["input_ids"] = sample_new["input_ids"][:self.cand_topk]
                sample_new["attention_mask"] = sample_new["attention_mask"][:self.cand_topk]
                if add_token_type_ids and "token_type_ids" in sample_new:
                    sample_new["token_type_ids"] = sample_new["token_type_ids"][:self.cand_topk]

                if self.split_info in ["dev", "train"]:
                    sample_new["score"] = torch.FloatTensor(
                        sample['scores'][:self.cand_topk])
                    sample_new["em"] = torch.LongTensor(
                        sample['scores_em'][:self.cand_topk])
                    sample_new["scores"] = torch.LongTensor(
                        sample['scores_list'][:self.cand_topk])

                data.append(sample_new)

        else:
            for sample in tqdm.tqdm(data_list[:max_ex_num], desc="preproc..."):
                _id = sample['_id']
                question = sample['question']
                q_t = self.qp_formatter.encode_pair(
                    question, text2=None, max_len=512, 
                    add_cls_token=True, add_sep_token=True, convert2tensor=False)

                if self.split_info in ["dev", "train"]:
                    for cand, score, em, scores in zip(sample['candidates'],
                                                       sample['scores'],
                                                       sample['scores_em'],
                                                       sample['scores_list']):

                        cand_t_1 = self.qp_formatter.encode_pair(
                            cand[0]['title'], text2=cand[0]['text'], 
                            max_len=max_len, add_cls_token=False, 
                            add_sep_token=True, convert2tensor=False)
                        cand_t_2 = self.qp_formatter.encode_pair(
                            cand[1]['title'], text2=cand[1]['text'], 
                            max_len=max_len, add_cls_token=False, 
                            add_sep_token=True, convert2tensor=False)
                        input_ids = q_t['input_ids'] + \
                            cand_t_1['input_ids'] + \
                            cand_t_2['input_ids']
                        attention_mask = [1] * len(input_ids)

                        sample_new = {
                            "input_ids": torch.LongTensor(input_ids),
                            "attention_mask": torch.LongTensor(attention_mask),
                            "score": score,
                            "em": em,
                            "scores": torch.LongTensor(scores),
                        }

                        if 'token_type_ids' in cand_t_1 and add_token_type_ids:
                            if "token_type_ids" not in sample_new:
                                sample_new["token_type_ids"] = []
                            sample_new['token_type_ids'] = torch.LongTensor(
                                [0]*len(q_t['input_ids']) + [1]*(len(input_ids) - len(q_t['input_ids'])))

                        data.append(sample_new)
                else:
                    for cand in sample['candidates']:

                        cand_t_1 = self.qp_formatter.encode_pair(
                            cand[0]['title'], text2=cand[0]['text'], 
                            max_len=max_len, add_cls_token=False, 
                            add_sep_token=True, convert2tensor=False)
                        cand_t_2 = self.qp_formatter.encode_pair(
                            cand[1]['title'], text2=cand[1]['text'], 
                            max_len=max_len, add_cls_token=False, 
                            add_sep_token=True, convert2tensor=False)
                        input_ids = q_t['input_ids'] + \
                            cand_t_1['input_ids'] + \
                            cand_t_2['input_ids']
                        attention_mask = [1] * len(input_ids)

                        sample_new = {
                            "input_ids": torch.LongTensor(input_ids),
                            "attention_mask": torch.LongTensor(attention_mask),
                        }

                        if 'token_type_ids' in cand_t_1 and add_token_type_ids:
                            if "token_type_ids" not in sample_new:
                                sample_new["token_type_ids"] = []
                            sample_new['token_type_ids'] = torch.LongTensor(
                                [0]*len(q_t['input_ids']) + [1]*(len(input_ids) - len(q_t['input_ids'])))

                        data.append(sample_new)

        return data

    def __getitem__(self, index):
        example = self.data[index]
        example_new = {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
        }
        if "token_type_ids" in example:
            example_new["token_type_ids"] = example["token_type_ids"]

        if self.split_info in ["dev", "train"]:
            example_new["score"] = example["score"]
            example_new["em"] = example["em"]
            example_new["scores"] = example["scores"]

        return example_new

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self, lbl="em"):
        if self.rerank_mode:
            def collate_fn(samples, pad_id=0, w_lbl=True):
                if len(samples) == 0:
                    return {}

                item = samples[0]

                collated_batch = {
                    "input_ids": collate_tokens_new_dim([ex["input_ids"] for ex in samples], pad_id),
                    "attention_mask": collate_tokens_new_dim([ex["attention_mask"] for ex in samples], 0),
                }

                if "token_type_ids" in item:
                    collated_batch["token_type_ids"] = collate_tokens_new_dim([ex["token_type_ids"] for ex in samples], 0)

                if w_lbl:
                    collated_batch["labels"] = collate_new_dim([ex[lbl] for ex in samples])
                return collated_batch
            return functools.partial(
                collate_fn, 
                pad_id=self.qp_formatter.pad_token_id, 
                w_lbl=self.split_info in ["dev", "train"])
        else:
            def collate_fn(samples, pad_id=0, w_lbl=True):
                if len(samples) == 0:
                    return {}

                item = samples[0]

                collated_batch = {
                    "input_ids": collate_tokens([ex["input_ids"] for ex in samples], pad_id),
                    "attention_mask": collate_tokens([ex["attention_mask"] for ex in samples], 0),
                }

                if "token_type_ids" in item:
                    collated_batch["token_type_ids"] = collate_tokens([ex["token_type_ids"] for ex in samples], 0)

                if w_lbl:
                    collated_batch["labels"] = default_collate([ex[lbl] for ex in samples])
                return collated_batch
                
            return functools.partial(
                collate_fn, 
                pad_id=self.qp_formatter.pad_token_id, 
                w_lbl=self.split_info in ["dev", "train"])


"""
"input_ids"
"attention_mask"
"score"
"em"
"scores"
"""


# modified from facebook research mdr
def mhop_retriever_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    item = samples[0]

    return {
        k: {kk: collate_tokens([s[k][kk] for s in samples], pad_id if kk.endswith('input_ids') else 0) for kk in item[k].keys()} for k in item.keys()
    }


def fullwiki_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    item = samples[0]

    return {
        k: collate_tokens([s[k] for s in samples], pad_id if k.endswith('input_ids') else 0) for k in item.keys()
    }


def hotpotqa_collate(samples):
    if len(samples) == 0:
        return {}

    item = samples[0]

    return {
        k: [x[k] for x in samples] for k in item.keys()
    }


def simple_collate(samples):
    return samples


def to_cuda_mhop_retriever(batch):
    return {
        k: {kk: vv.cuda()} for kk, vv in v.items() for k, v in batch.items()
    }


def to_cuda_fullwiki(batch):
    return {
        k: v.cuda() for k, v in batch.items()
    }


if __name__ == "__main__":
    import functools
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")

    # dev_data_distractor = "../data/hotpotqa/hotpot_dev_distractor_v1.json"
    # dev_data_fullwiki = "../data/hotpotqa/hotpot_dev_fullwiki_v1.json"
    # dev_test_fullwiki = "../data/hotpotqa/hotpot_test_fullwiki_v1.json"

    # dev distractor
    # _id, answer, question, supporting_facts, context, type, level

    # dev fullwiki
    # _id, answer, question, supporting_facts, context, type, level

    # test fullwiki
    # _id, question, context

    # hotpot_dataset = HotpotQADatasetForRetriever(
    #     in_file=dev_data_distractor,
    #     tokenizer=tokenizer,
    # )

    # for idx, item in zip(range(1), hotpot_dataset):
    #     pass

    data_path_train = "output/T5EncoderSimpleMomentumRetriever_ket5-base-en/hotpot_train_v1.1.json"
    data_path_valid = "output/T5EncoderSimpleMomentumRetriever_ket5-base-en/hotpot_dev_distractor_v1.json"
    qp_formatter = QueryPassageFormatter(
        tokenizer
    )

    ds_train = HotpotRerankerDataset(
        qp_formatter, data_path_train, rerank_mode=True)
    ds_valid = HotpotRerankerDataset(
        qp_formatter, data_path_valid, rerank_mode=True)

    for idx, data in zip(range(2), ds_train):
        print(idx, data)
        print(tokenizer.decode(data["input_ids"]))

    for idx, data in zip(range(2), ds_valid):
        print(idx, data)
        print(tokenizer.decode(data["input_ids"][0]))

    # FullWikiDatasetForRetriever.create_page_id_map_shard(
    #     input_data_pattern="../../../data/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts/*/*.bz2",
    #     repr_output_root="../../../data/hotpotqa/full_wiki",
    #     max_page_per_shard=100000)

    # bsz = 2
    # #tokenizer = AutoTokenizer.from_pretrained("google/electra-large-generator")
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # dataset = MhopDatasetForRetriever(
    #     tokenizer,
    #     "../../../data/hotpotqa/hotpot_train_with_neg_v0.json",
    #     add_cls_token=True,
    #     add_sep_token=True)

    # loader = DataLoader(
    #     dataset,
    #     batch_size=bsz,
    #     collate_fn=functools.partial(mhop_retriever_collate, pad_id=pad_id))

    # for idx, data in zip(range(2), loader):
    #     print(data)

    #     exit()

    # print('q_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["q_input_ids"][0]])
    # print('q_mask: ', [x for x in data["q_mask"][0]])

    # print('q_sp_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["q_sp_input_ids"][0]])
    # print('q_sp_mask: ', [x for x in data["q_sp_mask"][0]])

    # print('c1_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["c1_input_ids"][0]])
    # print('c1_mask: ', [x for x in data["c1_mask"][0]])

    # print('c2_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["c2_input_ids"][0]])
    # print('c2_mask: ', [x for x in data["c2_mask"][0]])

    # print('neg1_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["neg1_input_ids"][0]])
    # print('neg1_mask: ', [x for x in data["neg1_mask"][0]])

    # print('neg2_input_ids: ', [tokenizer._convert_id_to_token(int(x)) for x in data["neg2_input_ids"][0]])
    # print('neg2_mask: ', [x for x in data["neg2_mask"][0]])

    # for data in tqdm.tqdm(dataset):
    #     pass

    # for idx, data in zip(range(5), dataset):
    #     print(data['q_codes'])
    #     print('q_codes: ', [tokenizer._convert_id_to_token(int(x)) for x in data['q_codes']["input_ids"]])
    #     print(data['q_sp_codes'])
    #     print('q_sp_codes: ', [tokenizer._convert_id_to_token(int(x)) for x in data['q_sp_codes']["input_ids"]])
    #     print('start_para_codes: ', [tokenizer._convert_id_to_token(int(x)) for x in data['start_para_codes']["input_ids"]])
    #     print('end_para_codes: ', [tokenizer._convert_id_to_token(int(x)) for x in data['end_para_codes']["input_ids"]])
    #     print('neg_codes_1: ', [tokenizer._convert_id_to_token(int(x)) for x in data['neg_codes_1']["input_ids"]])
    #     print('neg_codes_2: ', [tokenizer._convert_id_to_token(int(x)) for x in data['neg_codes_2']["input_ids"]])

        # print('q_codes: ', tokenizer.decode(data['q_codes'].input_ids[0]))
        # print('q_sp_codes: ', tokenizer.decode(data['q_sp_codes'].input_ids[0]))
        # print('start_para_codes: ', tokenizer.decode(data['start_para_codes'].input_ids[0]))
        # print('end_para_codes: ', tokenizer.decode(data['end_para_codes'].input_ids[0]))
        # print('neg_codes_1: ', tokenizer.decode(data['neg_codes_1'].input_ids[0]))
        # print('neg_codes_2: ', tokenizer.decode(data['neg_codes_2'].input_ids[0]))
