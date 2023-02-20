
import json
import random
import regex as re
import numpy as np

import torch

from transformers import AutoTokenizer

from models import T5EncoderForSequenceClassificationFirstSubmeanObjmean

class Service:
    task = [
        {
            'name': "relation_extraction",
            'description': 'Relation Extraction Version 1.1'
        }
    ]

    def __init__(self):
        self.extract = RelationExtraction()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.extract.do_extract(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class RelationExtraction(object):
    def __init__(self):
        self.config = json.load(open("config.json", "r"))

        
        hf_path = self.config["kr_common-sense"].get("hf_path", "KETI-AIR/ke-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        
        self.extract_manager = {
            "kr_common-sense": self.load_extract_manager(hf_path)
        }
        
    
    @staticmethod
    def load_extract_manager(hf_path):
        model = T5EncoderForSequenceClassificationFirstSubmeanObjmean.from_pretrained(hf_path)
        model.eval()

        return model

    def do_extract(self, content):
        doc = content.get('doc', None)
        arg_pairs = content.get('arg_pairs', None)
        if doc is None:
            return {
                'error': "There is no document!!!"
            }
        elif arg_pairs is None:
            return {
                'error': "You have to pass argument pairs. But got Null argument."
            }
        else:
            doc_text = doc.get('text', '')
            doc_ln = doc.get('language', 'kr')
            doc_domain = doc.get('domain', 'common-sense')

            if doc_text == '':
                return {
                    'error': "Empty document string!!!"
                }

            doc_dict = self.convert_dict(doc_text, arg_pairs)

            comb_str = "{}_{}".format(doc_ln, doc_domain)
            if comb_str in self.extract_manager:
                result = self.extract(comb_str, doc_dict)
                return result
            else:
                return {
                    'error': f"The requested combination of language and question domain is currently unsupported. \
                        (kr, common-sense) is currently supported on this service. \
                            But got ({doc_ln},{doc_domain})"
                }


    @staticmethod
    def convert_dict(text, arg_pairs):
        re_dict = list()
        for arg in arg_pairs:
            re_dict.append({
                "sentence": text,
                "subject_entity": {
                    "word": text[arg[0][0]:arg[0][1]+1],
                    "start_idx": arg[0][0],
                    "end_idx": arg[0][1]
                },
                "object_entity": {
                    "word": text[arg[1][0]:arg[1][1]+1],
                    "start_idx": arg[1][0],
                    "end_idx": arg[1][1]
                }
            })
        return re_dict

    @staticmethod
    def re_preproc_for_classification_with_idx(
        x,
        with_feature_key=False,
        sep=' '):
        # mark span using start index of the entity
        def _mark_span_sub(text, start_idx, end_idx, mark):
            end_idx += 2
            text = text[:start_idx] + mark + text[start_idx:]
            text = text[:end_idx] + mark + text[end_idx:]
            return text

        # '*' for subejct entity '#' for object entity
        text = x["sentence"]

        text = _mark_span_sub(text,
                              x['subject_entity']['start_idx'],
                              x['subject_entity']['end_idx'],
                              '*')

        sbj_st, sbj_end = x['subject_entity']['start_idx'], x['subject_entity']['end_idx']
        obj_st, obj_end = x['object_entity']['start_idx'], x['object_entity']['end_idx']
        sbj_end += 3
        obj_end += 3
        if sbj_st < obj_st:
            obj_st += 2
            obj_end += 2
        else:
            sbj_st += 2
            sbj_end += 2

        # Compensate for 2 added "words" added in previous step
        span2st = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
        span2et = x['object_entity']['end_idx'] + 2 * (1 if x['subject_entity']['end_idx'] < x['object_entity']['end_idx'] else 0)
        text = _mark_span_sub(text, span2st, span2et, '#')

        strs_to_join = []
        if with_feature_key:
            strs_to_join.append('{}:'.format('text'))
        strs_to_join.append(text)

        ex = {}

        offset = len(sep.join(strs_to_join[:-1] +['']))
        sbj_st+=offset
        sbj_end+=offset
        obj_st+=offset
        obj_end+=offset

        ex['subject_entity'] = {
            "start_idx": sbj_st,
            "end_idx": sbj_end,
            "word": x['subject_entity']['word'],
        }
        ex['object_entity'] = {
            "start_idx": obj_st,
            "end_idx": obj_end,
            "word": x['object_entity']['word'],
        }

        joined = sep.join(strs_to_join)
        ex['inputs'] = joined

        return ex

    @staticmethod
    def tokenize_re_with_tk_idx(x, tokenizer):
        ret = {}
        inputs = x['inputs']
        input_hf = tokenizer(inputs, padding=True, truncation='longest_first', return_tensors='pt')
        input_ids = input_hf.input_ids
        attention_mask = input_hf.attention_mask

        subject_entity = x['subject_entity']
        object_entity = x['object_entity']

        subject_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                subject_entity['start_idx'],
                subject_entity['end_idx']
                )
            ]
        subject_tk_idx = [x for x in subject_tk_idx if x is not None]
        subject_tk_idx = sorted(set(subject_tk_idx))
        subject_start = subject_tk_idx[0]
        subject_end = subject_tk_idx[-1]

        object_tk_idx = [
            input_hf.char_to_token(x) for x in range(
                object_entity['start_idx'],
                object_entity['end_idx']
                )
            ]
        object_tk_idx = [x for x in object_tk_idx if x is not None]
        object_tk_idx = sorted(set(object_tk_idx))
        object_start = object_tk_idx[0]
        object_end = object_tk_idx[-1]

        subject_token_idx = torch.zeros_like(input_ids)
        object_token_idx = torch.zeros_like(input_ids)
        subject_token_idx[0, subject_start:subject_end] = 1
        object_token_idx[0, object_start:object_end] = 1

        ret['subject_token_idx'] = subject_token_idx
        ret['object_token_idx'] = object_token_idx
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask

        return ret

    @torch.no_grad()
    def extract(self, comb_str, doc_dict):
        from meta import _KLUE_RE_RELATIONS 
        predictions = []
        for doc in doc_dict:
            preproc_doc = self.re_preproc_for_classification_with_idx(doc)
            item = self.tokenize_re_with_tk_idx(preproc_doc, self.tokenizer)
            
            outputs = self.extract_manager[comb_str](
                input_ids = item["input_ids"],
                attention_mask = item["attention_mask"],
                subject_token_idx = item["subject_token_idx"],
                object_token_idx = item["object_token_idx"],
            )
            
            logits = outputs.logits
            indice = torch.argmax(logits, dim=-1).numpy()
            pred = self.extract_manager[comb_str].config.id2label[indice[0]]
            
            predictions.append(pred)
            

        return {
            "result": predictions
        }


if __name__ == "__main__":

    example_contents = [
        {
            "doc":{
                "text":"제2총군은 태평양 전쟁 말기에 일본 본토에 상륙하려는 연합군에게 대항하기 위해 설립된 일본 제국 육군의 총군이었다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [0,3],
                    [48,55]
                ]
            ]
        },
        {
            "doc":{
                "text":"동산병원과 동산의료선교복지회는 봉사 현장에서 다나 양의 안타까운 사연을 접했고, 다나 양과 어머니를 한국으로 초청해 입국부터 진료, 수술, 출국 등 전 과정을 무료로 지원하기로 했다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [25,26],
                    [0,3]
                ]
            ]
        },
        {
            "doc":{
                "text":"총계로 요한 바오로 2세와 베네딕토 16세는 10명의 예수회 추기경들을 임명하였다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [4,12],
                    [15,22]
                ]
            ]
        },
        {
            "doc":{
                "text":"배우 소이현, 인교진 씨 부부가 SBS '동상이몽2 너는 내 운명'(이하 동상이몽2)에서 하차한다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [3,5],
                    [8,10]
                ]
            ]
        },
        {
            "doc":{
                "text":"그룹 에프엑스(f(x)(x)) 멤버 엠버가 SM엔터테인먼트를 떠난다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [8,11],
                    [3,6]
                ]
            ]
        },
        {
            "doc":{
                "text":"지지율 2017년 38명의 국민의당. 안철수 대표 때 호남의 지지율이 3.4~3.5%를 벗어나지 못했습니다.",
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [21,23],
                    [15,18]
                ]
            ]
        },
        {
            "doc":{
                "text":'이후 촬영 중인 아이유 옆에서 유인나 씨가 지인과 통화를 하자 아이유는 "유인나 씨 저희 지금 촬영 중이거든요. 조금만 조용히 해주세요"라며 계속해서 장난을 치는 모습을 보였다.',
                "language":"kr",
                "domain":"common-sense"
            },
            "arg_pairs":[
                [
                    [9,11],
                    [17,19]
                ]
            ]
        },
    ]

    test_model = RelationExtraction()
    
    for ec in example_contents:
        ret = test_model.do_extract(ec)
        text = ec["doc"]["text"]
        aps = ec["arg_pairs"][0]
        print({
            "text": text,
            "subject": text[aps[0][0]: aps[0][1]+1],
            "object": text[aps[1][0]: aps[1][1]+1],
            "relation": ret["result"][0]
        })

    