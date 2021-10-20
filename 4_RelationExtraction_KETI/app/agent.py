
import json
import random
import regex as re
import numpy as np

import torch

from transformers import AutoTokenizer

from models import *
from models import load_model

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

        self.extract_manager = {
            "kr_common-sense": self.load_extract_manager(self.config["kr_common-sense"])
        }
        pre_trained_tokenizer = self.config["kr_common-sense"].get("pre_trained_tokenizer", "KETI-AIR/ke-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_tokenizer)

    @staticmethod
    def load_extract_manager(model_cfg):
        model_class = load_model(model_cfg["model_name"])
        model = model_class.from_pretrained(model_cfg["hf_path"])
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
            with_feature_key=True,
            sep=' '):
        # mark span using start index of the entity
        def _mark_span(text, span_str, span_idx, mark):
            pattern_tmpl = r'^((?:[\S\s]){N})(W)'
            pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
            pattern = pattern_tmpl.replace('W', span_str)
            return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

        # '*' for subejct entity '#' for object entity.

        text = x["sentence"]
        text = _mark_span(text, x['subject_entity']['word'],
                        x['subject_entity']['start_idx'], '*')

        sbj_st, sbj_end, sbj_form = x['subject_entity']['start_idx'], x['subject_entity']['end_idx'], x['subject_entity']['word']
        obj_st, obj_end, obj_form = x['object_entity']['start_idx'], x['object_entity']['end_idx'], x['object_entity']['word']
        sbj_end += 2
        obj_end += 2
        if sbj_st < obj_st:
            obj_st += 2
            obj_end += 2
        else:
            sbj_st += 2
            sbj_end += 2

        # Compensate for 2 added "words" added in previous step.
        span2_index = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
        text = _mark_span(text, x['object_entity']['word'], span2_index, '#')

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
    def tokenize_re_with_tk_idx(x, tokenizer, input_key='inputs'):
        ret = {}

        inputs = x[input_key]
        ret[f'{input_key}_pretokenized'] = inputs
        input_hf = tokenizer(inputs)
        input_ids = input_hf.input_ids

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

        entity_token_idx = np.array([[subject_start, subject_end], [object_start, object_end]])
        ret['entity_token_idx'] = np.expand_dims(entity_token_idx, axis=0)
        ret['inputs'] = tokenizer(inputs, return_tensors='pt').input_ids
        return ret

    @torch.no_grad()
    def extract(self, comb_str, doc_dict):
        from meta import _KLUE_RE_RELATIONS 
        result = list()
        for doc in doc_dict:
            preproc_doc = self.re_preproc_for_classification_with_idx(doc)
            inputs = self.tokenize_re_with_tk_idx(preproc_doc, self.tokenizer)
            outputs = self.extract_manager[comb_str](input_ids=inputs['inputs'], entity_token_idx=inputs['entity_token_idx'])
            label = torch.argmax(outputs['logits'], 1).numpy()[0]
            result.append({
                "subject": preproc_doc['subject_entity']['word'],
                "relation": _KLUE_RE_RELATIONS[label],
                "object": preproc_doc['object_entity']['word']
            })

        return {
            "result": result
        }


if __name__ == "__main__":

    example_content = {
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
    }

    test_model = RelationExtraction()
    ret = test_model.do_extract(example_content)
    print(ret)