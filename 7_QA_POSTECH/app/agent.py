import json
import random
import sys 
import os 

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.utils import *

from models.NGN import *
from typing import List 

from utils import get_answer_en, get_answer_ko

class Service:
    task = [
        {
            'name': "question_answering",
            'description': 'Question & Answering module for multi-hop reasoning'
        }
    ]

    def __init__(self):
        self.mrc_model = MRCModel()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.mrc_model.answer_predict(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class MRCModel(object):
    def __init__(self):

        # Initialize argument parser for initializing parameters
        parser = default_train_parser()
        argv = json_to_argv("predict.json")
        args = parser.parse_args(argv)
        args = complete_default_train_parser(args)

        torch.cuda.set_device(0)

        model_en_path = "./pretrained_model/model_final_en.pkl"
        model_ko_path = "./pretrained_model/model_final_ko.pkl"

        # english model initilization
        model_en = NoGraphNetwork(config=args, lang_type='en')
        if model_en_path is not None:
            model_en.load_state_dict(torch.load(model_en_path),strict=False)
        model_en.to(args.device)
        model_en.eval()

        # korean model initilization 
        model_ko = NoGraphNetwork(config=args, lang_type='ko')
        if model_ko_path is not None:
            model_ko.load_state_dict(torch.load(model_ko_path),strict=False)
        model_ko.to(args.device)
        model_ko.eval()

        self.model_en = model_en
        self.model_ko = model_ko

    def check_input(self, content):
        _id = content.get("_id", None)
        question = content.get('question', None).replace("\r\n", "")
        context = content.get('context', None)
        lang_type = content.get('lang_type', None)

        if _id is None:
            return {
                'error': "Invalid id"
            }
        if context is None:
            return {
                'error': "Invalid context"
            }
        elif question is None:
            return {
                'error': "Invalid question"
            }
        elif lang_type is None:
            return {
                "error": "Invalid language type"
            }
        
        return _id, question, context, lang_type

    def answer_predict(self, content):
        _id, question, context, lang_type = self.check_input(content)

        context_texts_list = []
        if isinstance(context[0][1], List):
            # context is list(str, list(str)) format
            context_texts_list = [" ".join(single_context[1].replace("\r\n", "")) for single_context in context]
        elif isinstance(context[0][1], str):
            # context is list(str, str) format
            context_texts_list = [single_context[1].replace("\r\n", "") for single_context in context]

        flattend_context_str = " ".join(context_texts_list)
        if lang_type == 'en':
            answer, sup_sents, sup_sents_no_index, answer_confidence_score = get_answer_en(flattend_context_str, question, self.model_en)

        elif lang_type == 'kr':
            answer, sup_sents, sup_sents_no_index, answer_confidence_score = get_answer_ko(flattend_context_str, question, self.model_ko)
        

        return {
            '_id': _id,
            'question': {
                'text': question,
                'language':lang_type,
                'domain': 'common-sense'
            },
            'answer': answer,
            'supporting_fact': sup_sents_no_index,
            'score': answer_confidence_score
        }
    