
import json
import random

from ner_test import T5EncoderCRFforNER

class Service:
    task = [
        {
            'name': "ner",
            'description': 'example task'
        }
    ]

    def __init__(self):
        self.ner_model = ExampleModel()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.ner_model.recog(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class ExampleModel(object):
    def __init__(self):
        self.ner_model = T5EncoderCRFforNER(model_dir='t5_encoder_ner')
    
    def recog(self, content):
        text = content.get('text', None)
        if text is None:
            return {
                'error': "invalid query"
            }
        else:
            return self.ner_model.generate(text)
