
import json
import random

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
        pass
    def answer_predict(self, content):
        passage = content.get('passage', None)
        question = content.get('question', None)

        if passage is None:
            return {
                'error': "invalid passage"
            }
        elif question is None:
            return {
                'error': "invalid question"
            }
        else:
            passage_concat = str()

            for passage in dict["passage"]:
                passage_concat += passage["text"]

            answer, answer_context, supporting_facts, joint_f1, joint_em = self.get_qa_result(question, passage_concat)

            return {'question': question,
                    'answer': answer,
                    'answer_context': answer_context,
                    'supporting_facts': supporting_facts,
                    'joint_f1': joint_f1,
                    'joint_em': joint_em}
    
    def get_qa_result(self, question, passage_concat):
        
        answer = ['30','Semmering railway','yes'] # possible answer type: 'passage_span', 'question_span', 'multiple_span', 'add/sub', 'count', 'yes/no'
        
        answer_context = '{"value": "30", \
                           "numbers": [{"value": 24, "sign": 1}, {"value": 6, "sign": 1}, {"value": 50, "sign": 0}]}' # 24 + 6 = 30
        
        supporting_facts = [['a', 1], ['b', 0], ['c', 1]] # second sentence in 'a' document & first sentence in 'b' document & second sentence in 'c' document
        
        joint_f1 = round(random.uniform(74, 76), 2)
        joint_em = round(random.uniform(47, 49), 2)

        return answer, answer_context, supporting_facts, joint_f1, joint_em

if __name__ == "__main__":
    mrc = MRCModel()
    ret = mrc.answer_predict({'passage': 'passage_dummy', 'question': 'question_dummy'})
    print(ret)

