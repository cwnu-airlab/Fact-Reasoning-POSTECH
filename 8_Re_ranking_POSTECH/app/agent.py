from random import *
import json


class Service:
    task = [
        {
            'name': 're-ranking',
            'description': 'Re-ranking answers and supporting facts that correspond to the question.'
        }
    ]

    def __init__(self):
        self.rerank_model = RerankModel()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200

    def do(self, content):
        try:
            ret = self.rerank_model.rerank_pairs(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class RerankModel(object):
    def __init__(self):
        self.device = 'cuda'

    def rerank_pairs(self, content):
        question = content.get("question", None)
        answers = content.get("answers", None)
        supporting_facts = content.get("supporting_facts", None)

        if (question is None) or (answers is None) or (supporting_facts is None):
            return {
                'error': "invalid query"
            }

        if len(answers) != len(supporting_facts):
            return {
                'error': "(list of answer) and (list of supporting fact) should have same length for ranking"
            }

        try:
            score = []
            for answer, supporting_fact in list(zip(answers, supporting_facts)):
                score.append(self.get_score(question, answer, supporting_fact))

            content["score"] = score

            return content
        except Exception as e:
            return {'error': "{}".format(e)}

    def get_score(self, question, answer, supporting_fact):
        return random()

