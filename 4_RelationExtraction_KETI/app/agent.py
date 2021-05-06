
import json
import random

class Service:
    task = [
        {
            'name': "relation_extraction",
            'description': 'dummy task'
        }
    ]

    def __init__(self):
        self.dummy = DummayModel()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.dummy.extract(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class DummayModel(object):
    def __init__(self):
        self.tuple_list = [
                ("bird", "location of bird", "a tree"),
                ("bird", "location of bird", "a nest"),
                ("bird", "location of bird", "the sky"),
                ("bird", "location of bird", "a cage"),
                ("bird", "location of bird", "the forest"),
                ("bird", "location of bird", "a roof"),
                ("bird", "location of bird", "a branch of a tree"),
                ("bird", "is capable of", "sing"),
                ("bird", "is capable of", "spread the wings"),
                ("bird", "is capable of", "fly"),
                ("bird", "is capable of", "build a nest"),
                ("bird", "is capable of", "fly"),
                ("bird", "is capable of", "land on a branch"),
        ]
    
    def extract(self, content):
        doc = content.get('doc', None)
        if doc is None:
            return {
                'error': "invalid doc"
            }
        else:
            arg_pairs = content.get('arg_pairs', [])

            return self._extract(doc, 
                arg_pairs=arg_pairs)

    
    def _extract(self, doc, arg_pairs=[]):
        ns = random.randint(1, 5)
        selected = random.sample(self.tuple_list, ns)
        return {
            'num_of_triples': ns,
            'triples': selected
        }


if __name__ == "__main__":
    dummy = DummayModel()
    ret = dummy._extract('text')
    print(ret)

