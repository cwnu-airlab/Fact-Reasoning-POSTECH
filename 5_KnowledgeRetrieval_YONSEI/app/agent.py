import json

class Service:
    task = [
        {
            'name': 'Knowledge_Retrieval', 
            'description':'dummy task'
        }
    ]

    def __init__(self):
        self.dummy = dummyModel()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.dummy.do_search(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400

class dummyModel(object):
    def __init__(self):
        self.supporting_facts = [('사업주','고용','근로자'),('사업주','허용','근로시간 단축')]

    def do_search(self, content):
        query = content.get('query', None)
        if query is None:
            return {
                'error': "invalid query"
            }
        else:
            max_num_retrieved = content.get('max_num_retrieved', 10)
            max_hop = content.get('max_hop', 5)
            num_retrieved = content.get('num_retrieved', -1)

            return self.search(query, max_num_retrieved=max_num_retrieved, max_hop=max_hop, num_retrieved=num_retrieved)
    
    def search(self, query, max_num_retrieved=10, max_hop=5,num_retrieved=-1):
        result = self.supporting_facts
        return {'supporting facts':result}
        

if __name__ == "__main__":
    model = dummyModel()
    result = model.search('question')
    print(result)