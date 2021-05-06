
import json
import random

class Service:
    task = [
        {
            'name': "passage_retrieval",
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


class DummayModel(object):
    def __init__(self):
        self.doc_list = [
                {"doc_id": "a", "score": 0.1, "text":"The Semmering railway (German: \"Semmeringbahn\" ) in Austria, which starts at Gloggnitz and leads over the Semmering to Mürzzuschlag was the first mountain railway in Europe built with a standard gauge track."},
                {"doc_id": "a", "score": 0.1, "text":"It is commonly referred to as the world's first true mountain railway, given the very difficult terrain and the considerable altitude difference that was mastered during its construction."},
                {"doc_id": "a", "score": 0.1, "text":"It is still fully functional as a part of the Southern Railway which is operated by the Austrian Federal Railways."},
                {"doc_id": "a", "score": 0.1, "text":"The Most–Moldava railway is a branch line in Czech Republic, which was originally built and operated by the Prague-Dux Railway."},
                {"doc_id": "a", "score": 0.1, "text":"The line, formerly known as the\"Teplitz Semmering Railway\" (\"Teplitzer Semmeringbahn\") runs from Most (\"Brüx\") over the Ore Mountains to Moldava (\"Moldau\") and used to have a junction with the Nossen-Moldau railway there in Saxony until 1945."},
                {"doc_id": "a", "score": 0.1, "text":"In the Czech Republic the line is known today as the \"Moldavská horská dráha\" (\"Moldau Mountain Railway\") or \"Krušnohorská železnice\" (\"Ore Mountain Railway\")."},
                {"doc_id": "a", "score": 0.1, "text":"The Southern Railway (German: \"Südbahn\" ) is a railway in Austria that runs from Vienna to Graz and the border with Slovenia at Spielfeld via Semmering and Bruck an der Mur."},
                {"doc_id": "a", "score": 0.1, "text":"It was originally built by the Austrian Southern Railway company and ran to Ljubljana and Trieste, the main seaport of the Austro-Hungarian Monarchy."},
                {"doc_id": "a", "score": 0.1, "text":"The twin-track, electrified section that runs through the current territory of Austria is owned and operated by Austrian Federal Railways (ÖBB) and is one of the major lines in the country."},
                {"doc_id": "a", "score": 0.1, "text":"The Rudyard Lake Steam Railway is a ridable miniature railway and the third railway of any gauge to run along the side of Rudyard Lake in Staffordshire."},
                {"doc_id": "a", "score": 0.1, "text":"The railway runs for 1+1/2 mi on the track bed of an old standard gauge North Staffordshire Railway line."},
                {"doc_id": "a", "score": 0.1, "text":"After the NSR line closed down, a small narrow gauge train ran on the site for two years before moving via Suffolk to Trago Mills in Devon."},
                {"doc_id": "a", "score": 0.1, "text":"The current line started in 1985 and is gauge, and operates to a timetable."},
                {"doc_id": "a", "score": 0.1, "text":"It was built by Peter Hanton of Congleton working on his own over a period of 10 years."},
                {"doc_id": "a", "score": 0.1, "text":"He sold the railway to the Rudyard Lake Steam Railway Ltd in October 2000 who have developed it since that date."}
        ]
    
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

            return self.search(query, 
                max_num_retrieved=max_num_retrieved,
                max_hop=max_hop,
                num_retrieved=num_retrieved)

    
    def search(self, query, max_num_retrieved=10, max_hop=5, num_retrieved=-1):
        ns = random.randint(1, 5)
        selected = random.sample(self.doc_list, ns)
        return {
            'num_retrieved_doc': ns,
            'retrieved_doc': selected
        }


if __name__ == "__main__":
    dummy = DummayModel()
    ret = dummy.search('text')
    print(ret)

