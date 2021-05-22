import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:12345/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
data = json.dumps(
    {
        "passage": [{"doc_id": "a", "score": 0.1, "text":"The Semmering railway (German: \"Semmeringbahn\" ) in Austria, which starts at Gloggnitz and leads over the Semmering to Mürzzuschlag was the first mountain railway in Europe built with a standard gauge track."},
                    {"doc_id": "b", "score": 0.1, "text":"It is commonly referred to as the world's first true mountain railway, given the very difficult terrain and the considerable altitude difference that was mastered during its construction."},
                    {"doc_id": "c", "score": 0.1, "text":"It is still fully functional as a part of the Southern Railway which is operated by the Austrian Federal Railways."}],
        "question": "what is the first mountain railway in Europe?"
    }
    )
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)

print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())