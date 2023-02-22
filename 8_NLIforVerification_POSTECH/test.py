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
        "question": {
            "text": "The telenova \"El Ardiente Secreto\" was based ona novel published under what pen name?",
            "language": "en",
            "domain": "common-sense"
        },
        "answer": "Currer Bell",
        "supporting_fact": "It was based on the Charlotte Bront\u00eb's novel \"Jane Eyre\". It was published on 16 October 1847, by Smith, Elder & Co. of London, England, under the pen name \"Currer Bell\". ",
        "score": 0.9540373086929321
    }
)
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())

