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
            "language":"en",
            "text":"Were both of the following rock groups formed in California: Dig and Thinking Fellers Union Local 282?",
            "domain":"common-sense"
        },
        "answer_list": ["yes",
                    "Califonia",
                    "no"],
        "supporting_facts": [
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco, California, though half of its members are from Iowa.",
            "Strangers from the Universe is the fifth album by Thinking Fellers Union Local 282, released on September 12, 1994 through Matador Records. Mother of All Saints is the fourth album by Thinking Fellers Union Local 282, released as a CD and double-LP on November 13, 1992 through Matador Records.",
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco"],
        "scores": [0.652, 0.57, 0.15]
    }
)
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())

