import json
import requests
from urllib.parse import urljoin

URL = 'http://ketiair.com:10022/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
#data = json.dumps(
#    {'doc': "what's your name"}
#    )

"""
data = json.dumps(
    {
        "doc":{
            "text":"제2총군은 태평양 전쟁 말기에 일본 본토에 상륙하려는 연합군에게 대항하기 위해 설립된 일본 제국 육군의 총군이었다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [0,3],
                [48,55]
            ]
        ]
    }
)
"""

"""
data = json.dumps(
    {
        "doc":{
            "text":"문성민은 경기대학교에 입학하여 황동일, 신영석과 함께 경기대학교의 전성기를 이끌면서 하계대회, 전국체전, 최강전 등 3관왕을 이룬다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [0,3],
                [5,8]
            ]
        ]
    }
)
"""

data = json.dumps(
    {
        "doc":{
            "text":"1938년 당시 경성기독교연합회 부위원장인 정춘수가 앞장서서 감리교 내선일체를 위해 7인 특별위원회를 조직했을 때 참가했고, 1939년 도쿄에서 조선과 일본의 감리교단 통합을 논의하는 회의가 개최되었을 때는 정춘수, 신흥우, 양주삼, 유형기 등과 함께 전권위원으로 참석했다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [121,123],
                [89,91]
            ]
        ]
    }
)

headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())
