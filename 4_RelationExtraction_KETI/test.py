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

data = json.dumps(
    {
        "doc":{
            "text":"킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [0,8],
                [10,14]
            ],
            [
                [0,8],
                [16,29]
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