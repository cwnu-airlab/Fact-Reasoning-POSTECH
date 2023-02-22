import json
import requests
from urllib.parse import urljoin

# URL = 'http://ketiair.com:10022/'
URL = 'http://127.0.0.1:12345/'

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

"""
data = json.dumps(
    {
        "doc":{
            "text":"동산병원과 동산의료선교복지회는 봉사 현장에서 다나 양의 안타까운 사연을 접했고, 다나 양과 어머니를 한국으로 초청해 입국부터 진료, 수술, 출국 등 전 과정을 무료로 지원하기로 했다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [25,26],
                [0,3]
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

"""
data = json.dumps(
    {
        "doc":{
            "text":"총계로 요한 바오로 2세와 베네딕토 16세는 10명의 예수회 추기경들을 임명하였다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [4,12],
                [15,22]
            ]
        ]
    }
)
"""

"""
data = json.dumps(
    {
        "doc":{
            "text":"배우 소이현, 인교진 씨 부부가 SBS '동상이몽2 너는 내 운명'(이하 동상이몽2)에서 하차한다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [3,5],
                [8,10]
            ]
        ]
    }
)
"""

"""
data = json.dumps(
    {
        "doc":{
            "text":"지지율 2017년 38명의 국민의당. 안철수 대표 때 호남의 지지율이 3.4~3.5%를 벗어나지 못했습니다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [21,23],
                [15,18]
            ]
        ]
    }
)
"""

"""
data = json.dumps(
    {
        "doc":{
            "text":'이후 촬영 중인 아이유 옆에서 유인나 씨가 지인과 통화를 하자 아이유는 "유인나 씨 저희 지금 촬영 중이거든요. 조금만 조용히 해주세요"라며 계속해서 장난을 치는 모습을 보였다.',
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [9,11],
                [17,19]
            ]
        ]
    }
)
"""


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
"""

data = json.dumps(
    {
        "doc":{
            "text":"배우 소이현, 인교진 씨 부부가 SBS '동상이몽2 너는 내 운명'(이하 동상이몽2)에서 하차한다.",
            "language":"kr",
            "domain":"common-sense"
        },
        "arg_pairs":[
            [
                [3,5],
                [8,10]
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
