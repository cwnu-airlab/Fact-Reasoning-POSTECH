import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:12341/'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
test_sentence = '19세기 후반, 작가 아서 코난 도일에 의해 탄생한 ‘명탐정’ 셜록 홈즈는 추리 소설 마니아뿐만 아니라 성장기의 청소년에게 참 많은 영향을 끼친 인물이다. 명석하게 사건을 해결하는 탐정 캐릭터를 떠올릴 때 가장 먼저 떠오르는 근대 인물 중 하나가 되었으니까. 셜록과 그의 조수이자 동료인 왓슨을 주요 등장인물로 설정한, 또는 그에서 모티브를 가져와 제작된 영화 및 드라마는 수 없이 많다.'
data = json.dumps(
    {
		'q_id':'test001',
		'supporting_facts':test_sentence}
    )
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())
