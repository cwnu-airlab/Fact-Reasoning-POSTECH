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
    {'text': "4월 29일 서울역에서 차정원 교수님 과제 책임인 지식추론 킥오프 미팅을 가졌다. 2주안에 EMNLP에 논문을 제출해야한다는 청천벽력같은 이야기가 나왔다."}
    )
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())