import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:5001/'
URL = 'http://thor.nlp.wo.tc:12347'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
data = json.dumps(
    {
        "_id": "3d2ef",
        "question": "Allison Gross is a character in a story which applies what writing technique that introduces characters in threes?",
        "context": [("Allison Gross",
                     'Allison Gross, a hideous witch, tries to bribe the narrator to be her "leman". She combed his hair, first. When a scarlet mantle, a silk shirt with pearls, and a golden cup all fail, she blows on a horn three times, making an oath to make him regret it; then she strikes him with a silver wand, turning him into a wyrm (dragon) bound to a tree. His sister Maisry came to him to comb his hair. One day the Seelie Court came by, and a queen stroked him three times, turning him back into his proper form.'),
                    ('Rule of three (writing)',
                     'The rule of three is a writing principle that suggests that events or characters introduced in threes are more humorous, satisfying, or effective in execution of the story and engaging the reader. The reader or audience of this form of text is also thereby more likely to remember the information conveyed. This is because having three entities combines both brevity and rhythm with having the smallest amount of information to create a pattern. It makes the author or speaker appear knowledgeable while being both simple and catchy.')
                     ],
        "lang_type": "en"
    }
    )
data = json.dumps({
    "_id": "asdf",
    "question": "샤를 드골 대통령과 콘라드 아데나워 총리가 서약한 조약은 무엇인가?",
    "context": [
        ("엘리제 궁전", "엘리제 궁전은 1848년부터 프랑스 대통령의 관저이다."),
        ("엘리제 조약", "엘리제 조약은 1963년 1월 22일 파리 엘리제 궁전에서 샤를 드골 대통령과 콘라드 아데나워 총리가 체결한 프랑스와 서독 간 우호 조약이다.")
        ],
    "lang_type":"kr",
    })
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)

print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())
