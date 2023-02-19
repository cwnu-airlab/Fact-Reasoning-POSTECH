import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:5001/'

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
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)

print(response.status_code)
print(response.request)

print(response.json())

print(response.raise_for_status())