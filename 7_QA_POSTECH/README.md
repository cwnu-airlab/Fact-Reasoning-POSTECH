## 7_QA_POSTECH
## Example

1. 영어, 한국어 모델 파일을 각각 다운받고, **./app/pretrained_model/** 폴더 안에 넣어주십시오.
* English model file : https://drive.google.com/file/d/1P041yUyQHpC_eIMa4NYPW2UNEUKzrJXn/view?usp=share_link
* Korean model file : https://drive.google.com/file/d/1GA3RvEYwmA4c0CrHPxnjSb7i7-JUywDD/view?usp=share_link
  

2. 다음 커맨드로 도커 이미지를 빌드하고 실행시키십시오.
```bash
    docker build --tag qa_postech . 
    docker run --gpus all --rm -d -it -p 5001:5000 --name qa qa_postech
```

3. test the app
```bash
    python3 test.py
```


### input 형식

|Key|Value|Explanation|
|-----|----|----------|
|_id|str|데이터를 구분하기 위한 고유값|
|question|str|질문 문장|
|context|list(Tuple(str,str))|문서명(또는 doc_id) 및 문서내용|
|lang_type|str|질문 문장의 언어|

이 때 context는 list of Tuple이며, Tuple의 첫번째 원소는 doc_id(혹은 document title), 두 번째 원소는 해당 document의 text입니다.
lang_type은 질문 문장의 언어로 영어의 경우 "en", 한국어의 경우 "kr"입니다.
예시는 아래와 같습니다.

```
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

```

### output 형식
|Key|Value|Explanation|
|-----|----|-----------|
|_id|str|데이터를 구분하기 위한 고유값|
|question|dict|질문|
|ㄴ text|str|질문 문장|
|ㄴ language|str|질문 문장의 언어|
|ㄴ domain|str|질문의 분야|
|answer|str|답안|
|supporting_fact|str|근거 문장|
|score|float|answer confidence score|

예시는 다음과 같습니다.
```
{
   "_id":"3d2ef",
   "question":{
      "text":"Allison Gross is a character in a story which applies what writing technique that introduces characters in threes?",
      "language":"en",
      "domain":"common-sense"
   },
   "answer":"rule of three",
   "supporting_fact":"Allison Gross, a hideous witch, tries to bribe the narrator to be her \"leman\". The rule of three is a writing principle that suggests that events or characters introduced in threes are more humorous, satisfying, or effective in execution of the story and engaging the reader. ",
   "score":0.997473955154419
}
```
