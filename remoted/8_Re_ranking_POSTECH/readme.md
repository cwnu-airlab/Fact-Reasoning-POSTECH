## 8_Re_ranking_POSTECH
## Example

1. 영어, 한국어 모델 파일을 각각 다운받고, **app/** 폴더 안에 넣어주십시오.
* English model file (Common-sense): https://drive.google.com/file/d/1i5PYBuWYljAgz6nsODghyH8OhpmGox2y/view?usp=sharing
* Korean model file (Common-sense): https://drive.google.com/file/d/1FoJOukODgY1ImsfyDMMeHfE2Mvy4qqef/view?usp=sharing

2. 다음 커맨드로 도커 이미지를 빌드하고 실행시키십시오.
```bash
    docker build --tag reranking_postech . 
    docker run --gpus all --rm -d -it -p 12345:5000 --name rerank reranking_postech
```

3. test the app
```bash
    python3 test.py
```



### input 형식
질문(question) 및 여러 모듈의 answer list, supporting_fact list, score list(각 정답에 대해 각자의 모듈이 추론한 confidence score)가 들어갑니다.

supporting fact는 각 QA모듈당 하나의 문장(str)으로 concat해서 넣어주세요.

```
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "answer_list":[
    "Yes",
    "데드풀 감독과 킬러의 보디가드 감독은 같습니다.",
    "아니오."
  ],
  "supporting_facts":[
       "데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
       "킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
       "패트릭 휴스는 오스트레일리아의 영화 감독이다. 대표작으로 영화 레드힐(2010), 익스펜더블 3(2012), 킬러의 보디가드(2017)가 있다."
  ],
  "scores": [0.652, 0.57, 0.15]
}

```

### output 형식
각 answer, supporting fact pair에 대한 점수(score)가 같이 나옵니다.

```
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "answer_list":[
    "Yes",
    "데드풀 감독과 킬러의 보디가드 감독은 같습니다.",
    "아니오."
  ],
  "supporting_facts":[
       "데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
       "킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
       "패트릭 휴스는 오스트레일리아의 영화 감독이다. 대표작으로 영화 레드힐(2010), 익스펜더블 3(2012), 킬러의 보디가드(2017)가 있다."
  ]
  "scores": [0.652, 0.57, 0.15],
  "reranking_score": [0.4725750982761383, 0.17386648058891296, 0.35355842113494873]
}
```
