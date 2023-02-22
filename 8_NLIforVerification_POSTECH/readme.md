## 8_NLIforVerification_POSTECH
## Example

1. 영어, 한국어 모델 파일을 각각 다운받고, **app/** 폴더 안에 넣어주십시오.
* English model file (Common-sense): https://drive.google.com/file/d/1i5PYBuWYljAgz6nsODghyH8OhpmGox2y/view?usp=sharing
* Korean model file (Common-sense): https://drive.google.com/file/d/1FoJOukODgY1ImsfyDMMeHfE2Mvy4qqef/view?usp=sharing

2. 다음 커맨드로 도커 이미지를 빌드하고 실행시키십시오.
```bash
    docker build --tag verification_postech . 
    docker run --gpus all --rm -d -it -p 12345:5000 --name verification verification_postech
```

3. test the app
```bash
    python3 test.py
```



### input 형식
질문(question) 및 정답, supporting fact, (optional) score (정답에 대해 QA모듈이 추론한 confidence score)이 input으로 들어갑니다.
supporting fact는 하나의 문장(str)으로 concat해서 넣어주세요.

```
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "answer":"데드풀 감독과 킬러의 보디가드 감독은 같습니다.",
  "supporting_fact": "데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
  "score": 0.652
}

```

### output 형식
특정 question에 대해 QA모듈이 생성한 answer, supporting fact이 올바를지에 대한 예측 결과(correct / incorrect) 가 나옵니다.

```
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "answer":"데드풀 감독과 킬러의 보디가드 감독은 같습니다.",
  "supporting_fact": "데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
  "score": 0.652,
  "label": "correct"
}
```
