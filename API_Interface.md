# IITP Fact Reasoning API 정의서

본 문서는 '비정형 텍스트를 학습하여 쟁점별 사실과 논리적 근거 추론이 가능한 인공지능 원천기술' 과제 관련 각 모듈 API 연동에 관한 내용을 정의한다.

아래 모듈들은 모두 POST 방식으로 동작한다.

## 01-Summerization-CWNU

* 웹 API 정보: `http://220.68.54.38:12341` # 수정필요(현재 내부망에서만 접근 가능)

* 입력 파라미터
  * 03-PassageRetrieval의 출력과 동일

| Key           | Value      | Explanation                   |
| ------------- | ---------- | ----------------------------- |
| question      | dict       | (required) 질문               |
| ㄴ text       | str        | 질문 문장                     |
| ㄴ language   | str        | 질문 문장의 언어              |
| ㄴ domain     | str        | 질문의 분야                   |
| retrieved_doc | list[dict] | (required) 추출된 근거 문서들 |
| ㄴ doc_id     | str        | 데이터를 구분하기 위한 고유값 |
| ㄴ text       | str        | 문서 본문                     |
| ㄴ score      | float      | 문서와 질문의 점수            |

* 예시

```json
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "retrieved_doc":[
    {
      "doc_id":"sample_id_01",
      "text":"데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
      "score":0.642
    },
    {
      "doc_id":"sample_id_02",
      "text":"킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
      "score":0.536
    },
    {
      "doc_id":"sample_id_03",
      "text":"패트릭 휴스는 오스트레일리아의 영화 감독이다. 대표작으로 영화 레드힐(2010), 익스펜더블 3(2012), 킬러의 보디가드(2017)가 있다.",
      "score":0.219
    }
    ]
}
```



## 02-KG2Text-CWNU

* 웹 API 정보: `http://220.68.54.38:12342` # 수정필요(현재 내부망에서만 접근 가능)

* 입력 파라미터
  * 06-KnowledgeMerging의 출력과 동일

| Key     | Value                    | Explanation                                               |
| ------- | ------------------------ | --------------------------------------------------------- |
| tirples | list[tuple(str,str,str)] | (required) 관계 트리플(subject, object, predicate) 리스트 |

* 예시

```json
{
  "triples":[
    ["팀 밀러","데드풀","감독"],
    ["패트릭 휴스","킬러의 보디가드","감독"],
    ["패트릭 휴스","영화 감독","직업"]
    ]
}
```



## 03-PassageRetrieval-KETI

* 웹 API 정보: `http://aircketi.iptime.org:10021/`

* 입력 파라미터

| Key               | Value | Explanation                                                  |
| ----------------- | ----- | ------------------------------------------------------------ |
| question          | dict  | (required) 질문                                              |
| ㄴ text           | str   | 질문 문장                                                    |
| ㄴ language       | str   | 질문 문장의 언어                                             |
| ㄴ domain         | str   | 질문의 분야                                                  |
| max_num_retrieved | int   | (optional) 반환할 최대 문서 수, default=10                   |
| max_hop           | int   | (optional) 최대 홉 제한, default=5                           |
| num_retrieved     | int   | (optional) 반환할 문서 수, 해당 옵션 설정 시 max_num_retrieved 옵션 무시, default=-1 |

* 예시

```json
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "max_num_retrieved":10,
  "max_hop":5,
  "num_retrieved":-1
}
```



## 04-RelationExtraction-KETI

* 웹 API 정보: `http://aircketi.iptime.org:10022/`

* 입력 파라미터

| Key       | Value                                      | Explanation                                                  |
| --------- | ------------------------------------------ | ------------------------------------------------------------ |
| doc       | str                                        | (required) 관계 추출을 수행할 문서                           |
| arg_pairs | list[tuple[tuple[int,int],tuple[int,int]]] | (optional) 관계를 알고 싶은 객체 쌍들의 문서 내 시작･종료 위치, 주로 주어･목적어 쌍 |

* 예시

```json
{
  "doc":"킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
  "arg_pairs":[
    [
      [0,7],
      [11,14]
    ],
    [
      [0,7],
      [17,29]
    ]
  ]
}
```



## 06-KnowledgeMerging-YONSEI

* 웹 API 정보: 

* 입력 파라미터
  * 04-RelationExtractoin의 출력과 동일

| Key         | Value                    | Explanation                                         |
| ----------- | ------------------------ | --------------------------------------------------- |
| question    | dict                     | (required) 질문                                     |
| ㄴ text     | str                      | 질문 문장                                           |
| ㄴ language | str                      | 질문 문장의 언어                                    |
| ㄴ domain   | str                      | 질문의 분야                                         |
| triples     | list[tuple[str,str,str]] | (required) 문서에서 추출된 Arg0, Arg1 트리플 리스트 |

* 예시

```json
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "triples":[
    ["팀 밀러","데드풀","감독"],
    ["패트릭 휴스","킬러의 보디가드","감독"],
    ["패트릭 휴스","영화 감독","직업"]
    ]
}
```



## 08-ReRanking-POSTECH

* 웹 API 정보:

* 입력 파라미터
  * 01-Summerization, 02-KG2Text의 출력과 동일

| Key                     | Value               | Explanation                               |
| ----------------------- | ------------------- | ----------------------------------------- |
| question                | dict                | (required) 질문                           |
| ㄴ text                 | str                 | 질문 문장                                 |
| ㄴ language             | str                 | 질문 문장의 언어                          |
| ㄴ domain               | str                 | 질문의 분야                               |
| answer_list             | list(str)           | (required) 질문에 대한 답변 리스트        |
| supporting_facts_list   | list[list[str,str]] | (required) 근거 문서명 및 문장번호 리스트 |
| supporting_passage_list | list[dict]          | (required) 추출된 근거 문서들             |
| ㄴ doc_id               | str                 | 데이터를 구분하기 위한 고유값             |
| ㄴ text                 | str                 | 문서 본문                                 |
| ㄴ score                | float               | 문서와 질문의 점수                        |

* 예시

```json
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
  "supporting_facts_list":[
    [
      "데드풀",
      2
    ],
    [
      "킬러의 보디가드",
      2
    ],
    [
      "데드풀",
      0
    ]
  ],
  "supporting_passage_list":[
    {
      "doc_id":"sample_id_01",
      "text":"데드풀은 2016년 공개된 미국의 영화이다. 엑스맨 영화 시리즈의 여덟 번째 영화이며, 데드풀을 주인공으로 한 작품이다. 시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다. 주인공 웨이드 윌슨 / 데드풀 역에는 전작 《엑스맨 탄생: 울버린》에서 데드풀 역이었던 라이언 레이놀즈가 그대로 역할을 이어가며, 이외에 모레나 바카링, 에드 스크라인, T. J. 밀러, 지나 카라노 등이 출연하였다. 북미에서 2016년 2월 12일 개봉했다.",
      "score":0.642
    },
    {
      "doc_id":"sample_id_02",
      "text":"킬러의 보디가드는 2017년 미국의 액션 코미디 영화로, 살인청부업자와 그의 보디가드 콤비의 이야기를 그린다. 패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.",
      "score":0.536
    },
    {
      "doc_id":"sample_id_03",
      "text":"패트릭 휴스는 오스트레일리아의 영화 감독이다. 대표작으로 영화 레드힐(2010), 익스펜더블 3(2012), 킬러의 보디가드(2017)가 있다.",
      "score":0.219
    }
    ]
}
```



