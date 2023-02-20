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

* 웹 API 정보: `http://ketiair.com:10021/`

* 입력 파라미터

| Key               | Value | Explanation                                                  |
| ----------------- | ----- | ------------------------------------------------------------ |
| question          | dict  | (required) 질문                                              |
| ㄴ text           | str   | 질문 문장                                                    |
| ㄴ language       | str   | 질문 문장의 언어                                             |
| ㄴ domain         | str   | 질문의 분야                                                  |
| context | list[ list[ str, str ]]   | 컨텍스트 ([[title1, text1], [title2, text2], ...])                  |
| max_num_retrieved | int   | (optional) 반환할 최대 문서 수, default=10                   |
| max_hop           | int   | (optional) 최대 홉 제한, default=2                           |
| num_retrieved     | int   | (optional) 반환할 문서 수, 해당 옵션 설정 시 max_num_retrieved 옵션 무시, default=-1 |

<details>
<summary> * 입력 예시 (ko_common-sense) (click)</summary>
<div markdown="1">

```python
{
    'question': {
        'text': '2013년 오클랜드 레이더스의 새 쿼터백은 몇 년도에 태어났습니까?',
                'language': 'kr',
                'domain': 'common-sense',
    },
    'max_num_retrieved': 6,
    'context': [['앨런 밀러 (미식축구)',
                 ['앨런 밀러(1937년 6월 19일 ~ )는 전 대학 미식축구 풀백이다.',
                  '그는 보스턴 칼리지에서 대학 미식축구를 했다.',
                  '보스턴 칼리지에 있을 때, 밀러는 1959년에 올 이스트와 올 뉴잉글랜드 팀의 일원이었고 1958년과 '
                  '1959년에는 카톨릭 올 아메리칸 팀의 일원이었다.',
                  '밀러는 1959년 오멜리아 트로피 수상자로 뽑혔고 1960년 앨라배마주 모빌에서 열린 시니어볼 올스타 게임에서 '
                  '노스 스쿼드의 일원이었다.',
                  '많은 BC 선수들처럼, 그는 1960년 창단 첫 해에 아메리칸 풋볼 리그의 보스턴 패트리어츠와 프로 계약을 '
                  '맺었다.',
                  '밀러는 1960년에 패트리어츠의 선두 러셔였다.',
                  '1961년, 그는 AFL의 오클랜드 레이더스로 트레이드되었고, 1961년 AFL 올스타였다.',
                  '그는 1965년까지 레이더스 팀에서 뛰었다.',
                  '밀러는 1961년 AFL 올스타 팀, 1963-65년 오클랜드 레이더스 주장, 1965년 최우수 선수상을 '
                  '수상했다.']],
                ['데이비스 산 (오클랜드 주)',
                 ['마운트 데이비스(Mount Davis)는 미국 캘리포니아주 오클랜드에 있는 오클랜드-앨러메다 카운티 콜리세움의 '
                  '수용 인원 20,000명의 구역입니다.',
                  '이는 1995년 오클랜드 시의회의 요청으로 지어졌으며, LA 레이더스 미식축구팀을 오클랜드로 다시 데려오기 위한 '
                  '목적으로 지어졌으며, 전 오클랜드 레이더스 구단주 알 데이비스의 이름을 따서 명명되었다.',
                  '2006년부터 마운트 데이비스의 최상급 좌석은 오클랜드 애슬레틱스 야구 경기 내내 방수포로 덮여 있었다.',
                  '2013년부터는 오클랜드 레이더스 전 경기에서도 톱티어 좌석이 타프로 가려졌다.']],
                ['코너 쿡',
                 ['코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 리그 오클랜드 레이더스의 '
                  '쿼터백이다.',
                  '2013년부터 2015년까지 미시간주 스파탄스에서 대학 미식축구 선수로 활약하며 주전 쿼터백으로 활약했다.',
                  '그는 미시간 주에서 통산 최다 우승 기록을 보유하고 있다.',
                  '쿡은 2016년 NFL 드래프트 4라운드에서 오클랜드 레이더스에 선발됐다.',
                  '쿡은 당초 데릭 카와 맷 맥글린의 3군 백업으로 활약한 뒤 2016년 미국프로축구연맹(NFL) 마지막 정규시즌 '
                  '레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 NFL 경기에 출전했다.',
                  '이어 휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 지명돼 NFL 역사상 첫 쿼터백으로 플레이오프 첫 '
                  '선발 등판했다.']],
                ['2017 오클랜드 해적 시즌',
                 ['2017 Oakland Raiders 시즌은 오클랜드 램프 프랜차이즈 (Oakland Raiders '
                  'Franchise)의 58 번째 전반적인 시즌, 프랜차이즈 (Franchise)의 48 번째 시즌, 오클랜드로 '
                  '돌아온 것부터 24 시즌, 잭 델 리오 (Jack Del Rio)',
                  '해적들은 1983 년부터 처음으로 클럽이 아직 로스 앤젤레스에 있었을 때 처음으로 첫 번째 AFC West '
                  'Title을 획득하고 있습니다.',
                  '해적들은 테네시 타이탄에서 9 월 10 일에 시즌을 시작했으며, 로스 앤젤레스 충전기에서 12 월 31 일 '
                  '시즌을 마쳤습니다.',
                  '그들이 2016 년에했던 것처럼 멕시코 시티에서 한 홈즈 게임을 할 것입니다.']],
                ['오클랜드 레이더스 감독 목록',
                 ['내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다.',
                  '라이더스 프랜차이즈는 1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 NFL로 이적한 미네소타 '
                  '바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 8번째 회원이 됐다.',
                  'Raiders는 AFL-NFL 합병 이후 1970년에 NFL에 합류했다.',
                  '그들은 오클랜드로 돌아가기 전 1982년과 1995년 사이에 로스앤젤레스에서 뛰었다.',
                  '라이더스는 2015시즌 말 현재 AFL과 NFL에서 총 56시즌 동안 852경기를 뛰었다.',
                  '이 경기에서 두 명의 코치가 팀과 함께 슈퍼볼 우승을 차지했다. 1976년 존 매든, 1980년과 1983년 톰 '
                  '플로레스.',
                  '1966년 존 라우치라는 한 코치가 AFL 챔피언십에서 우승했다.',
                  '다른 세 감독인 아트 셸, 존 그루든, 빌 캘러한도 레이더스를 플레이오프에 진출시켰다.',
                  '캘러한은 레이더스를 슈퍼볼로 이끌었다.',
                  '그는 감독으로서 첫 해에 이 일을 했다.']],
                ['테리 쿤즈',
                 ['테리 팀 쿤즈(Terry Tim Kunz, 1952년 10월 26일 ~ )는 전 미식축구 러닝백으로, 내셔널 '
                  '풋볼 리그의 오클랜드 레이더스 소속으로 한 시즌 뛰었다.',
                  '그는 1976년 NFL 드래프트 8라운드에서 오클랜드 레이더스에 의해 드래프트 되었다.',
                  '쿤즈는 콜로라도 볼더 대학교에서 대학 미식축구를 했고 콜로라도 휘트리지에 있는 휘트리지 고등학교에 다녔다.',
                  '그는 슈퍼볼 XI를 우승한 오클랜드 레이더스 팀의 일원이었다.']],
                ['2013년 오클랜드 레이더스 시즌입니다.',
                 ['오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, 그리고 데니스 앨런 감독 밑에서 '
                  '두 번째였습니다.',
                  '4승 12패의 기록으로, 레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 진출하지 '
                  '못했습니다.',
                  '레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 맞이했습니다.',
                  '프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, '
                  '2주차에는 잭슨빌 재규어스를 물리쳤습니다.',
                  '결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 프라이어가 매트 맥글로인 대신 벤치를 차지하게 '
                  '되었습니다.',
                  '이 경기 전에, 이글스 쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 가장 '
                  '많이 허용했던 터치다운 패스였어요.']],
                ['테렐 프라이어',
                 ['테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 ~ )는 미식축구 내셔널 풋볼 '
                  '리그(NFL) 워싱턴 레드스킨스의 와이드 리시버이다.']],
                ['조 버겔',
                 ['조셉 존 버겔(Joseph John Bugel, 1940년 3월 10일 ~ )은 전 미식축구 코치이자 대학 '
                  '선수로 46년 동안 대학과 프로 스포츠에서 다양한 코치직을 역임했다.',
                  '내셔널 풋볼 리그(NFL)에서 두 차례 사령탑을 맡았지만 1981년부터 1989년까지, 2004년부터 '
                  '2009년까지 워싱턴 레드스킨스에서 가장 주목받는 NFL 역사상 최고의 공격 라인 코치 중 하나로 널리 인정받고 '
                  '있다.',
                  '그는 디트로이트 라이온스 1975–76, 휴스턴 오일러스 1977–80, 워싱턴 레드스킨스 1981–89, '
                  '오클랜드 레이더스 1995–96, 샌디에이고 차저스 1998–2001 및 레드스킨스의 공격 라인 코치 또는 '
                  '어시스턴트를 2004년부터 다시 역임했습니다.',
                  '그는 또한 피닉스 카디널스(1990–93)와 오클랜드 레이더스(1997)의 감독을 역임했다.',
                  '부겔은 피닉스 카디널스와 오클랜드 레이더스의 감독으로 5시즌 동안 56패 24승 기록을 세웠다.',
                  '그는 1982년 레드스킨스의 훈련 캠프에서 공격 라인 유닛으로 불렸던 별명인 "The Hogs"를 만든 것으로 '
                  '가장 잘 알려져 있다.',
                  '부겔은 "보스 호그"라는 별명으로 알려져 있다.']],
                ['1980년 오클랜드 레이더스 시즌',
                 ['1980년 오클랜드 레이더스 시즌은 팀이 1979년부터 9승 7패의 성적을 올리려고 노력하면서 시작되었다.',
                  '오클랜드 레이더스 프랜차이즈의 20주년 기념일이었고 그들의 두 번째 슈퍼볼 우승으로 끝이 났다.',
                  '시즌이 시작되기 전에 알 데이비스는 레이더스를 오클랜드에서 로스앤젤레스로 이전할 계획을 발표했다.',
                  '그러나 피트 로젤 NFL 커미셔너는 접근금지 명령을 내려 이를 저지했다.',
                  '그는 심지어 알 데이비스를 주인으로 내보내려고도 했다. 이 사건이 법정으로 넘어갔기 때문이다.',
                  '오클랜드에서 뛰고 있는 레이더스는 케니 스태블러에게 휴스턴 오일러스로부터 댄 파스토리니를 인수한 후 새로운 '
                  '쿼터백으로 시즌을 맞이했다.',
                  '그러나 파스토리니가 부상을 입고 짐 플런켓이 교체되면서 파스토리니가 고전했고 레이더스 팀은 2-3으로 뒤졌다.',
                  '플런켓이 레이더스 공격에 적중했음을 증명했습니다.',
                  '수비진은 가로채기(35개), 턴오버(52개), 캐리당 야드(3.4YPA) 등에서 리그 선두를 달렸다.',
                  '레스터 헤이즈는 13번의 가로채기로 NFL을 이끌었다.',
                  '이 팀은 11승 5패로 6연승을 거두며 와일드카드로 플레이오프에 진출했다.',
                  '와일드 카드 게임에서 라이더스는 오클랜드에서 휴스턴 오일러스를 27-7로 꺾었다. 라이더스의 수비가 전 팀 '
                  '동료인 케니 스테이블러를 두 번 따돌렸기 때문이다.',
                  '기온이 영하 30도를 기록하는 혹한의 날씨 속에서 경기를 하던 레이더스는 클리블랜드에서 벌어진 수비전에서 '
                  '브라운스를 14-12로 대파했다.',
                  '샌디에이고에서 열리는 AFC 챔피언십 게임에서는 레이더스가 차저스를 34-27로 꺾고 AFC 와일드카드로서는 '
                  '처음으로 슈퍼볼에 진출하게 되면서 승부차기가 될 것이다.',
                  '짐 플런켓의 MVP 성적과 로드 마틴의 3개의 가로채기로 두드러진 레이더스는 슈퍼볼 15에서 필라델피아 이글스를 '
                  '27대 10으로 물리쳤다.']]],
}
```
</div>
</details>

<details>
<summary> * 출력 예시 (ko_common-sense) (click)</summary>
<div markdown="1">

```python
{'num_retrieved_doc': 4,
 'retrieved_doc': [{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 리그 '
                            '오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 스파탄스에서 대학 '
                            '미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 주에서 통산 최다 우승 '
                            '기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 4라운드에서 오클랜드 레이더스에 '
                            '선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 백업으로 활약한 뒤 2016년 '
                            '미국프로축구연맹(NFL) 마지막 정규시즌 레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 '
                            'NFL 경기에 출전했다. 이어 휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 '
                            '지명돼 NFL 역사상 첫 쿼터백으로 플레이오프 첫 선발 등판했다.',
                    'title': '코너 쿡'},
                   {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, 그리고 '
                            '데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, 레이더스는 11년 '
                            '연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 진출하지 못했습니다. '
                            '레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 맞이했습니다. 프라이어는 인상적인 '
                            '패션으로 시즌을 시작했고, 인디애나폴리스 콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, '
                            '2주차에는 잭슨빌 재규어스를 물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 '
                            '텍산스와의 경기에서 프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 '
                            '전에, 이글스 쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 '
                            '역사상 가장 많이 허용했던 터치다운 패스였어요.',
                    'title': '2013년 오클랜드 레이더스 시즌입니다.'},
                   {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 ~ )는 '
                            '미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 리시버이다.',
                    'title': '테렐 프라이어'},
                   {'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 프랜차이즈는 '
                            '1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 NFL로 이적한 미네소타 '
                            '바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 8번째 회원이 됐다. '
                            'Raiders는 AFL-NFL 합병 이후 1970년에 NFL에 합류했다. 그들은 '
                            '오클랜드로 돌아가기 전 1982년과 1995년 사이에 로스앤젤레스에서 뛰었다. 라이더스는 '
                            '2015시즌 말 현재 AFL과 NFL에서 총 56시즌 동안 852경기를 뛰었다. 이 '
                            '경기에서 두 명의 코치가 팀과 함께 슈퍼볼 우승을 차지했다. 1976년 존 매든, '
                            '1980년과 1983년 톰 플로레스. 1966년 존 라우치라는 한 코치가 AFL '
                            '챔피언십에서 우승했다. 다른 세 감독인 아트 셸, 존 그루든, 빌 캘러한도 레이더스를 '
                            '플레이오프에 진출시켰다. 캘러한은 레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 '
                            '이 일을 했다.',
                    'title': '오클랜드 레이더스 감독 목록'}],
 'top_n_candidates': [[{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 '
                                '리그 오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 '
                                '스파탄스에서 대학 미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 '
                                '주에서 통산 최다 우승 기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 '
                                '4라운드에서 오클랜드 레이더스에 선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 '
                                '백업으로 활약한 뒤 2016년 미국프로축구연맹(NFL) 마지막 정규시즌 '
                                '레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 NFL 경기에 출전했다. 이어 '
                                '휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 지명돼 NFL 역사상 첫 '
                                '쿼터백으로 플레이오프 첫 선발 등판했다.',
                        'title': '코너 쿡'},
                       {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
                                '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
                                '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
                                '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
                                '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
                                '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
                                '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
                                '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
                                '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
                                '가장 많이 허용했던 터치다운 패스였어요.',
                        'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
                      [{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 '
                                '리그 오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 '
                                '스파탄스에서 대학 미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 '
                                '주에서 통산 최다 우승 기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 '
                                '4라운드에서 오클랜드 레이더스에 선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 '
                                '백업으로 활약한 뒤 2016년 미국프로축구연맹(NFL) 마지막 정규시즌 '
                                '레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 NFL 경기에 출전했다. 이어 '
                                '휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 지명돼 NFL 역사상 첫 '
                                '쿼터백으로 플레이오프 첫 선발 등판했다.',
                        'title': '코너 쿡'},
                       {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
                                '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
                                '리시버이다.',
                        'title': '테렐 프라이어'}],
                      [{'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
                                '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
                                '리시버이다.',
                        'title': '테렐 프라이어'},
                       {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
                                '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
                                '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
                                '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
                                '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
                                '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
                                '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
                                '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
                                '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
                                '가장 많이 허용했던 터치다운 패스였어요.',
                        'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
                      [{'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 '
                                '프랜차이즈는 1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 '
                                'NFL로 이적한 미네소타 바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 '
                                '8번째 회원이 됐다. Raiders는 AFL-NFL 합병 이후 1970년에 '
                                'NFL에 합류했다. 그들은 오클랜드로 돌아가기 전 1982년과 1995년 사이에 '
                                '로스앤젤레스에서 뛰었다. 라이더스는 2015시즌 말 현재 AFL과 NFL에서 총 '
                                '56시즌 동안 852경기를 뛰었다. 이 경기에서 두 명의 코치가 팀과 함께 슈퍼볼 '
                                '우승을 차지했다. 1976년 존 매든, 1980년과 1983년 톰 플로레스. '
                                '1966년 존 라우치라는 한 코치가 AFL 챔피언십에서 우승했다. 다른 세 감독인 '
                                '아트 셸, 존 그루든, 빌 캘러한도 레이더스를 플레이오프에 진출시켰다. 캘러한은 '
                                '레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 이 일을 했다.',
                        'title': '오클랜드 레이더스 감독 목록'},
                       {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
                                '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
                                '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
                                '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
                                '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
                                '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
                                '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
                                '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
                                '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
                                '가장 많이 허용했던 터치다운 패스였어요.',
                        'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
                      [{'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
                                '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
                                '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
                                '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
                                '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
                                '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
                                '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
                                '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
                                '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
                                '가장 많이 허용했던 터치다운 패스였어요.',
                        'title': '2013년 오클랜드 레이더스 시즌입니다.'},
                       {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
                                '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
                                '리시버이다.',
                        'title': '테렐 프라이어'}],
                      [{'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 '
                                '프랜차이즈는 1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 '
                                'NFL로 이적한 미네소타 바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 '
                                '8번째 회원이 됐다. Raiders는 AFL-NFL 합병 이후 1970년에 '
                                'NFL에 합류했다. 그들은 오클랜드로 돌아가기 전 1982년과 1995년 사이에 '
                                '로스앤젤레스에서 뛰었다. 라이더스는 2015시즌 말 현재 AFL과 NFL에서 총 '
                                '56시즌 동안 852경기를 뛰었다. 이 경기에서 두 명의 코치가 팀과 함께 슈퍼볼 '
                                '우승을 차지했다. 1976년 존 매든, 1980년과 1983년 톰 플로레스. '
                                '1966년 존 라우치라는 한 코치가 AFL 챔피언십에서 우승했다. 다른 세 감독인 '
                                '아트 셸, 존 그루든, 빌 캘러한도 레이더스를 플레이오프에 진출시켰다. 캘러한은 '
                                '레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 이 일을 했다.',
                        'title': '오클랜드 레이더스 감독 목록'},
                       {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
                                '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
                                '리시버이다.',
                        'title': '테렐 프라이어'}]]}
```
</div>
</details>

<details>
<summary> * 입력 예시 (en_common-sense) (click)</summary>
<div markdown="1">

```python
{
    'question': {
        "text": 'Were Scott Derrickson and Ed Wood of the same nationality?',
        'language': 'en',
        'domain': 'common-sense',
    },
    'max_num_retrieved': 6,
    'context': [
        [
            'Ed Wood (film)',
            [
                'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.',
                " The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.",
                ' Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
            ]
        ],
        [
            'Scott Derrickson',
            [
                'Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.',
                ' He lives in Los Angeles, California.',
                ' He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."'
            ]
        ],
        [
            'Woodson, Arkansas',
            [
                'Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.',
                ' Its population was 403 at the 2010 census.',
                ' It is part of the Little Rock–North Little Rock–Conway Metropolitan Statistical Area.',
                ' Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.',
                ' Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr.'
            ]
        ],
        [
            'Tyler Bates',
            [
                'Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.',
                ' Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."',
                ' He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.',
                ' With Gunn, he has scored every one of the director\'s films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.',
                ' In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".'
            ]
        ],
        [
            'Ed Wood',
            [
                'Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.'
            ]
        ],
        [
            'Deliver Us from Evil (2014 film)',
            [
                'Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.',
                ' The film is officially based on a 2001 non-fiction book entitled "Beware the Night" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was "inspired by actual accounts".',
                ' The film stars Eric Bana, Édgar Ramírez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.'
            ]
        ],
        [
            'Adam Collis',
            [
                'Adam Collis is an American filmmaker and actor.',
                ' He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.',
                ' He also studied cinema at the University of Southern California from 1991 to 1997.',
                ' Collis first work was the assistant director for the Scott Derrickson\'s short "Love in the Ruins" (1995).',
                ' In 1998, he played "Crankshaft" in Eric Koyanagi\'s "Hundred Percent".'
            ]
        ],
        [
            'Sinister (film)',
            [
                'Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.',
                ' It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.'
            ]
        ],
        [
            'Conrad Brooks',
            [
                'Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.',
                ' He moved to Hollywood, California in 1948 to pursue a career in acting.',
                ' He got his start in movies appearing in Ed Wood films such as "Plan 9 from Outer Space", "Glen or Glenda", and "Jail Bait."',
                ' He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.',
                ' He also has since gone on to write, produce and direct several films.'
            ]
        ],
        [
            'Doctor Strange (2016 film)',
            [
                'Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.',
                ' It is the fourteenth film of the Marvel Cinematic Universe (MCU).',
                ' The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.',
                ' In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.'
            ]
        ]
    ],
}
```
</div>
</details>

<details>
<summary> * 출력 예시 (en_common-sense) (click)</summary>
<div markdown="1">

```python
{'num_retrieved_doc': 6,
 'retrieved_doc': [{'text': 'Scott Derrickson (born July 16, 1966) is an '
                            'American director, screenwriter and producer.  He '
                            'lives in Los Angeles, California.  He is best '
                            'known for directing horror films such as '
                            '"Sinister", "The Exorcism of Emily Rose", and '
                            '"Deliver Us From Evil", as well as the 2016 '
                            'Marvel Cinematic Universe installment, "Doctor '
                            'Strange."',
                    'title': 'Scott Derrickson'},
                   {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
                            'December 10, 1978) was an American filmmaker, '
                            'actor, writer, producer, and director.',
                    'title': 'Ed Wood'},
                   {'text': 'Adam Collis is an American filmmaker and actor.  '
                            'He attended the Duke University from 1986 to 1990 '
                            'and the University of California, Los Angeles '
                            'from 2007 to 2010.  He also studied cinema at the '
                            'University of Southern California from 1991 to '
                            '1997.  Collis first work was the assistant '
                            'director for the Scott Derrickson\'s short "Love '
                            'in the Ruins" (1995).  In 1998, he played '
                            '"Crankshaft" in Eric Koyanagi\'s "Hundred '
                            'Percent".',
                    'title': 'Adam Collis'},
                   {'text': 'Conrad Brooks (born Conrad Biedrzycki on January '
                            '3, 1931 in Baltimore, Maryland) is an American '
                            'actor.  He moved to Hollywood, California in 1948 '
                            'to pursue a career in acting.  He got his start '
                            'in movies appearing in Ed Wood films such as '
                            '"Plan 9 from Outer Space", "Glen or Glenda", and '
                            '"Jail Bait."  He took a break from acting during '
                            'the 1960s and 1970s but due to the ongoing '
                            'interest in the films of Ed Wood, he reemerged in '
                            'the 1980s and has become a prolific actor.  He '
                            'also has since gone on to write, produce and '
                            'direct several films.',
                    'title': 'Conrad Brooks'},
                   {'text': 'Sinister is a 2012 supernatural horror film '
                            'directed by Scott Derrickson and written by '
                            'Derrickson and C. Robert Cargill.  It stars Ethan '
                            'Hawke as fictional true-crime writer Ellison '
                            'Oswalt who discovers a box of home movies in his '
                            'attic that puts his family in danger.',
                    'title': 'Sinister (film)'},
                   {'text': 'Deliver Us from Evil is a 2014 American '
                            'supernatural horror film directed by Scott '
                            'Derrickson and produced by Jerry Bruckheimer.  '
                            'The film is officially based on a 2001 '
                            'non-fiction book entitled "Beware the Night" by '
                            'Ralph Sarchie and Lisa Collier Cool, and its '
                            'marketing campaign highlighted that it was '
                            '"inspired by actual accounts".  The film stars '
                            'Eric Bana, Édgar Ramírez, Sean Harris, Olivia '
                            'Munn, and Joel McHale in the main roles and was '
                            'released on July 2, 2014.',
                    'title': 'Deliver Us from Evil (2014 film)'}],
 'top_n_candidates': [[{'text': 'Scott Derrickson (born July 16, 1966) is an '
                                'American director, screenwriter and '
                                'producer.  He lives in Los Angeles, '
                                'California.  He is best known for directing '
                                'horror films such as "Sinister", "The '
                                'Exorcism of Emily Rose", and "Deliver Us From '
                                'Evil", as well as the 2016 Marvel Cinematic '
                                'Universe installment, "Doctor Strange."',
                        'title': 'Scott Derrickson'},
                       {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
                                'December 10, 1978) was an American filmmaker, '
                                'actor, writer, producer, and director.',
                        'title': 'Ed Wood'}],
                      [{'text': 'Adam Collis is an American filmmaker and '
                                'actor.  He attended the Duke University from '
                                '1986 to 1990 and the University of '
                                'California, Los Angeles from 2007 to 2010.  '
                                'He also studied cinema at the University of '
                                'Southern California from 1991 to 1997.  '
                                'Collis first work was the assistant director '
                                'for the Scott Derrickson\'s short "Love in '
                                'the Ruins" (1995).  In 1998, he played '
                                '"Crankshaft" in Eric Koyanagi\'s "Hundred '
                                'Percent".',
                        'title': 'Adam Collis'},
                       {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
                                'December 10, 1978) was an American filmmaker, '
                                'actor, writer, producer, and director.',
                        'title': 'Ed Wood'}],
                      [{'text': 'Scott Derrickson (born July 16, 1966) is an '
                                'American director, screenwriter and '
                                'producer.  He lives in Los Angeles, '
                                'California.  He is best known for directing '
                                'horror films such as "Sinister", "The '
                                'Exorcism of Emily Rose", and "Deliver Us From '
                                'Evil", as well as the 2016 Marvel Cinematic '
                                'Universe installment, "Doctor Strange."',
                        'title': 'Scott Derrickson'},
                       {'text': 'Conrad Brooks (born Conrad Biedrzycki on '
                                'January 3, 1931 in Baltimore, Maryland) is an '
                                'American actor.  He moved to Hollywood, '
                                'California in 1948 to pursue a career in '
                                'acting.  He got his start in movies appearing '
                                'in Ed Wood films such as "Plan 9 from Outer '
                                'Space", "Glen or Glenda", and "Jail Bait."  '
                                'He took a break from acting during the 1960s '
                                'and 1970s but due to the ongoing interest in '
                                'the films of Ed Wood, he reemerged in the '
                                '1980s and has become a prolific actor.  He '
                                'also has since gone on to write, produce and '
                                'direct several films.',
                        'title': 'Conrad Brooks'}],
                      [{'text': 'Adam Collis is an American filmmaker and '
                                'actor.  He attended the Duke University from '
                                '1986 to 1990 and the University of '
                                'California, Los Angeles from 2007 to 2010.  '
                                'He also studied cinema at the University of '
                                'Southern California from 1991 to 1997.  '
                                'Collis first work was the assistant director '
                                'for the Scott Derrickson\'s short "Love in '
                                'the Ruins" (1995).  In 1998, he played '
                                '"Crankshaft" in Eric Koyanagi\'s "Hundred '
                                'Percent".',
                        'title': 'Adam Collis'},
                       {'text': 'Conrad Brooks (born Conrad Biedrzycki on '
                                'January 3, 1931 in Baltimore, Maryland) is an '
                                'American actor.  He moved to Hollywood, '
                                'California in 1948 to pursue a career in '
                                'acting.  He got his start in movies appearing '
                                'in Ed Wood films such as "Plan 9 from Outer '
                                'Space", "Glen or Glenda", and "Jail Bait."  '
                                'He took a break from acting during the 1960s '
                                'and 1970s but due to the ongoing interest in '
                                'the films of Ed Wood, he reemerged in the '
                                '1980s and has become a prolific actor.  He '
                                'also has since gone on to write, produce and '
                                'direct several films.',
                        'title': 'Conrad Brooks'}],
                      [{'text': 'Sinister is a 2012 supernatural horror film '
                                'directed by Scott Derrickson and written by '
                                'Derrickson and C. Robert Cargill.  It stars '
                                'Ethan Hawke as fictional true-crime writer '
                                'Ellison Oswalt who discovers a box of home '
                                'movies in his attic that puts his family in '
                                'danger.',
                        'title': 'Sinister (film)'},
                       {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
                                'December 10, 1978) was an American filmmaker, '
                                'actor, writer, producer, and director.',
                        'title': 'Ed Wood'}],
                      [{'text': 'Deliver Us from Evil is a 2014 American '
                                'supernatural horror film directed by Scott '
                                'Derrickson and produced by Jerry '
                                'Bruckheimer.  The film is officially based on '
                                'a 2001 non-fiction book entitled "Beware the '
                                'Night" by Ralph Sarchie and Lisa Collier '
                                'Cool, and its marketing campaign highlighted '
                                'that it was "inspired by actual accounts".  '
                                'The film stars Eric Bana, Édgar Ramírez, Sean '
                                'Harris, Olivia Munn, and Joel McHale in the '
                                'main roles and was released on July 2, 2014.',
                        'title': 'Deliver Us from Evil (2014 film)'},
                       {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
                                'December 10, 1978) was an American filmmaker, '
                                'actor, writer, producer, and director.',
                        'title': 'Ed Wood'}]]}
```
</div>
</details>


<details>
<summary> * 입력 예시 (kr_law) (click)</summary>
<div markdown="1">

```python
{
    'question': {
        "text": '연소근로자의 동의나 노동부 장관의 인가없이 야간이나 휴일근로할 할 경우 벌칙이 있나요',
        'language': 'kr',
        'domain': 'law',
    },
    'max_num_retrieved': 6,
    'context': [
        [
            '근로기준법 0113조',
            [
                '제113조(벌칙) 제45조를 위반한 자는 1천만원 이하의 벌금에 처한다.'
            ]
        ],
        [
            '목재의 지속가능한 이용에 관한 법률 0004조',
            [
                '제4조(책무)\n\t① 국가 및 지방자치단체는 목재문화의 진흥과 목재교육의 활성화 및 목재제품의 체계적·안정적 공급에 필요한 시책을 수립·시행하여 목재의 지속가능한 이용이 증진되도록 노력하여야 한다. <개정 2017.3.21>\n\t② 산림청장은 국내 또는 원산국의 목재수확 관계 법령을 준수하여 생산(이하 "합법벌채"라 한다)된 목재 또는 목재제품이 유통·이용될 수 있도록 필요한 시책을 수립·시행하여야 한다. <신설 2017.3.21>\n\t③ 목재생산업자는 합법벌채된 목재 또는 목재제품이 수입·유통 및 생산·판매되도록 노력하여야 한다. <신설 2017.3.21>\n'
            ]
        ],
        [
            '공직자윤리법의 시행에 관한 헌법재판소 규칙 0009조',
            [
                '제9조의4(재산형성과정 소명 요구 등)\n\t① 위원회는 등록의무자가 다음 각 호의 어느 하나에 해당하는 경우에는 법 제8조제13항에 따라 재산형성과정의 소명을 요구할 수 있다. <개정 2020.10.13>\t\t1. 직무와 관련하여 부정한 재산증식을 의심할 만한 상당한 사유가 있는 경우\n\t\t2. 법 제8조의2제6항에 따른 다른 법령을 위반하여 부정하게 재물 또는 재산상 이익을 얻었다는 혐의를 입증하기 위한 경우\n\t\t3. 재산상의 문제로 사회적 물의를 일으킨 경우\n\t\t4. 등록의무자의 보수 수준 등을 고려할 때 특별한 사유 없이 재산의 뚜렷한 증감이 있는 경우\n\t\t5. 제1호부터 제4호까지의 규정에 상당하는 사유로 위원회가 소명 요구를 의결한 경우\n\n\t② 재산형성과정의 소명을 요구받은 사람은 특별한 사유가 없으면 요구받은 날부터 20일 이내에 별지 제3호의5서식의 소명서 및 증빙자료를 위원회에 제출하여야 한다. <개정 2020.10.13>\n\t③ 재산형성과정의 소명을 요구받은 사람은 분실ㆍ멸실 및 훼손 등의 사유로 증빙자료를 제출할 수 없는 경우에는 위원회에 그 사실을 소명하고, 거래시기ㆍ거래상대방 및 거래목적 등을 주요내용으로 하는 증빙자료를 대체할 수 있는 별지 제3호의6서식의 소명서(이하 "증빙자료대체소명서"라 한다)를 위원회에 제출하여야 한다. <개정 2020.10.13>\n\t④ 위원회는 증빙자료대체소명서의 내용에 대한 사실관계를 검증하는 과정에서 추가소명 또는 증빙자료 제출을 요구할 수 있다.\n'
            ]
        ],
        [
            '행정심판법 0054조',
            [
                '제54조(전자정보처리조직을 이용한 송달 등)\n\t① 피청구인 또는 위원회는 제52조제1항에 따라 행정심판을 청구하거나 심판참가를 한 자에게 전자정보처리조직과 그와 연계된 정보통신망을 이용하여 재결서나 이 법에 따른 각종 서류를 송달할 수 있다. 다만, 청구인이나 참가인이 동의하지 아니하는 경우에는 그러하지 아니하다.\n\t② 제1항 본문의 경우 위원회는 송달하여야 하는 재결서 등 서류를 전자정보처리조직에 입력하여 등재한 다음 그 등재 사실을 국회규칙, 대법원규칙, 헌법재판소규칙, 중앙선거관리위원회규칙 또는 대통령령으로 정하는 방법에 따라 전자우편 등으로 알려야 한다.\n\t③ 제1항에 따른 전자정보처리조직을 이용한 서류 송달은 서면으로 한 것과 같은 효력을 가진다.\n\t④ 제1항에 따른 서류의 송달은 청구인이 제2항에 따라 등재된 전자문서를 확인한 때에 전자정보처리조직에 기록된 내용으로 도달한 것으로 본다. 다만, 제2항에 따라 그 등재사실을 통지한 날부터 2주 이내(재결서 외의 서류는 7일 이내)에 확인하지 아니하였을 때에는 등재사실을 통지한 날부터 2주가 지난 날(재결서 외의 서류는 7일이 지난 날)에 도달한 것으로 본다.\n\t⑤ 서면으로 심판청구 또는 심판참가를 한 자가 전자정보처리조직의 이용을 신청한 경우에는 제52조ㆍ제53조 및 이 조를 준용한다.\n\t⑥ 위원회, 피청구인, 그 밖의 관계 행정기관 간의 서류의 송달 등에 관하여는 제52조ㆍ제53조 및 이 조를 준용한다.\n\t⑦ 제1항 본문에 따른 송달의 방법이나 그 밖에 필요한 사항은 국회규칙, 대법원규칙, 헌법재판소규칙, 중앙선거관리위원회규칙 또는 대통령령으로 정한다.\n'
            ]
        ],
        [
            '출입국관리법 0036조',
            [
                '제36조(체류지 변경의 신고)\n\t① 제31조에 따라 등록을 한 외국인이 체류지를 변경하였을 때에는 대통령령으로 정하는 바에 따라 전입한 날부터 15일 이내에 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 그 체류지를 관할하는 지방출입국ㆍ외국인관서의 장에게 전입신고를 하여야 한다. <개정 2014.3.18, 2016.3.29, 2018.3.20, 2020.6.9>\n\t② 외국인이 제1항에 따른 신고를 할 때에는 외국인등록증을 제출하여야 한다. 이 경우 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 지방출입국ㆍ외국인관서의 장은 그 외국인등록증에 체류지 변경사항을 적은 후 돌려주어야 한다. <개정 2014.3.18, 2016.3.29>\n\t③ 제1항에 따라 전입신고를 받은 지방출입국ㆍ외국인관서의 장은 지체 없이 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 체류지 변경 사실을 통보하여야 한다. <개정 2014.3.18, 2016.3.29>\n\t④ 제1항에 따라 직접 전입신고를 받거나 제3항에 따라 지방출입국ㆍ외국인관서의 장으로부터 체류지 변경통보를 받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 지체 없이 종전 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 체류지 변경신고서 사본을 첨부하여 외국인등록표의 이송을 요청하여야 한다. <개정 2014.3.18, 2016.3.29>\n\t⑤ 제4항에 따라 외국인등록표 이송을 요청받은 종전 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 이송을 요청받은 날부터 3일 이내에 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 외국인등록표를 이송하여야 한다. <개정 2016.3.29>\n\t⑥ 제5항에 따라 외국인등록표를 이송받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 신고인의 외국인등록표를 정리하고 제34조제2항에 따라 관리하여야 한다. <개정 2016.3.29>\n\t⑦ 제1항에 따라 전입신고를 받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 지방출입국ㆍ외국인관서의 장은 대통령령으로 정하는 바에 따라 그 사실을 지체 없이 종전 체류지를 관할하는 지방출입국ㆍ외국인관서의 장에게 통보하여야 한다. <개정 2014.3.18, 2016.3.29>\n'
            ]
        ],
        [
            '지방자치단체 보조금 관리에 관한 법률 0038조',
            [
                '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 5천만원 이하의 벌금에 처한다.\n\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 대하여 금지된 행위를 한 자\n\n'
            ]
        ],
        [
            '가사소송법 0038조',
            [
                '제38조(증거 조사) 가정법원은 필요하다고 인정할 경우에는 당사자 또는 법정대리인을 당사자 신문(訊問) 방식으로 심문(審問)할 수 있고, 그 밖의 관계인을 증인 신문 방식으로 심문할 수 있다.\n'
            ]
        ],
        [
            '수산업협동조합법 0176조',
            [
                '제176조(벌칙)\n\t① 조합등 또는 중앙회의 임직원이 다음 각 호의 어느 하나에 해당하는 행위로 조합등 또는 중앙회에 손실을 끼쳤을 때에는 10년 이하의 징역 또는 1억원 이하의 벌금에 처한다. <개정 2014.10.15, 2015.2.3, 2016.5.29>\t\t1. 조합등 또는 중앙회의 사업 목적 외의 용도로 자금을 사용하거나 대출하는 행위\n\t\t2. 투기의 목적으로 조합등 또는 중앙회의 재산을 처분하거나 이용하는 행위\n\n\t② 제1항의 징역형과 벌금형은 병과(倂科)할 수 있다.\n'
            ]
        ],
        [
            '출입국관리법 시행령 0059조',
            [
                '제59조(신문조서)\n\t① 법 제48조제3항에 따른 용의자신문조서에는 다음 각 호의 사항을 적어야 한다.\t\t1. 국적ㆍ성명ㆍ성별ㆍ생년월일ㆍ주소 및 직업\n\t\t2. 출입국 및 체류에 관한 사항\n\t\t3. 용의사실의 내용\n\t\t4. 그 밖에 범죄경력 등 필요한 사항\n\n\t② 출입국관리공무원은 법 제48조제6항 또는 제7항에 따라 통역이나 번역을 하게 한 때에는 통역하거나 번역한 사람으로 하여금 조서에 간인(間印)한 후 서명 또는 기명날인하게 하여야 한다.\n'
            ]
        ],
        [
            '관광진흥법 시행령 0017조',
            [
                '제17조(의견 청취) 위원장은 위원회의 심의사항과 관련하여 필요하다고 인정하면 관계인 또는 안전ㆍ소방 등에 대한 전문가를 출석시켜 그 의견을 들을 수 있다.\n'
            ]
        ],
        [
            '공직자윤리법의 시행에 관한 중앙선거관리위원회 규칙 0001조',
            [
                '제1조(목적) 이 규칙은 「공직자윤리법」에서 중앙선거관리위원회규칙에 위임된 사항과 그 밖에 그 법의 시행에 관하여 필요한 사항을 규정함을 목적으로 한다. <개정 2006.1.24, 2009.2.19>\n'
            ]
        ],
        [
            '자본시장과 금융투자업에 관한 법률 시행령 0204조',
            [
                '제204조(안정조작의 방법 등)\n\t① 제203조에 따른 투자매매업자는 법 제176조제3항제1호에 따라 그 증권의 투자설명서에 다음 각 호의 사항을 모두 기재한 경우만 안정조작을 할 수 있다. 다만, 제203조제2호의 경우에는 인수계약의 내용에 이를 기재하여야 한다.\t\t1. 안정조작을 할 수 있다는 뜻\n\t\t2. 안정조작을 할 수 있는 증권시장의 명칭\n\n\t② 제203조에 따른 투자매매업자는 투자설명서나 인수계약의 내용에 기재된 증권시장 외에서는 안정조작을 하여서는 아니 된다.\n\t③ 제203조에 따른 투자매매업자는 안정조작을 할 수 있는 기간(이하 "안정조작기간"이라 한다) 중에 최초의 안정조작을 한 경우에는 지체 없이 다음 각 호의 사항을 기재한 안정조작신고서(이하 "안정조작신고서"라 한다)를 금융위원회와 거래소에 제출하여야 한다.\t\t1. 안정조작을 한 투자매매업자의 상호\n\t\t2. 다른 투자매매업자와 공동으로 안정조작을 한 경우에는 그 다른 투자매매업자의 상호\n\t\t3. 안정조작을 한 증권의 종목 및 매매가격\n\t\t4. 안정조작을 개시한 날과 시간\n\t\t5. 안정조작기간\n\t\t6. 안정조작에 의하여 그 모집 또는 매출을 원활하게 하려는 증권의 모집 또는 매출가격과 모집 또는 매출가액의 총액\n\t\t7. 안정조작을 한 증권시장의 명칭\n\n\t④ 제203조에 따른 투자매매업자는 다음 각 호에서 정하는 가격을 초과하여 안정조작의 대상이 되는 증권(이하 "안정조작증권"이라 한다)을 매수하여서는 아니 된다.\t\t1. 안정조작개시일의 경우\t\t\t가. 최초로 안정조작을 하는 경우: 안정조작개시일 전에 증권시장에서 거래된 해당 증권의 직전 거래가격과 안정조작기간의 초일 전 20일간의 증권시장에서의 평균거래가격 중 낮은 가격. 이 경우 평균거래가격의 계산방법은 금융위원회가 정하여 고시한다.\n\t\t\t나. 최초 안정조작 이후에 안정조작을 하는 경우: 그 투자매매업자의 안정조작 개시가격\n\n\t\t2. 안정조작개시일의 다음 날 이후의 경우: 안정조작 개시가격(같은 날에 안정조작을 한 투자매매업자가 둘 이상 있는 경우에는 이들 투자매매업자의 안정조작 개시가격 중 가장 낮은 가격)과 안정조작을 하는 날 이전에 증권시장에서 거래된 해당 증권의 직전거래가격 중 낮은 가격\n\n\t⑤ 제203조에 따른 투자매매업자는 안정조작을 한 증권시장마다 안정조작개시일부터 안정조작종료일까지의 기간 동안 안정조작증권의 매매거래에 대하여 해당 매매거래를 한 날의 다음 날까지 다음 각 호의 사항을 기재한 안정조작보고서(이하 "안정조작보고서"라 한다)를 작성하여 금융위원회와 거래소에 제출하여야 한다.\t\t1. 안정조작을 한 증권의 종목\n\t\t2. 매매거래의 내용\n\t\t3. 안정조작을 한 투자매매업자의 상호\n\n\t⑥ 금융위원회와 거래소는 안정조작신고서와 안정조작보고서를 다음 각 호에서 정하는 날부터 3년간 비치하고, 인터넷 홈페이지 등을 이용하여 공시하여야 한다.\t\t1. 안정조작신고서의 경우: 이를 접수한 날\n\t\t2. 안정조작보고서의 경우: 안정조작 종료일의 다음 날\n\n\t⑦ 법 제176조제3항제1호에서 "대통령령으로 정하는 날"이란 모집되거나 매출되는 증권의 모집 또는 매출의 청약기간의 종료일 전 20일이 되는 날을 말한다. 다만, 20일이 되는 날과 청약일 사이의 기간에 모집가액 또는 매출가액이 확정되는 경우에는 그 확정되는 날의 다음 날을 말한다.\n\t⑧ 제1항부터 제7항까지에서 규정한 사항 외에 안정조작신고서ㆍ안정조작보고서의 서식과 작성방법 등에 관하여 필요한 사항은 금융위원회가 정하여 고시한다.\n'
            ]
        ],
        [
            '민사집행법 0182조',
            [
                '제182조(사건의 이송)\n\t① 압류된 선박이 관할구역 밖으로 떠난 때에는 집행법원은 선박이 있는 곳을 관할하는 법원으로 사건을 이송할 수 있다.\n\t② 제1항의 규정에 따른 결정에 대하여는 불복할 수 없다.\n'
            ]
        ],
        [
            '구직자 취업촉진 및 생활안정지원에 관한 법률 0024조',
            [
                '제24조(소멸시효)\n\t① 구직촉진수당등을 지급받거나 제28조에 따라 반환받을 권리는 3년간 행사하지 아니하면 시효로 소멸한다.\n\t② 제1항에 따른 소멸시효는 수급자 또는 고용노동부장관의 청구로 중단된다.\n'
            ]
        ],
        [
            '검찰압수물사무규칙 0067조',
            [
                '제67조(사건종결전 처분의 정리) 사건종결전에 이 절에 정한 환부등의 처분을 한 때에는 압수물사무담당직원은 압수표에 그 뜻을 기재하고 소속과장의 확인을 받아야 한다. 이 경우 압수물 총목록 및 압수조서등에도 그 뜻을 기재하여야 한다.\n'
            ]
        ],
        [
            '지방공무원법 0075조',
            [
                '제75조의2(적극행정의 장려)\n\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 같다)을 장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. 이 경우 대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 한다.\n\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 장 소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 관한 사항\n\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 적극적으로 추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 제시를 요청한 사항\n\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n\n\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 중대한 과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 하지 아니한다.\n\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 필요한 사업을 발굴하고 추진할 수 있다.\n\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n'
            ]
        ],
        [
            '체육시설의 설치ㆍ이용에 관한 법률 시행규칙 0007조',
            [
                '제7조(대중골프장업의 세분) 영 제7조제2항에 따라 대중골프장업의 종류를 다음 각 호와 같이 세분 한다.\n\t\t\t1. 정규 대중골프장업\n\t\t2. 일반 대중골프장업\n\t\t3. 간이골프장업\n\n'
            ]
        ],
        [
            '부동산등기규칙 0152조',
            [
                '제152조(가처분등기 이후의 등기의 말소)\n\t① 소유권이전등기청구권 또는 소유권이전등기말소등기(소유권보존등기말소등기를 포함한다. 이하 이 조에서 같다)청구권을 보전하기 위한 가처분등기가 마쳐진 후 그 가처분채권자가 가처분채무자를 등기의무자로 하여 소유권이전등기 또는 소유권말소등기를 신청하는 경우에는, 법 제94조제1항에 따라 가처분등기 이후에 마쳐진 제3자 명의의 등기의 말소를 단독으로 신청할 수 있다. 다만, 다음 각 호의 등기는 그러하지 아니하다.\t\t1. 가처분등기 전에 마쳐진 가압류에 의한 강제경매개시결정등기\n\t\t2. 가처분등기 전에 마쳐진 담보가등기, 전세권 및 저당권에 의한 임의경매개시결정등기\n\t\t3. 가처분채권자에게 대항할 수 있는 주택임차권등기등\n\n\t② 가처분채권자가 제1항에 따른 소유권이전등기말소등기를 신청하기 위하여는 제1항 단서 각 호의 권리자의 승낙이나 이에 대항할 수 있는 재판이 있음을 증명하는 정보를 첨부정보로서 등기소에 제공하여야 한다.\n'
            ]
        ],
        [
            '물품관리법 시행령 0027조',
            [
                '제27조(회계 간의 관리전환)\n\t① 법 제22조제2항에서 "대통령령으로 정하는 관리전환의 경우"란 다음 각 호의 어느 하나에 해당하는 경우를 말한다.\t\t1. 6개월 이내에 반환하는 조건으로 물품을 관리전환하는 경우\n\t\t2. 제26조제2호에 따른 물품을 관리전환하는 경우\n\t\t3. 각 중앙관서의 장이 조달청장과 협의하여 무상으로 관리전환하기로 정한 경우\n\n\t② 법 제22조제2항에 따라 관리전환을 유상으로 정리할 때의 가액은 해당 물품의 대장가격(臺帳價格)으로 한다. 다만, 대장가격으로 정리하기 곤란할 때에는 시가(時價)로 정리할 수 있다.\n'
            ]
        ],
        [
            '노동조합 및 노동관계조정법 시행령 0020조',
            [
                '제20조(방산물자 생산업무 종사자의 범위) 법 제41조제2항에서 "주로 방산물자를 생산하는 업무에 종사하는 자"라 함은 방산물자의 완성에 필요한 제조ㆍ가공ㆍ조립ㆍ정비ㆍ재생ㆍ개량ㆍ성능검사ㆍ열처리ㆍ도장ㆍ가스취급 등의 업무에 종사하는 자를 말한다.\n'
            ]
        ],
        [
            '혁신의료기기 지원 및 관리 등에 관한 규칙 0011조',
            [
            '제11조(혁신의료기기소프트웨어의 변경허가 또는 변경인증) 법 제24조제4항에 따라 혁신의료기기소프트웨어의 변경허가 또는 변경인증을 받으려는 자는 그 변경이 있은 날부터 30일 이내에 「의료기기법 시행규칙」 제26조제3항에 따른 변경허가(변경인증) 신청서(전자문서로 된 신청서를 포함한다)에 다음 각 호의 자료(전자문서를 포함한다)를 첨부하여 식품의약품안전처장 또는 「의료기기법」 제42조에 따른 한국의료기기안전정보원(이하 "정보원"이라 한다)에 제출해야 한다.\n\t\t\t1. 변경사실을 확인할 수 있는 서류\n\t\t2. 「의료기기법」 제6조제5항에 따른 기술문서와 임상시험자료(혁신의료기기소프트웨어의 안전성ㆍ유효성에 영향을 미치는 경우로서 식품의약품안전처장이 정하여 고시하는 변경사항만 해당한다)\n\t\t3. 제15조에 따른 시설과 제조 및 품질관리체계의 기준에 적합함을 증명하는 자료(제조소 또는 영업소 등이 변경되는 경우로서 식품의약품안전처장이 정하여 고시하는 변경사항만 해당한다)\n\n'
            ]
        ]
    ],
}
```
</div>
</details>

<details>
<summary> * 출력 예시 (kr_law) (click)</summary>
<div markdown="1">

```python
{'num_retrieved_doc': 6,
 'retrieved_doc': [{'text': '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 5천만원 '
                            '이하의 벌금에 처한다.\n'
                            '\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n'
                            '\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 대하여 '
                            '금지된 행위를 한 자\n'
                            '\n',
                    'title': '지방자치단체 보조금 관리에 관한 법률 0038조'},
                   {'text': '제75조의2(적극행정의 장려)\n'
                            '\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 등 '
                            '공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 같다)을 '
                            '장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. 이 경우 '
                            '대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 한다.\n'
                            '\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 장 '
                            '소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 '
                            '인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 '
                            '적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 관한 '
                            '사항\n'
                            '\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 적극적으로 '
                            '추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 제시를 요청한 '
                            '사항\n'
                            '\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n'
                            '\n'
                            '\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 중대한 '
                            '과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 하지 '
                            '아니한다.\n'
                            '\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 필요한 '
                            '사업을 발굴하고 추진할 수 있다.\n'
                            '\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 '
                            '적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n',
                    'title': '지방공무원법 0075조'},
                   {'text': '제59조(신문조서)\n'
                            '\t① 법 제48조제3항에 따른 용의자신문조서에는 다음 각 호의 사항을 적어야 '
                            '한다.\t\t1. 국적ㆍ성명ㆍ성별ㆍ생년월일ㆍ주소 및 직업\n'
                            '\t\t2. 출입국 및 체류에 관한 사항\n'
                            '\t\t3. 용의사실의 내용\n'
                            '\t\t4. 그 밖에 범죄경력 등 필요한 사항\n'
                            '\n'
                            '\t② 출입국관리공무원은 법 제48조제6항 또는 제7항에 따라 통역이나 번역을 하게 한 '
                            '때에는 통역하거나 번역한 사람으로 하여금 조서에 간인(間印)한 후 서명 또는 기명날인하게 '
                            '하여야 한다.\n',
                    'title': '출입국관리법 시행령 0059조'},
                   {'text': '제24조(소멸시효)\n'
                            '\t① 구직촉진수당등을 지급받거나 제28조에 따라 반환받을 권리는 3년간 행사하지 '
                            '아니하면 시효로 소멸한다.\n'
                            '\t② 제1항에 따른 소멸시효는 수급자 또는 고용노동부장관의 청구로 중단된다.\n',
                    'title': '구직자 취업촉진 및 생활안정지원에 관한 법률 0024조'},
                   {'text': '제113조(벌칙) 제45조를 위반한 자는 1천만원 이하의 벌금에 처한다.',
                    'title': '근로기준법 0113조'},
                   {'text': '제9조의4(재산형성과정 소명 요구 등)\n'
                            '\t① 위원회는 등록의무자가 다음 각 호의 어느 하나에 해당하는 경우에는 법 '
                            '제8조제13항에 따라 재산형성과정의 소명을 요구할 수 있다. <개정 '
                            '2020.10.13>\t\t1. 직무와 관련하여 부정한 재산증식을 의심할 만한 상당한 '
                            '사유가 있는 경우\n'
                            '\t\t2. 법 제8조의2제6항에 따른 다른 법령을 위반하여 부정하게 재물 또는 재산상 '
                            '이익을 얻었다는 혐의를 입증하기 위한 경우\n'
                            '\t\t3. 재산상의 문제로 사회적 물의를 일으킨 경우\n'
                            '\t\t4. 등록의무자의 보수 수준 등을 고려할 때 특별한 사유 없이 재산의 뚜렷한 '
                            '증감이 있는 경우\n'
                            '\t\t5. 제1호부터 제4호까지의 규정에 상당하는 사유로 위원회가 소명 요구를 의결한 '
                            '경우\n'
                            '\n'
                            '\t② 재산형성과정의 소명을 요구받은 사람은 특별한 사유가 없으면 요구받은 날부터 20일 '
                            '이내에 별지 제3호의5서식의 소명서 및 증빙자료를 위원회에 제출하여야 한다. <개정 '
                            '2020.10.13>\n'
                            '\t③ 재산형성과정의 소명을 요구받은 사람은 분실ㆍ멸실 및 훼손 등의 사유로 증빙자료를 '
                            '제출할 수 없는 경우에는 위원회에 그 사실을 소명하고, 거래시기ㆍ거래상대방 및 거래목적 '
                            '등을 주요내용으로 하는 증빙자료를 대체할 수 있는 별지 제3호의6서식의 소명서(이하 '
                            '"증빙자료대체소명서"라 한다)를 위원회에 제출하여야 한다. <개정 2020.10.13>\n'
                            '\t④ 위원회는 증빙자료대체소명서의 내용에 대한 사실관계를 검증하는 과정에서 추가소명 '
                            '또는 증빙자료 제출을 요구할 수 있다.\n',
                    'title': '공직자윤리법의 시행에 관한 헌법재판소 규칙 0009조'}],
 'top_n_candidates': [[{'text': '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 '
                                '5천만원 이하의 벌금에 처한다.\n'
                                '\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n'
                                '\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 '
                                '대하여 금지된 행위를 한 자\n'
                                '\n',
                        'title': '지방자치단체 보조금 관리에 관한 법률 0038조'},
                       {'text': '제75조의2(적극행정의 장려)\n'
                                '\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 '
                                '등 공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 '
                                '같다)을 장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. '
                                '이 경우 대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 '
                                '한다.\n'
                                '\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 '
                                '장 소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 '
                                '인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 '
                                '적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 '
                                '관한 사항\n'
                                '\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 '
                                '적극적으로 추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 '
                                '제시를 요청한 사항\n'
                                '\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n'
                                '\n'
                                '\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 '
                                '중대한 과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 '
                                '하지 아니한다.\n'
                                '\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 '
                                '필요한 사업을 발굴하고 추진할 수 있다.\n'
                                '\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 '
                                '적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n',
                        'title': '지방공무원법 0075조'}],
                      [{'text': '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 '
                                '5천만원 이하의 벌금에 처한다.\n'
                                '\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n'
                                '\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 '
                                '대하여 금지된 행위를 한 자\n'
                                '\n',
                        'title': '지방자치단체 보조금 관리에 관한 법률 0038조'},
                       {'text': '제59조(신문조서)\n'
                                '\t① 법 제48조제3항에 따른 용의자신문조서에는 다음 각 호의 사항을 적어야 '
                                '한다.\t\t1. 국적ㆍ성명ㆍ성별ㆍ생년월일ㆍ주소 및 직업\n'
                                '\t\t2. 출입국 및 체류에 관한 사항\n'
                                '\t\t3. 용의사실의 내용\n'
                                '\t\t4. 그 밖에 범죄경력 등 필요한 사항\n'
                                '\n'
                                '\t② 출입국관리공무원은 법 제48조제6항 또는 제7항에 따라 통역이나 번역을 '
                                '하게 한 때에는 통역하거나 번역한 사람으로 하여금 조서에 간인(間印)한 후 서명 '
                                '또는 기명날인하게 하여야 한다.\n',
                        'title': '출입국관리법 시행령 0059조'}],
                      [{'text': '제24조(소멸시효)\n'
                                '\t① 구직촉진수당등을 지급받거나 제28조에 따라 반환받을 권리는 3년간 행사하지 '
                                '아니하면 시효로 소멸한다.\n'
                                '\t② 제1항에 따른 소멸시효는 수급자 또는 고용노동부장관의 청구로 중단된다.\n',
                        'title': '구직자 취업촉진 및 생활안정지원에 관한 법률 0024조'},
                       {'text': '제113조(벌칙) 제45조를 위반한 자는 1천만원 이하의 벌금에 처한다.',
                        'title': '근로기준법 0113조'}],
                      [{'text': '제24조(소멸시효)\n'
                                '\t① 구직촉진수당등을 지급받거나 제28조에 따라 반환받을 권리는 3년간 행사하지 '
                                '아니하면 시효로 소멸한다.\n'
                                '\t② 제1항에 따른 소멸시효는 수급자 또는 고용노동부장관의 청구로 중단된다.\n',
                        'title': '구직자 취업촉진 및 생활안정지원에 관한 법률 0024조'},
                       {'text': '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 '
                                '5천만원 이하의 벌금에 처한다.\n'
                                '\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n'
                                '\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 '
                                '대하여 금지된 행위를 한 자\n'
                                '\n',
                        'title': '지방자치단체 보조금 관리에 관한 법률 0038조'}],
                      [{'text': '제75조의2(적극행정의 장려)\n'
                                '\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 '
                                '등 공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 '
                                '같다)을 장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. '
                                '이 경우 대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 '
                                '한다.\n'
                                '\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 '
                                '장 소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 '
                                '인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 '
                                '적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 '
                                '관한 사항\n'
                                '\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 '
                                '적극적으로 추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 '
                                '제시를 요청한 사항\n'
                                '\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n'
                                '\n'
                                '\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 '
                                '중대한 과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 '
                                '하지 아니한다.\n'
                                '\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 '
                                '필요한 사업을 발굴하고 추진할 수 있다.\n'
                                '\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 '
                                '적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n',
                        'title': '지방공무원법 0075조'},
                       {'text': '제9조의4(재산형성과정 소명 요구 등)\n'
                                '\t① 위원회는 등록의무자가 다음 각 호의 어느 하나에 해당하는 경우에는 법 '
                                '제8조제13항에 따라 재산형성과정의 소명을 요구할 수 있다. <개정 '
                                '2020.10.13>\t\t1. 직무와 관련하여 부정한 재산증식을 의심할 만한 '
                                '상당한 사유가 있는 경우\n'
                                '\t\t2. 법 제8조의2제6항에 따른 다른 법령을 위반하여 부정하게 재물 또는 '
                                '재산상 이익을 얻었다는 혐의를 입증하기 위한 경우\n'
                                '\t\t3. 재산상의 문제로 사회적 물의를 일으킨 경우\n'
                                '\t\t4. 등록의무자의 보수 수준 등을 고려할 때 특별한 사유 없이 재산의 '
                                '뚜렷한 증감이 있는 경우\n'
                                '\t\t5. 제1호부터 제4호까지의 규정에 상당하는 사유로 위원회가 소명 요구를 '
                                '의결한 경우\n'
                                '\n'
                                '\t② 재산형성과정의 소명을 요구받은 사람은 특별한 사유가 없으면 요구받은 날부터 '
                                '20일 이내에 별지 제3호의5서식의 소명서 및 증빙자료를 위원회에 제출하여야 '
                                '한다. <개정 2020.10.13>\n'
                                '\t③ 재산형성과정의 소명을 요구받은 사람은 분실ㆍ멸실 및 훼손 등의 사유로 '
                                '증빙자료를 제출할 수 없는 경우에는 위원회에 그 사실을 소명하고, '
                                '거래시기ㆍ거래상대방 및 거래목적 등을 주요내용으로 하는 증빙자료를 대체할 수 있는 '
                                '별지 제3호의6서식의 소명서(이하 "증빙자료대체소명서"라 한다)를 위원회에 '
                                '제출하여야 한다. <개정 2020.10.13>\n'
                                '\t④ 위원회는 증빙자료대체소명서의 내용에 대한 사실관계를 검증하는 과정에서 '
                                '추가소명 또는 증빙자료 제출을 요구할 수 있다.\n',
                        'title': '공직자윤리법의 시행에 관한 헌법재판소 규칙 0009조'}],
                      [{'text': '제1조(목적) 이 규칙은 「공직자윤리법」에서 중앙선거관리위원회규칙에 위임된 사항과 '
                                '그 밖에 그 법의 시행에 관하여 필요한 사항을 규정함을 목적으로 한다. <개정 '
                                '2006.1.24, 2009.2.19>\n',
                        'title': '공직자윤리법의 시행에 관한 중앙선거관리위원회 규칙 0001조'},
                       {'text': '제75조의2(적극행정의 장려)\n'
                                '\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 '
                                '등 공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 '
                                '같다)을 장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. '
                                '이 경우 대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 '
                                '한다.\n'
                                '\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 '
                                '장 소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 '
                                '인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 '
                                '적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 '
                                '관한 사항\n'
                                '\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 '
                                '적극적으로 추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 '
                                '제시를 요청한 사항\n'
                                '\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n'
                                '\n'
                                '\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 '
                                '중대한 과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 '
                                '하지 아니한다.\n'
                                '\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 '
                                '필요한 사업을 발굴하고 추진할 수 있다.\n'
                                '\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 '
                                '적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n',
                        'title': '지방공무원법 0075조'}]]}
```
</div>
</details>





## 04-RelationExtraction-KETI

* 웹 API 정보: `http://ketiair.com:10022/`

* 입력 파라미터

| Key       | Value                                      | Explanation                                                  |
| --------- | ------------------------------------------ | ------------------------------------------------------------ |
| doc       | str                                        | (required) 관계 추출을 수행할 문서                           |
| arg_pairs | list[list[list[int,int],list[int,int]]] | (optional) 관계를 알고 싶은 객체 쌍들의 문서 내 시작･종료 위치, 주로 주어･목적어 쌍 |

* 입력 예시

```json
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
```

</div>
</details>

<details>
<summary> * 출력 예시 (click)</summary>
<div markdown="1">

```json
{
  "result":
    [
      {
        "subject": "문성민",
        "relation": "per:schools_attended",
        "object": "경기대학"
      },
      {
        "subject": "문성민",
        "relation": "per:spouse",
        "object": "이선희"
    }
  ]
}
```
</div>
</details>

* 출력 예시

```json
{"result": ["per:spouse"]}
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



