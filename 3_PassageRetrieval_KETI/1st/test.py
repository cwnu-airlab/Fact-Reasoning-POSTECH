import json
import requests
from urllib.parse import urljoin

import pprint

URL = 'http://ketiair.com:10021'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'

# '_id': '5a7ba03c554299294a54aa34',
# 'answer': '1989',
# 'supporting_facts': [['2013년 오클랜드 레이더스 시즌입니다.', '2'], ['테렐 프라이어', '0']],
# 'type': 'bridge'
# 'level': 'medium',

ko_content = {
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


# 'type': 'comparison',
# 'level': 'hard'
# '_id': '5a8b57f25542995d1e6f1371',
# 'answer': 'yes',
# 'supporting_facts': [['Scott Derrickson', 0], ['Ed Wood', 0]],

en_content = {
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

# english common-sense
data = json.dumps(
    en_content
)
headers = {'Content-Type': 'application/json; charset=utf-8'}  # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
print(response.status_code)
print(response.request)
pprint.pprint(response.json())

# {'num_retrieved_doc': 6,
#  'retrieved_doc': [{'text': 'Scott Derrickson (born July 16, 1966) is an '
#                             'American director, screenwriter and producer.  He '
#                             'lives in Los Angeles, California.  He is best '
#                             'known for directing horror films such as '
#                             '"Sinister", "The Exorcism of Emily Rose", and '
#                             '"Deliver Us From Evil", as well as the 2016 '
#                             'Marvel Cinematic Universe installment, "Doctor '
#                             'Strange."',
#                     'title': 'Scott Derrickson'},
#                    {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
#                             'December 10, 1978) was an American filmmaker, '
#                             'actor, writer, producer, and director.',
#                     'title': 'Ed Wood'},
#                    {'text': 'Adam Collis is an American filmmaker and actor.  '
#                             'He attended the Duke University from 1986 to 1990 '
#                             'and the University of California, Los Angeles '
#                             'from 2007 to 2010.  He also studied cinema at the '
#                             'University of Southern California from 1991 to '
#                             '1997.  Collis first work was the assistant '
#                             'director for the Scott Derrickson\'s short "Love '
#                             'in the Ruins" (1995).  In 1998, he played '
#                             '"Crankshaft" in Eric Koyanagi\'s "Hundred '
#                             'Percent".',
#                     'title': 'Adam Collis'},
#                    {'text': 'Conrad Brooks (born Conrad Biedrzycki on January '
#                             '3, 1931 in Baltimore, Maryland) is an American '
#                             'actor.  He moved to Hollywood, California in 1948 '
#                             'to pursue a career in acting.  He got his start '
#                             'in movies appearing in Ed Wood films such as '
#                             '"Plan 9 from Outer Space", "Glen or Glenda", and '
#                             '"Jail Bait."  He took a break from acting during '
#                             'the 1960s and 1970s but due to the ongoing '
#                             'interest in the films of Ed Wood, he reemerged in '
#                             'the 1980s and has become a prolific actor.  He '
#                             'also has since gone on to write, produce and '
#                             'direct several films.',
#                     'title': 'Conrad Brooks'},
#                    {'text': 'Sinister is a 2012 supernatural horror film '
#                             'directed by Scott Derrickson and written by '
#                             'Derrickson and C. Robert Cargill.  It stars Ethan '
#                             'Hawke as fictional true-crime writer Ellison '
#                             'Oswalt who discovers a box of home movies in his '
#                             'attic that puts his family in danger.',
#                     'title': 'Sinister (film)'},
#                    {'text': 'Deliver Us from Evil is a 2014 American '
#                             'supernatural horror film directed by Scott '
#                             'Derrickson and produced by Jerry Bruckheimer.  '
#                             'The film is officially based on a 2001 '
#                             'non-fiction book entitled "Beware the Night" by '
#                             'Ralph Sarchie and Lisa Collier Cool, and its '
#                             'marketing campaign highlighted that it was '
#                             '"inspired by actual accounts".  The film stars '
#                             'Eric Bana, Édgar Ramírez, Sean Harris, Olivia '
#                             'Munn, and Joel McHale in the main roles and was '
#                             'released on July 2, 2014.',
#                     'title': 'Deliver Us from Evil (2014 film)'}],
#  'top_n_candidates': [[{'text': 'Scott Derrickson (born July 16, 1966) is an '
#                                 'American director, screenwriter and '
#                                 'producer.  He lives in Los Angeles, '
#                                 'California.  He is best known for directing '
#                                 'horror films such as "Sinister", "The '
#                                 'Exorcism of Emily Rose", and "Deliver Us From '
#                                 'Evil", as well as the 2016 Marvel Cinematic '
#                                 'Universe installment, "Doctor Strange."',
#                         'title': 'Scott Derrickson'},
#                        {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
#                                 'December 10, 1978) was an American filmmaker, '
#                                 'actor, writer, producer, and director.',
#                         'title': 'Ed Wood'}],
#                       [{'text': 'Adam Collis is an American filmmaker and '
#                                 'actor.  He attended the Duke University from '
#                                 '1986 to 1990 and the University of '
#                                 'California, Los Angeles from 2007 to 2010.  '
#                                 'He also studied cinema at the University of '
#                                 'Southern California from 1991 to 1997.  '
#                                 'Collis first work was the assistant director '
#                                 'for the Scott Derrickson\'s short "Love in '
#                                 'the Ruins" (1995).  In 1998, he played '
#                                 '"Crankshaft" in Eric Koyanagi\'s "Hundred '
#                                 'Percent".',
#                         'title': 'Adam Collis'},
#                        {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
#                                 'December 10, 1978) was an American filmmaker, '
#                                 'actor, writer, producer, and director.',
#                         'title': 'Ed Wood'}],
#                       [{'text': 'Scott Derrickson (born July 16, 1966) is an '
#                                 'American director, screenwriter and '
#                                 'producer.  He lives in Los Angeles, '
#                                 'California.  He is best known for directing '
#                                 'horror films such as "Sinister", "The '
#                                 'Exorcism of Emily Rose", and "Deliver Us From '
#                                 'Evil", as well as the 2016 Marvel Cinematic '
#                                 'Universe installment, "Doctor Strange."',
#                         'title': 'Scott Derrickson'},
#                        {'text': 'Conrad Brooks (born Conrad Biedrzycki on '
#                                 'January 3, 1931 in Baltimore, Maryland) is an '
#                                 'American actor.  He moved to Hollywood, '
#                                 'California in 1948 to pursue a career in '
#                                 'acting.  He got his start in movies appearing '
#                                 'in Ed Wood films such as "Plan 9 from Outer '
#                                 'Space", "Glen or Glenda", and "Jail Bait."  '
#                                 'He took a break from acting during the 1960s '
#                                 'and 1970s but due to the ongoing interest in '
#                                 'the films of Ed Wood, he reemerged in the '
#                                 '1980s and has become a prolific actor.  He '
#                                 'also has since gone on to write, produce and '
#                                 'direct several films.',
#                         'title': 'Conrad Brooks'}],
#                       [{'text': 'Adam Collis is an American filmmaker and '
#                                 'actor.  He attended the Duke University from '
#                                 '1986 to 1990 and the University of '
#                                 'California, Los Angeles from 2007 to 2010.  '
#                                 'He also studied cinema at the University of '
#                                 'Southern California from 1991 to 1997.  '
#                                 'Collis first work was the assistant director '
#                                 'for the Scott Derrickson\'s short "Love in '
#                                 'the Ruins" (1995).  In 1998, he played '
#                                 '"Crankshaft" in Eric Koyanagi\'s "Hundred '
#                                 'Percent".',
#                         'title': 'Adam Collis'},
#                        {'text': 'Conrad Brooks (born Conrad Biedrzycki on '
#                                 'January 3, 1931 in Baltimore, Maryland) is an '
#                                 'American actor.  He moved to Hollywood, '
#                                 'California in 1948 to pursue a career in '
#                                 'acting.  He got his start in movies appearing '
#                                 'in Ed Wood films such as "Plan 9 from Outer '
#                                 'Space", "Glen or Glenda", and "Jail Bait."  '
#                                 'He took a break from acting during the 1960s '
#                                 'and 1970s but due to the ongoing interest in '
#                                 'the films of Ed Wood, he reemerged in the '
#                                 '1980s and has become a prolific actor.  He '
#                                 'also has since gone on to write, produce and '
#                                 'direct several films.',
#                         'title': 'Conrad Brooks'}],
#                       [{'text': 'Sinister is a 2012 supernatural horror film '
#                                 'directed by Scott Derrickson and written by '
#                                 'Derrickson and C. Robert Cargill.  It stars '
#                                 'Ethan Hawke as fictional true-crime writer '
#                                 'Ellison Oswalt who discovers a box of home '
#                                 'movies in his attic that puts his family in '
#                                 'danger.',
#                         'title': 'Sinister (film)'},
#                        {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
#                                 'December 10, 1978) was an American filmmaker, '
#                                 'actor, writer, producer, and director.',
#                         'title': 'Ed Wood'}],
#                       [{'text': 'Deliver Us from Evil is a 2014 American '
#                                 'supernatural horror film directed by Scott '
#                                 'Derrickson and produced by Jerry '
#                                 'Bruckheimer.  The film is officially based on '
#                                 'a 2001 non-fiction book entitled "Beware the '
#                                 'Night" by Ralph Sarchie and Lisa Collier '
#                                 'Cool, and its marketing campaign highlighted '
#                                 'that it was "inspired by actual accounts".  '
#                                 'The film stars Eric Bana, Édgar Ramírez, Sean '
#                                 'Harris, Olivia Munn, and Joel McHale in the '
#                                 'main roles and was released on July 2, 2014.',
#                         'title': 'Deliver Us from Evil (2014 film)'},
#                        {'text': 'Edward Davis Wood Jr. (October 10, 1924 – '
#                                 'December 10, 1978) was an American filmmaker, '
#                                 'actor, writer, producer, and director.',
#                         'title': 'Ed Wood'}]]}


print(response.raise_for_status())


# korean common-sense
data = json.dumps(
    ko_content
)
response = requests.post(urljoin(URL, task_q), data=data, headers=headers)
pprint.pprint(response.json())

# {'num_retrieved_doc': 4,
#  'retrieved_doc': [{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 리그 '
#                             '오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 스파탄스에서 대학 '
#                             '미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 주에서 통산 최다 우승 '
#                             '기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 4라운드에서 오클랜드 레이더스에 '
#                             '선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 백업으로 활약한 뒤 2016년 '
#                             '미국프로축구연맹(NFL) 마지막 정규시즌 레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 '
#                             'NFL 경기에 출전했다. 이어 휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 '
#                             '지명돼 NFL 역사상 첫 쿼터백으로 플레이오프 첫 선발 등판했다.',
#                     'title': '코너 쿡'},
#                    {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, 그리고 '
#                             '데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, 레이더스는 11년 '
#                             '연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 진출하지 못했습니다. '
#                             '레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 맞이했습니다. 프라이어는 인상적인 '
#                             '패션으로 시즌을 시작했고, 인디애나폴리스 콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, '
#                             '2주차에는 잭슨빌 재규어스를 물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 '
#                             '텍산스와의 경기에서 프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 '
#                             '전에, 이글스 쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 '
#                             '역사상 가장 많이 허용했던 터치다운 패스였어요.',
#                     'title': '2013년 오클랜드 레이더스 시즌입니다.'},
#                    {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 ~ )는 '
#                             '미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 리시버이다.',
#                     'title': '테렐 프라이어'},
#                    {'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 프랜차이즈는 '
#                             '1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 NFL로 이적한 미네소타 '
#                             '바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 8번째 회원이 됐다. '
#                             'Raiders는 AFL-NFL 합병 이후 1970년에 NFL에 합류했다. 그들은 '
#                             '오클랜드로 돌아가기 전 1982년과 1995년 사이에 로스앤젤레스에서 뛰었다. 라이더스는 '
#                             '2015시즌 말 현재 AFL과 NFL에서 총 56시즌 동안 852경기를 뛰었다. 이 '
#                             '경기에서 두 명의 코치가 팀과 함께 슈퍼볼 우승을 차지했다. 1976년 존 매든, '
#                             '1980년과 1983년 톰 플로레스. 1966년 존 라우치라는 한 코치가 AFL '
#                             '챔피언십에서 우승했다. 다른 세 감독인 아트 셸, 존 그루든, 빌 캘러한도 레이더스를 '
#                             '플레이오프에 진출시켰다. 캘러한은 레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 '
#                             '이 일을 했다.',
#                     'title': '오클랜드 레이더스 감독 목록'}],
#  'top_n_candidates': [[{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 '
#                                 '리그 오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 '
#                                 '스파탄스에서 대학 미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 '
#                                 '주에서 통산 최다 우승 기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 '
#                                 '4라운드에서 오클랜드 레이더스에 선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 '
#                                 '백업으로 활약한 뒤 2016년 미국프로축구연맹(NFL) 마지막 정규시즌 '
#                                 '레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 NFL 경기에 출전했다. 이어 '
#                                 '휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 지명돼 NFL 역사상 첫 '
#                                 '쿼터백으로 플레이오프 첫 선발 등판했다.',
#                         'title': '코너 쿡'},
#                        {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
#                                 '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
#                                 '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
#                                 '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
#                                 '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
#                                 '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
#                                 '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
#                                 '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
#                                 '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
#                                 '가장 많이 허용했던 터치다운 패스였어요.',
#                         'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
#                       [{'text': '코너 쿡(Connor Cook, 1993년 1월 29일 ~ )은 미국 내셔널 풋볼 '
#                                 '리그 오클랜드 레이더스의 쿼터백이다. 2013년부터 2015년까지 미시간주 '
#                                 '스파탄스에서 대학 미식축구 선수로 활약하며 주전 쿼터백으로 활약했다. 그는 미시간 '
#                                 '주에서 통산 최다 우승 기록을 보유하고 있다. 쿡은 2016년 NFL 드래프트 '
#                                 '4라운드에서 오클랜드 레이더스에 선발됐다. 쿡은 당초 데릭 카와 맷 맥글린의 3군 '
#                                 '백업으로 활약한 뒤 2016년 미국프로축구연맹(NFL) 마지막 정규시즌 '
#                                 '레이더스전에서 카와 맥글린이 부상을 당한 뒤 첫 NFL 경기에 출전했다. 이어 '
#                                 '휴스턴 텍사스를 상대로 한 레이더스 플레이오프 선발투수로 지명돼 NFL 역사상 첫 '
#                                 '쿼터백으로 플레이오프 첫 선발 등판했다.',
#                         'title': '코너 쿡'},
#                        {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
#                                 '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
#                                 '리시버이다.',
#                         'title': '테렐 프라이어'}],
#                       [{'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
#                                 '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
#                                 '리시버이다.',
#                         'title': '테렐 프라이어'},
#                        {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
#                                 '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
#                                 '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
#                                 '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
#                                 '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
#                                 '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
#                                 '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
#                                 '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
#                                 '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
#                                 '가장 많이 허용했던 터치다운 패스였어요.',
#                         'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
#                       [{'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 '
#                                 '프랜차이즈는 1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 '
#                                 'NFL로 이적한 미네소타 바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 '
#                                 '8번째 회원이 됐다. Raiders는 AFL-NFL 합병 이후 1970년에 '
#                                 'NFL에 합류했다. 그들은 오클랜드로 돌아가기 전 1982년과 1995년 사이에 '
#                                 '로스앤젤레스에서 뛰었다. 라이더스는 2015시즌 말 현재 AFL과 NFL에서 총 '
#                                 '56시즌 동안 852경기를 뛰었다. 이 경기에서 두 명의 코치가 팀과 함께 슈퍼볼 '
#                                 '우승을 차지했다. 1976년 존 매든, 1980년과 1983년 톰 플로레스. '
#                                 '1966년 존 라우치라는 한 코치가 AFL 챔피언십에서 우승했다. 다른 세 감독인 '
#                                 '아트 셸, 존 그루든, 빌 캘러한도 레이더스를 플레이오프에 진출시켰다. 캘러한은 '
#                                 '레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 이 일을 했다.',
#                         'title': '오클랜드 레이더스 감독 목록'},
#                        {'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
#                                 '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
#                                 '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
#                                 '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
#                                 '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
#                                 '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
#                                 '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
#                                 '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
#                                 '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
#                                 '가장 많이 허용했던 터치다운 패스였어요.',
#                         'title': '2013년 오클랜드 레이더스 시즌입니다.'}],
#                       [{'text': '오클랜드 레이더스 시즌은 내셔널 풋볼 리그 44번째 시즌이었고, 통산 54번째, '
#                                 '그리고 데니스 앨런 감독 밑에서 두 번째였습니다. 4승 12패의 기록으로, '
#                                 '레이더스는 11년 연속 우승하지 못한 시즌을 보냈고 11년 연속 플레이오프에 '
#                                 '진출하지 못했습니다. 레이더스는 테렐 프라이어의 새로운 쿼터백으로 시즌을 '
#                                 '맞이했습니다. 프라이어는 인상적인 패션으로 시즌을 시작했고, 인디애나폴리스 '
#                                 '콜츠와의 1주차 경기에서 거의 역전승을 거뒀고, 2주차에는 잭슨빌 재규어스를 '
#                                 '물리쳤습니다. 결국 팀과 프라이어는 식었고, 결국 휴스턴 텍산스와의 경기에서 '
#                                 '프라이어가 매트 맥글로인 대신 벤치를 차지하게 되었습니다. 이 경기 전에, 이글스 '
#                                 '쿼터백 닉 폴스가 7개의 터치다운 패스를 던졌을 때, 이 경기는 레이더스가 역사상 '
#                                 '가장 많이 허용했던 터치다운 패스였어요.',
#                         'title': '2013년 오클랜드 레이더스 시즌입니다.'},
#                        {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
#                                 '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
#                                 '리시버이다.',
#                         'title': '테렐 프라이어'}],
#                       [{'text': '내셔널 풋볼 리그 오클랜드 레이더스에는 20명의 감독이 있었다. 라이더스 '
#                                 '프랜차이즈는 1959년 미국 캘리포니아주 오클랜드에서 창단됐으며 1960년 '
#                                 'NFL로 이적한 미네소타 바이킹스의 대체 선수로 아메리칸 풋볼 리그(AFL)의 '
#                                 '8번째 회원이 됐다. Raiders는 AFL-NFL 합병 이후 1970년에 '
#                                 'NFL에 합류했다. 그들은 오클랜드로 돌아가기 전 1982년과 1995년 사이에 '
#                                 '로스앤젤레스에서 뛰었다. 라이더스는 2015시즌 말 현재 AFL과 NFL에서 총 '
#                                 '56시즌 동안 852경기를 뛰었다. 이 경기에서 두 명의 코치가 팀과 함께 슈퍼볼 '
#                                 '우승을 차지했다. 1976년 존 매든, 1980년과 1983년 톰 플로레스. '
#                                 '1966년 존 라우치라는 한 코치가 AFL 챔피언십에서 우승했다. 다른 세 감독인 '
#                                 '아트 셸, 존 그루든, 빌 캘러한도 레이더스를 플레이오프에 진출시켰다. 캘러한은 '
#                                 '레이더스를 슈퍼볼로 이끌었다. 그는 감독으로서 첫 해에 이 일을 했다.',
#                         'title': '오클랜드 레이더스 감독 목록'},
#                        {'text': '테렐 프라이어 시니어(Terrelle Prior Sr., 1989년 6월 20일 '
#                                 '~ )는 미식축구 내셔널 풋볼 리그(NFL) 워싱턴 레드스킨스의 와이드 '
#                                 '리시버이다.',
#                         'title': '테렐 프라이어'}]]}
