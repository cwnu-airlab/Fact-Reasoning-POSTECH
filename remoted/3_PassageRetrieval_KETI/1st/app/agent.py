
import json
import random

import torch

from transformers import AutoTokenizer

from mhop_dataset import QueryPassageFormatter
from passage_loader import SearchManager

from mhop_retriever import *
from mhop_retriever import load_model

class Service:
    task = [
        {
            'name': "passage_retrieval",
            'description': 'dummy task'
        }
    ]

    def __init__(self):
        self.retriever = Retriever()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.retriever.do_search(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


# {
#   "question":{
#     "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
#     "language":"kr",
#     "domain":"common-sense"
#   },
#   "max_num_retrieved":10,
#   "max_hop":5,
#   "num_retrieved":-1
# }


class Retriever(object):
    def __init__(self):
        self.config = json.load(open("config.json", "r"))

        self.search_manager = {
            comb: self.load_search_manager(self.config[comb]) for comb in ["kr_common-sense", "en_common-sense"]
        }
    
    @staticmethod
    def load_search_manager(model_cfg):
        model_class = load_model(model_cfg["model_name"])
        model = model_class.from_pretrained(model_cfg["hf_path"])
        model.eval()

        pre_trained_tokenizer = model_cfg.get("pre_trained_tokenizer", "KETI-AIR/ke-t5-base")
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_tokenizer)
        qp_formatter = QueryPassageFormatter(
            tokenizer,
            max_q_len=70,
            max_q_sp_len=350,
            max_c_len=350,
            remove_question_mark=False,
            add_cls_token=True,
            add_sep_token=True,
            add_token_type_ids=False,
            cls_token_id=-1,
            sep_token_id=-1,
        )
        search_manager = SearchManager(
            qp_formatter,
            model,
            page_info_path=None,
            exhaustive_scorer=False,
            normalize=False,
            use_gpu=model_cfg["use_gpu"],
            model_gpu=model_cfg["model_gpu"],
            scorer_gpu=model_cfg["scorer_gpu"],
        )
        if model_cfg["use_gpu"]:
            search_manager.cuda()
        
        return search_manager
    
    def do_search(self, content):
        question = content.get('question', None)
        context = content.get('context', None)
        if question is None:
            return {
                'error': "There is no question!!!"
            }
        elif context is None:
            return {
                'error': "You have to pass context set. But got Null context."
            }
        else:
            context = self.convert_context(context)
            question_txt = question.get('text', '')
            question_ln = question.get('language', 'kr')
            question_domain = question.get('domain', 'common-sense')
            max_num_retrieved = content.get('max_num_retrieved', 10)
            max_hop = content.get('max_hop', 2)
            num_retrieved = content.get('num_retrieved', -1)

            if question_txt == '':
                return {
                    'error': "Empty question string!!!"
                }

            query = {
                "question": question_txt,
                "context": context,
            }

            comb_str = "{}_{}".format(question_ln, question_domain)
            if comb_str in self.search_manager:
                return self.search(
                    comb_str, 
                    query, 
                    max_num_retrieved=max_num_retrieved,
                    max_hop=max_hop,
                    num_retrieved=num_retrieved)
            else:
                return {
                    'error': f"The requested combination of language and question domain is currently unsupported. \
                        (kr, common-sense), (en, common-sense) are currently supported on this service. \
                            But got ({question_ln},{question_domain})"
                }
    
    @staticmethod
    def convert_context(context):
        return [{'title': item[0], 'text':' '.join(item[1])} for item in context]

    
    @torch.no_grad()
    def search(self, comb_str, query, max_num_retrieved=10, max_hop=2, num_retrieved=-1):
        max_num_retrieved = max(4, max_num_retrieved)
        retrived_items = self.search_manager[comb_str].search(query, top_k=max_num_retrieved, n_hop=max_hop, num_cands_per_topk=2)

        if num_retrieved > 1:
            max_num_retrieved = num_retrieved

        top_n_docs = self.get_n_docs(retrived_items, max_num_retrieved=max_num_retrieved)
        
        return {
            'num_retrieved_doc': len(top_n_docs),
            'retrieved_doc': top_n_docs,
            'top_n_candidates': retrived_items[:max_num_retrieved]
        }

    
    def get_n_docs(self, items, max_num_retrieved=10):
        title_set = set()
        result_set = []

        for item in items:
            for ctx in item:
                if ctx['title'] not in title_set:
                    result_set.append(ctx)
                    title_set.add(ctx['title'])

                if len(title_set) >= max_num_retrieved:
                    return result_set
        return result_set


if __name__ == "__main__":

    example_content = {
        'question': {
                'text': '매거리 컨트리 파크가 위치한 계획된 정착지의 건설은 몇 년도에 시작되었는가?',
                'language': 'kr',
                'domain': 'common-sense',
                
            }, 
        'context': [
            [
                '코스스턴 레이크스 컨트리 파크', 
                [
                    '코스메스턴 레이크스 컨트리 파크는 글래모건 시의 베일이 소유하고 관리하는 영국의 공공 컨트리 파크이다.', 
                    '그것은 카디프에서 7.3마일 (11.7 킬로미터) 떨어진 글래모건의 베일 페나스와 설리 사이에 위치해 있습니다.', 
                    '2013년 5월 1일 시골공원은 지역자연보호구역 LNR로 지정되었다.', 
                    '부품은 특별한 과학적 관심의 장소입니다.', 
                    '공원, 방문객 센터, 카페는 일년 내내 문을 연다.'
                ]
            ], 
            [
                '크레이가본', 
                [
                    '크레이거본( )은 북아일랜드 아마 주의 주도이다.', 
                    '그것의 건설은 1965년에 시작되었고 북아일랜드의 초대 총리인 제임스 크레이그의 이름을 따서 지어졌습니다.', 
                    '그것은 러건과 포트다운을 통합한 새로운 직선 도시의 심장부가 될 예정이었지만, 이 계획은 버려졌고 제안된 작업의 절반도 되지 않았다.', 
                    '오늘날 지역 주민들 사이에서 "크레이가본"은 두 마을 사이의 지역을 가리킨다.', 
                    '두 개의 인공 호수 옆에 지어졌으며 넓은 주거 지역(브라운로우), 두 번째로 작은 지역(맨더빌), 그리고 실질적인 쇼핑 센터, 법원 및 구의회 본부를 포함하는 중심 지역(하이필드)으로 구성되어 있다.', 
                    '야생동물의 안식처인 호수는 산책로가 있는 삼림지대로 둘러싸여 있다.', 
                    '이 지역에는 수상 스포츠 센터, 애완 동물원과 골프 코스, 스키 슬로프도 있습니다.', 
                    '대부분의 크레이가본에서는 자동차가 보행자와 완전히 분리되어 있으며, 회전교차가 광범위하게 사용된다.'
                ]
            ], 
            [
                '우드게이트 밸리 컨트리 파크', 
                [
                    '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다.', 
                    '그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다.', 
                    '이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'
                ]
            ], 
            [
                '룽푸산 컨트리 파크', 
                [
                    '룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다.', 
                    "사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다.", 
                    '중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다.', 
                    '그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다.', 
                    'Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다.', 
                    '이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다.', '이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다.'
                ]
            ], 
            [
                '마헤리 컨트리 파크', 
                [
                    'Maghery Country Park는 북아일랜드 아마 주 County Amagh의 Maghery 마을에 있는 공원입니다.', 
                    '그것은 30에이커의 면적에 걸쳐 있고 5킬로미터의 삼림 산책로와 피크닉 장소를 포함하고 있으며 조류 관찰, 낚시, 그리고 산책을 위해 사용된다.', 
                    '코니 아일랜드는 해안에서 1km 떨어져 있고 주말에는 공원에서 보트 여행을 할 수 있습니다.', 
                    '이곳은 중요한 지역 편의 시설이자 관광 명소이며 크레이가본 구의회에서 관리하고 있다.'
                ]
            ], 
            [
                '킹피셔 컨트리 파크', 
                [
                    '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다.', 
                    '그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다.', 
                    '처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다.', 
                    '이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다.', 
                    '이것은 지역 자연 보호 구역입니다.'
                ]
            ], 
            [
                '플로버 코브 컨트리 파크', 
                [
                    '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다.', 
                    '최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다.', 
                    '1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'
                ]
            ], 
            [
                '팻신렝 컨트리 파크', 
                [
                    '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다.', 
                    '1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다.', 
                    '그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다.', 
                    '호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'
                ]
            ], 
            [
                '코니 아일랜드, 러프 니', 
                [
                    '코니 아일랜드는 북아일랜드의 러프 니에 있는 섬이다.', 
                    '아마흐 카운티의 마헤리에서 약 1km 떨어진 곳에 위치해 있으며 숲이 우거져 있고 면적은 약 9에이커에 달한다.', 
                    '그것은 Lough Neagh의 남서쪽 모퉁이에 있는 Blackwater 강과 Bann 강 어귀 사이에 있다.', 
                    '섬으로의 보트 여행은 Maghery Country Park 또는 Kinnego Marina에서 주말에 이용할 수 있습니다.', 
                    '이 섬은 내셔널 트러스트가 소유하고 있으며 크레이거본 자치구가 그들을 대신하여 관리하고 있다.', 
                    '코니 아일랜드 플랫은 섬에 인접한 바위 돌출부입니다.', 
                    '새뮤얼 루이스가 코니 섬을 아마그 카운티의 유일한 섬이라고 불렀지만, 아마그의 Lough Neagh 구역에는 Croaghan 섬과 Padian, Rathlin 섬, Derrywaragh Island의 변두리 사례도 포함되어 있습니다.'
                ]
            ], 
            [
                '발록 컨트리 파크', 
                [
                    '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다.', 
                    '1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다.', 
                    '발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다.', 
                    '원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다.', 
                    'Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'
                ]
            ]
        ], 
        'max_num_retrieved': 6
    }

    # ['마헤리 컨트리 파크', '0'], ['크레이가본', '1']

    test_model = Retriever()
    ret = test_model.do_search(example_content)
    print(ret)




# {'num_retrieved_doc': 6, 
# 'retrieved_doc': [
#     {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#     {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#     {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}, 
#     {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#     {'title': '킹피셔 컨트리 파크', 'text': '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다. 그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다. 처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다. 이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다. 이것은 지역 자연 보호 구역입니다.'}, 
#     {'title': '발록 컨트리 파크', 'text': '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다. 1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다. 발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다. 원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다. Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'}], 
#     'top_n_candidates': [
#         [
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}], 
#         [
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#             {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}], 
#         [
#             {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#             {'title': '킹피셔 컨트리 파크', 'text': '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다. 그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다. 처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다. 이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다. 이것은 지역 자연 보호 구역입니다.'}], 
#         [   
#             {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#             {'title': '발록 컨트리 파크', 'text': '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다. 1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다. 발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다. 원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다. Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'}], 
#         [   
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#             {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}], 
#         [   
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}]]}


