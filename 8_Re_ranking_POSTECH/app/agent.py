import json
import torch

from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizerFast
import torch.nn.functional as F

torch.manual_seed(42)
#random.seed(42)
#np.random.seed(42)


class Service:
    task = [
        {
            'name': 're-ranking',
            'description': 'Re-ranking answers and supporting facts that correspond to the question.'
        }
    ]

    def __init__(self):
        self.rerank_model = RerankModel()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200

    def do(self, content):
        try:
            ret = self.rerank_model.rerank_pairs(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class RerankModel(object):
    def __init__(self):
        self.device = 'cuda'

        # Initialize English Common-sense model
        config_eng = ElectraConfig.from_pretrained("google/electra-base-discriminator")
        config_eng.num_labels = 3
        self.rerank_model_eng = ElectraForSequenceClassification(config=config_eng)
        self.tokenizer_eng = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

        # Load English trained Model
        checkpoint_eng = torch.load('0525_eng.pth', map_location=self.device)
        self.rerank_model_eng.load_state_dict(checkpoint_eng['model'])
        self.rerank_model_eng.to(self.device)

        # Initialize Korean Common-sense model
        config_kor = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
        config_kor.num_labels = 3
        self.rerank_model_kor_common_sense = ElectraForSequenceClassification(config=config_kor)
        self.rerank_model_kor_legal = ElectraForSequenceClassification(config=config_kor)
        self.tokenizer_kor = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")

        # Load Korean trained Model
        checkpoint_kor_common_sense = torch.load('0513_kor.pth', map_location=self.device)
        checkpoint_kor_legal = torch.load('0513_kor.pth', map_location=self.device)

        self.rerank_model_kor_common_sense.load_state_dict(checkpoint_kor_common_sense['model'])
        self.rerank_model_kor_common_sense.to(self.device)
        self.rerank_model_kor_legal.load_state_dict(checkpoint_kor_legal['model'])
        self.rerank_model_kor_legal.to(self.device)

        self.rerank_model = None
        self.tokenizer = None
        self.domain = None
        self.language = None

    def check_input(self, content):
        question = content.get("question", None)
        answers = content.get("answer_list", None)
        supporting_facts = content.get("supporting_facts", None)

        if not question or not answers or not supporting_facts:
            return {
                'error': "invalid query"
            }

        if len(answers) != len(supporting_facts):
            return {
                'error': "(list of answer) and (list of supporting fact) should have same length for ranking"
            }

        self.domain = question['domain']
        self.language = question['language']

        if self.language == 'kr':
            if self.domain == 'common-sense':
                self.rerank_model = self.rerank_model_kor_common_sense
            else:
                self.rerank_model = self.rerank_model_kor_legal
            self.tokenizer = self.tokenizer_kor
        else:
            self.rerank_model = self.rerank_model_eng
            self.tokenizer = self.tokenizer_eng

        return question, answers, supporting_facts

    def rerank_pairs(self, content):
        output = self.check_input(content)
        question, answers, supporting_facts = self.check_input(content)

        try:
            score = []
            for answer, supporting_fact in list(zip(answers, supporting_facts)):
                score.append(self.get_score(question['text'], answer, supporting_fact))

            score_softmax = F.softmax(torch.FloatTensor(score))
            score_softmax = [float(elem) for elem in score_softmax]
            content["reranking_score"] = score_softmax

            return content
        except Exception as e:
            return {'error': "{}".format(e)}

    def get_score(self, question, answer, supporting_fact):
        hypothesis = question + ' ' + answer
        premise = supporting_fact

        encoded_input = self.tokenizer.batch_encode_plus([(premise, hypothesis)],
                                                         padding="max_length", max_length=256,
                                                         truncation=True,
                                                         return_tensors='pt')

        self.rerank_model.eval()
        with torch.no_grad():
            outputs = self.rerank_model(input_ids=encoded_input["input_ids"].to(self.device),
                                        attention_mask=encoded_input["attention_mask"].to(self.device))

            score = F.softmax(outputs.logits).squeeze()[0]
            return float(score)

if __name__=="__main__":
    data = {'question': {'text': '샤를 드골 대통령과 콘라드 아데나워 총리가 서명한 조약은 어느 도시에서 이루어졌는가?', 'language': 'kr', 'domain': 'common-sense'}, 
            'answer_list': ['부다페스트', '헝가리의 항구, 오클라호마와 벨라우즈 타이완 주입니다. argues) 베스르', '헝가리의 항구, 오클라호마와 벨라우즈 타이완 주입니다. argues) 로르'], 
            'supporting_facts': ['아제르외그(Azördög)는 헝가리의 연극으로,', '폴 거리의 소년》(The Boys of Paul Street, 헝가리어: A Pál', '몰나르는 부다페스트에서 태어났다. 그는 제2차 세계 대전 동안 헝가리 유대인에 대한 박해']} 
    print(data)
    model = RerankModel()
    print(model)
    output = model.rerank_pairs(data)
    print(output)
