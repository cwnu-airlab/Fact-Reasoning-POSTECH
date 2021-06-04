import json
import torch
import transformers

class Service:
    task = [
        {
            'name': "text-summarization",
            'description': 'dummy system'
        }
    ]

    def __init__(self):
        self.model = Model()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.model.run(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400

class Model(object):

	def __init__(self):
		tokenizer_path = 'model/sentencepiece.model'
		self.tokenizer = transformers.T5Tokenizer.from_pretrained(tokenizer_path)
		model_path = {
				'supporting_facts':'model/',
				'answers':'model/'
				}
		self.model = {
				'supporting_facts':transformers.T5ForConditionalGeneration.from_pretrained(model_path['supporting_facts']),
				'answers':transformers.T5ForConditionalGeneration.from_pretrained(model_path['answers'])
				}
	
	def get_ids(self, sentence):
		input_ids = self.tokenizer.encode(sentence)
		input_ids = torch.tensor([input_ids])
		return input_ids

	def get_supporting_facts(self, inputs):
		inputs = self.get_ids(inputs)
		predict = self.model['supporting_facts'].generate(inputs, num_beams=3, num_return_sequences=3)
		
		output = list()
		for pred in predict:
			pred = self.tokenizer.decode(pred)
			output.append(pred)
		return output

	def get_answers(self, inputs):
		inputs = self.get_ids(inputs)
		predict = self.model['answers'].generate(inputs, num_beams=3, num_return_sequences=3)
		
		output = list()
		for pred in predict:
			pred = self.tokenizer.decode(pred)
			output.append(pred)
		return output


	def run(self, content):
		text = content.get('supporting_facts',None)
		
		if text is None:
			return {'error':'invalid query'}
		else:
			content['supporting_facts'] = self.get_supporting_facts(text)
			content['answers'] = self.get_answers(text)
			return content

if __name__=='__main__':
	model = Model()
	inputs = '19세기 후반, 작가 아서 코난 도일에 의해 탄생한 ‘명탐정’ 셜록 홈즈는 추리 소설 마니아뿐만 아니라 성장기의 청소년에게 참 많은 영향을 끼친 인물이다. 명석하게 사건을 해결하는 탐정 캐릭터를 떠올릴 때 가장 먼저 떠오르는 근대 인물 중 하나가 되었으니까. 셜록과 그의 조수이자 동료인 왓슨을 주요 등장인물로 설정한, 또는 그에서 모티브를 가져와 제작된 영화 및 드라마는 수 없이 많다.'
	data = {'q_id':'test001',
			'supporting_facts':inputs}
	predict = model.run(content = data)
	print(predict)
