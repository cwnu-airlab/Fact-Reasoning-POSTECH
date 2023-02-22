import json
import torch
import transformers

class Service:
    task = [
        {
            'name': "KG-to-Text",
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
		text = ['<ENT> {} <PRED> {} <OBJ> {}'.format(*d) for d in text]
		text = ' '.join(text)
		
		if text is None:
			return {'error':'invalid query'}
		else:
			content['supporting_facts'] = self.get_supporting_facts(text)
			content['answers'] = self.get_answers(text)
			return content

if __name__=='__main__':
	model = Model()
	inputs = [('사업주','고용','근로자'),('사업주','허용','근로시간 단축')]
	data = {'q_id':'test001',
			'supporting_facts':inputs}
	predict = model.run(content = data)
	print(predict)
