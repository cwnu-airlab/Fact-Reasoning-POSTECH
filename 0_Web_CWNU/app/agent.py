import json
import requests
import logging
from urllib.parse import urljoin

SPACE = '\t'

def make_logger(name=None):
	#1 logger instance를 만든다.
	logger = logging.getLogger(name)

	#2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
	logger.setLevel(logging.DEBUG)

	#3 formatter 지정
	formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

	#4 handler instance 생성
	console = logging.StreamHandler()
	file_handler = logging.FileHandler(filename="log.log")

	#5 handler 별로 다른 level 설정
	console.setLevel(logging.INFO)
	file_handler.setLevel(logging.DEBUG)

	#6 handler 출력 format 지정
	console.setFormatter(formatter)
	file_handler.setFormatter(formatter)

	#7 logger에 handler 추가
	logger.addHandler(console)
	logger.addHandler(file_handler)

	return logger

logger = make_logger()


class Service:
	task = [
		{
			'name': "text-summarization",
			'description': 'dummy system'
		}
	]

	def __init__(self):
		self.headers = {'Content-Type': 'application/json; charset=utf-8'} # optional
		self.sample_questions = [
				'데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?',
				'샤이닝을 부른 그룹 보컬의 고향은?',
				]
		pass
	
	@classmethod
	def get_task_list(cls):
		return json.dumps(cls.task), 200
	
	def do(self, content):
		logger.info(content)
		if content['question'].strip() == '':
			content['question'] = content['example']
		try:
			ret = self.predict(content)
			if 'error' in ret.keys():
				return json.dumps(ret), 400
			return json.dumps(ret), 200
		except Exception as e:
			return json.dumps(
				{
					'error': "{}".format(e)
				}
			), 400

	##TODO 예외처리하기 각 기관에서 오류 발생 시(400) '서비스명-기관: 에러내용' 출력하고 입력은 정상값으로 대치해서 다음으로 전달
	def get_output(self,url,data):
		data = json.dumps(data)
		try:
			response = requests.post(urljoin(url, '/api/task'), data=data, headers=self.headers)
			return response.json(), response.status_code
		except requests.exceptions.ConnectionError as e:
			error = {'error':'{}'.format(e)}
			return error, 521
			

	def get_passage_retrieval(self, question):
		data = {'query':question}
		out, status = self.get_output('http://thor.nlp.wo.tc:12343/',data)
		if status == 200:
			return out, data
		else:
			out['error'] = f"[{status}]{out['error']}"
			return out, data

	def get_relation_extraction(self, question):
		data = {'doc':question}
		out, status = self.get_output('http://thor.nlp.wo.tc:12344/',data)
		if status == 200:
			return out, data
		else:
			out['error'] = f"[{status}]{out['error']}"
			return out, data

	def get_summarize(self, question, supportings):
		try:
			contexts = [d['text'] for d in supportings['retrieved_doc']]
			
			outs = {'answers':list(), 'supporting_facts':list()}
			for context in contexts:
				data = {'q_id':question, 'supporting_facts':context}
				out, status = self.get_output('http://thor.nlp.wo.tc:12341/',data)
				if status == 200:
					outs['supporting_facts'].append( out['supporting_facts'] )
					outs['answers'].append( out['answers'] )
				else:
					out['error'] = f"[{status}]{out['error']}"
					return out, data
			###XXX
			outs['answers'] = outs['answers'][-1]
			outs['supporting_facts'] = outs['supporting_facts'][-1]
			####
			return outs, data
		except (KeyError) as e:
			out = {'error':f"KeyError:{e}"}
			return out, {'q_id':question, 'supporting_facts':supportings}

	def get_kg_retrieval(self, question, supportings):
		data = {'query':question}
		out, status = self.get_output('http://thor.nlp.wo.tc:12345/',data)
		if status == 200:
			return out, data
		else:
			out['error'] = f"[{status}]{out['error']}"
			return out, data

	def get_kg2Text(self, question, supportings):
		try:
			contexts = supportings['supporting facts']
			
			data = {'q_id':question,'supporting_fact':contexts}
			out, status = self.get_output('http://thor.nlp.wo.tc:12342/',data)
			if status == 200:
				out.pop('q_id') ##XXX
				return out, data
			else:
				out['error'] = f"[{status}] {out['error']}"
				return out, data
		except (KeyError) as e:
			out = {'error':f"KeyError:{e}"}
			return out, {'q_id':question, 'supporting_facts':supportings}
	
	def get_rerank(self, question, answers, supportings):
		data = {'question':question, 'answers':answers, 'supporting_facts':supportings}
		out, status = self.get_output('http://thor.nlp.wo.tc:12348/',data)
		if status == 200:
			out['supporting_facts'] = [[d] for d in out['supporting_facts']]
			return out, data
		else:
			out['error'] = f"[{status}]{out['error']}"
			return out, data
	##TODO/

	
	def predict(self, text):
		if text is None: return {'error':'invalid query'}

		question = text['question']
		system_result = dict()

		## 3 PassageRetrieval
		supporting_passage, input_3 = self.get_passage_retrieval(question)
		system_result[3] = {'name':'PassageRetrieval','manager':'KETI','input':input_3,'output':supporting_passage}

		## 1 summarizer
		seq2seq_output, input_1 = self.get_summarize(question, supporting_passage)
		system_result[1] = {'name':'Summarizer','manager':'CWNU','input':input_1 ,'output':seq2seq_output}

		## 4 RelationExtraction
		supporting_graph, input_4 = self.get_relation_extraction(question)
		system_result[4] = {'name':'RelationRetrieval','manager':'KETI','input':input_4 ,'output':supporting_graph}

		## 5 KGRetrieval ##TODO 6 대신 사용 중, 수정 필요
		merged_supporting_graph, input_5 = self.get_kg_retrieval(question, supporting_graph)
		system_result[5] = {'name':'KnowledgeRetrieval','manager':'YONSEI','input':input_5 ,'output':merged_supporting_graph}

		## 2 kg2Text
		kg2seq_output, input_2 = self.get_kg2Text(question, merged_supporting_graph)
		system_result[2] = {'name':'KG2Text','manager':'CWNU','input':input_2 ,'output':kg2seq_output}

		## output merge
		seq_output = {d:(seq2seq_output[d] if d in seq2seq_output else [])+(kg2seq_output[d] if d in kg2seq_output else [])
						for d in ['answers','supporting_facts']}

		## 8 ReRanking
		rerank_output, input_8 = self.get_rerank(question, seq_output['answers'], seq_output['supporting_facts'])
		system_result[8] = {'name':'ReRanking','manager':'POSTECH','input':input_8 ,'output':rerank_output}

		####### make output context
		if question in self.sample_questions:
			system_result['final_output'] = self.sample(question)
		else:
			try:
				system_result['final_output'] = [f"{rerank_output['answers'][i]}\n    ･{'    ･'.join(rerank_output['supporting_facts'][i])}".replace('    ','\t')
							for i in range(len(rerank_output['answers']))]
			except (KeyError) as e:
				system_result['final_output'] = [f"KeyError:{e}"]	
		return {'output':get_result_text(system_result)}

	def sample(self,text):
		if text == '데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?':
			return [
					'･Answer: 아니요, 다른 사람입니다.',
					'･Supporting Facts:',
					'\t･[<a href="https://ko.wikipedia.org/wiki/데드풀_(영화)"> 데드풀_(영화)] </a>]',
					'\t\t시각효과와 애니메이션 연출자였던 팀 밀러가 감독을 맡았고 렛 리스와 폴 워닉이 각본을 썼다.',
					'\t･[<a href="https://ko.wikipedia.org/wiki/킬러의_보디가드"> 킬러의 보디가드 </a>]',
					'\t\t패트릭 휴스 감독이 연출하고 라이언 레이놀즈, 새뮤얼 L. 잭슨, 게리 올드먼, 엘로디 융, 살마 아예크가 출연한다.'
					]
		elif text == '샤이닝을 부른 그룹 보컬의 고향은?':
			return [
					'･ Answer: 서울',
					'･ Supporting Facts:',
					'\t･ [<a href="https://ko.wikipedia.org/wiki/자우림"> 자우림 </a>]',
					'\t\t자우림(紫雨林)은 대한민국의 3인조 혼성 록 밴드이다. 기타를 맡은 이선규와 보컬의 김윤아, 베이스 기타의 김진만으로 구성되어 있으며, 드럼의 구태훈은 탈퇴하였다.',
					'\t･ [<a href="https://ko.wikipedia.org/wiki/김윤아"> 김윤아 </a>]',
					'\t\t김윤아는 대한민국 서울특별시 강남구에서 태어났다.',
					'\t\t(김윤아, 출생, 대한민국 서울특별시 강남구)'
					]
		else:
			raise KeyError('"{}" is not in sample list.'.format(text))

def get_result_text(system_result):

	result = []
	result += ['\n'.join(system_result['final_output']).replace('\t','&nbsp;'*4),'\n\n']
	result += ['<details style="border:1px solid #aaa;border-radius:4px" open="open">\n']
	result += ['<summary style="text-align:center; font-weight:bold"> Real Result of System </summary>']
	for key in [3,1,4,5,2,8]:
		system_result[key]['input'] = json.dumps(system_result[key]['input'], indent=4, ensure_ascii=False)
		system_result[key]['input'] = system_result[key]['input'].replace(' ','&nbsp;')
		if 'error' in system_result[key]['output']:
			title = '<summary>'+html_font(f"<{key}_{system_result[key]['name']}-{system_result[key]['manager']}>",cls='error_comment',color='#FAC3C3')+'</summary>'
			system_result[key]['output'] = html_font(system_result[key]['output']['error'],cls='error_comment',color='#FFFFFF',font_color='red')
			output = f"<details open='open' style='border:1px solid #aaa;border-radius:4px'> <summary>Output</summary> {system_result[key]['output']}</details>"
		else:
			title = '<summary>'+html_font(f"<{key}_{system_result[key]['name']}-{system_result[key]['manager']}>")+'</summary>'
			system_result[key]['output'] = json.dumps(system_result[key]['output'], indent=4, ensure_ascii=False)
			system_result[key]['output'] = system_result[key]['output'].replace(' ','&nbsp;')
			output = f"<details style='border:1px solid #aaa;border-radius:4px'> <summary>Output</summary> {system_result[key]['output']}</details>"
		text = ['<details>',
				title,
				f"<details style='border:1px solid #aaa;border-radius:4px'> <summary>Input</summary> {system_result[key]['input']}</details> ",
				output,
				'</details>']
		result += '\n'.join(text)+'\n'
	result += ['</details>\n']
	result = ''.join(result)
	return result.replace('\n','<br/>')


def html_font(text,cls='result_head',color='powderblue',font_color='#000000'):
	return f'<a class="{cls}" style="background-color:{color};font-weight:bold;color:{font_color}">'+text+'</a>'


if __name__=='__main__':
	service = Service()
	content = {'question':{'text': '', 'cli_ip': '10.100.54.146', 'domain': 'common-sense', 'example': '데드풀 감독이랑  킬러의 보디가드 감독이 같은 사람이야?', 'language': 'kr'}}
	predict = service.do(content = content)
