import sys
import io
import requests

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	print('get_session', flush=True)
	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	print('get_response', flush=True)
	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	print('save_content', flush=True)
	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

if __name__ == "__main__":
	## 구글 드라이브 공유 링크를 직접 코드에서 다운로드받을 수 있도록 수정
	print(sys.argv)
	url = sys.argv[1]
	file_id = url.split('/')[5]
	destination = sys.argv[2]
	download_file_from_google_drive(file_id, destination)
