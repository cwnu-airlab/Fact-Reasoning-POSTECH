# 개요

개발한 시스템의 결과물을 공유하기 위한 API를 구축한다.
* 지식 그래프 기반 근거 문장 생성 시스템

### 실행 방법

* 환경 설정
```bash
make set_model

or

pip install -r requirements.txt 
cd app/
python download_file_from_google_drive.py https://drive.google.com/file/d/1YPk_wQozsMmXl0iL6TUKsdjxcfoUdDQF/view?usp=sharing t5_small_ke.tar.gz
tar zxvf t5_small_ke.tar.gz
wget https://github.com/AIRC-KETI/ke-t5/raw/main/vocab/sentencepiece.model
```

* API 실행
```bash
cd app/
python -m flask run --host=0.0.0.0
```

* 시스템 동작 확인
```bash
python test.py
```

#### docker 실행

```bash
docker build -t factual_reasoning/kg2text ./
docker run --gpus all --rm -d -it -p 12342:5000 --name kg2text factual_reasoning/kg2text
```
