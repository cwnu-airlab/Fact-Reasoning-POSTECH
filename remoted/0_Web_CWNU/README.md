# 개요

개발한 시스템의 결과물을 공유하기 위한 API를 구축한다.
* 결과 확인을 위한 web 페이지

### 실행 방법

* 환경 설정
```bash
cd app/
pip install -r requirements.txt 
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
docker build -t factual_reasoning/web ./
docker run --gpus all --rm -d -it -p 12340:5000 --name web factual_reasoning/web
```
