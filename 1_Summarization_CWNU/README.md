# 개요

개발한 시스템의 결과물을 공유하기 위한 API를 구축한다.
* 자연어 처리 기반 응답 근거 문장 생성 시스템

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
1. docker download
```bash
sudo apt-get install docker.io
```

2. docker group에 사용자 추가
```bash
sudo vi /etc/group
# docker: sujin
sudo service docker restart
```

3. docker build (docker image load)
```bash
docker build -t factual_reasoning/summary ./
## 이미지 목록 확인
docker image list
```

4. docker 실행
```bash
docker run --gpus all --rm -d -it -p 12341:5000 --name summary factual_reasoning/summary
## 실행 중인 컨테이너 확인
docker ps
```

5. 사용 중지
```bash
# 실행 중인 컨테이너 종료
docker stop summary

# 모든 docker 컨테이너 삭제
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

# 모든 docker 이미지 삭제
docker rmi $(docker image -q)
```
