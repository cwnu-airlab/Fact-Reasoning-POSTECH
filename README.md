# 개요

비정형 텍스트를 학습하여 쟁점별 사실과 논리적 근거 추론이 가능한 인공지능 원천기술
Artificial intelligence technology inferring issues and logically supporting facts from raw text

#### History
* (2021.04.28) FIRST INIT by 성수진
* (2021.04.30) dummy system 초안 작성by 김산
* (2021.05.06) 수정 by 성수진

#### example

* [t5 ner example](./example/) 


#### 관련 링크

* 파일 공유: [구글 드라이브](https://drive.google.com/drive/folders/1abwTalLiAhGk3c4CdMgxJtUf53fRzWzP?usp=sharing)
* 공개 github
  * https://github.com/cwnu-airlab/Fact-Reasoning-CWNU
  * https://github.com/cwnu-airlab/Fact-Reasoning-KETI
  * https://github.com/cwnu-airlab/Fact-Reasoning-YONSEI
  * https://github.com/cwnu-airlab/Fact-Reasoning-POSTECH



## 기술개발 목표

<img src='https://tva1.sinaimg.cn/large/008i3skNgy1gpzt8dpbexj30ud0u0hdt.jpg' width=70%>


## 정량 실적

#### 성능지표

  | **평가항목**                                       | **단위**      | **세계최고** **보유국/보유기업**              | **연구개발 전 국내수준** | **1차년도 목표치** | **최종개발 목표치** | 평가방법                             |
  | -------------------------------------------------- | ------------- | --------------------------------------------- | ------------------------ | ------------------ | ------------------- | ------------------------------------ |
  | 1. 비정형 텍스트 기반 근거문장 생성 정확도         | ROUGE-L       | 24.3% (미국/구글)                             | -                        | **24.3**           | **25**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 2. 트리플 및 관계 정보 기반 근거 문장 생성 정확도  | BLEU          | 14.3 (미국/알렌 연구소)                       | -                        | **14.3**           | **15**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 3. 질의 내용 기반 근거 문서 검색 정확도            | EM            | 63.82 (중국/Kingsoft AL Lab.)                 | -                        | **64**             | **65**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 4. 문서 내 지식 관계를 추출하는 기술               | F1            | 63.4 (미국/USC, Stanford Univ.)               | -                        | **65**             | **70**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 5. 멀티-홉 추론이 가능한 질의응답 기술             | EM/F1         | 74.88 (중국/Kingsoft AL Lab.)                 | -                        | **48/75**          | **49/76**           | 최종년도 공인인증 / 1차년도 자체평가 |
  | 6. 질의응답 결과 검증을 위한 자연어 추론 정확도    | Accuracy      | -                                             | -                        | **-**              | **85**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 7. 질의 분석 기반 뉴럴-심볼릭 표현 추출기술        | F1            | 65 (중국/The Chinese Univ. of Hong Kong)      | -                        | **65**             | **70**              | 최종년도 공인인증 / 1차년도 자체평가 |
  | 8. 재순위화 이후 최종 논리적 근거 문장 생성 정확도 | ROUGE-L /BLEU | 24.3 (미국/Google Ai) 14.3 (미국/알렌 연구소) | -                        | **25/15**          | **26/16**           | 최종년도 공인인증 / 1차년도 자체평가 |

#### 산출물
