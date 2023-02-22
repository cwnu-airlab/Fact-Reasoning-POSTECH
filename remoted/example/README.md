더미 파일을 만드는 방식은 **2_PassageRetrieval_KETI** 또는 **3_RelationExtraction_KETI**를 참조하시기 바랍니다.

실제 pytorch를 이용하여 모델을 만들었을 경우에 대한 예는 **example** 디렉토리를 참고하시면 됩니다.

example에서는 dockerfile에서 git에 공개한 코드를 클론한 뒤 코드들의 위치를 옮기는 식으로 해놨지만, 직접 폴더에 복사하셔서 만드는게 더 간단하긴 합니다.

코드를 옮기는 번거로움을 없애긴 위해선 코드를 setup.py와 같은 방식으로 패키징해서 사용하시면 됩니다.

## Example

1. Download model

T5 encoder ner.tar.gz: [Download](https://drive.google.com/file/d/14BnwwTZExHoTKyyoyDi3LKL3vH9TJpcT/view?usp=sharing)

```bash
    python download_file_from_google_drive.py https://drive.google.com/file/d/14BnwwTZExHoTKyyoyDi3LKL3vH9TJpcT/view?usp=sharing t5_encoder_ner.tar.gz
```

2. extract files

```bash
    tar xvzf t5_encoder_ner.tar.gz
```

3. build the docker image and run it

```bash
    docker build -t factual_reasoning/ner_example .
    docker run --gpus all --rm -d -it -p 12345:5000 --name ner_example factual_reasoning/ner_example
```

4. test the app

``` bash
    python test.py
```
