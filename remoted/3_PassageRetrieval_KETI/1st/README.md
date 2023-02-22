<!--
 Copyright 2021 san kim
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

## Passage Retrieval

URL: http://aircketi.iptime.org:10021/


## Google driver links


[ket5-base-ko-0929](https://drive.google.com/file/d/1d8FgJPZ5my-VAdyd7HPQoWzzCP1FGSg9/view?usp=sharing)
[ket5-base-en](https://drive.google.com/file/d/1VR3npo-Th3mA0Dfw-_L0aB1yXIr3UJx6/view?usp=sharing)

## Download models from Google drive

```bash
    python download_file_from_google_drive.py https://drive.google.com/file/d/1d8FgJPZ5my-VAdyd7HPQoWzzCP1FGSg9/view?usp=sharing T5EncoderSimpleMomentumRetriever_ket5-base-ko-0929_weights.tgz
    tar xvzf T5EncoderSimpleMomentumRetriever_ket5-base-ko-0929_weights.tgz

    python download_file_from_google_drive.py https://drive.google.com/file/d/1VR3npo-Th3mA0Dfw-_L0aB1yXIr3UJx6/view?usp=sharing T5EncoderSimpleMomentumRetriever_ket5-base-en_weights.tgz
    tar xvzf T5EncoderSimpleMomentumRetriever_ket5-base-en_weights.tgz
    
```

## Build and run docker

```bash
docker build -t factual_reasoning/passage_retrieval .
docker run --gpus all --rm -d -it -p 12345:5000 --name passage_retrieval factual_reasoning/passage_retrieval
```

## Test

after modify the URL variable in `test.py`...

```bash
python test.py
```


# Performance

## KE-T5 En Common-sense across top_k

```json
{
    "1": {
        "em": 0.47589466576637407,
        "prec": 0.675151924375422,
        "recall": 0.675151924375422,
        "f1": 0.675151924375422
    },
    "2": {
        "em": 0.5978392977717758,
        "prec": 0.7661039837947333,
        "recall": 0.7661039837947333,
        "f1": 0.7661039837947333
    },
    "4":{
        "em": 0.7151924375422012,        
        "prec": 0.8429439567859555,        
        "recall": 0.8429439567859555,
        "f1": 0.8429439567859555            
    },
    "6": {
        "em": 0.7778528021607022,
        "prec": 0.8808912896691424,
        "recall": 0.8808912896691424,
        "f1": 0.8808912896691424
    },
    "8": {
        "em": 0.8209318028359217,
        "prec": 0.9055367994598245,
        "recall": 0.9055367994598245,
        "f1": 0.9055367994598245
    },
    "10": {
        "em": 0.8536124240378122,
        "prec": 0.92390276839973,
        "recall": 0.92390276839973,
        "f1": 0.92390276839973
    },
    "12": {
        "em": 0.8826468602295746,
        "prec": 0.9390952059419311,
        "recall": 0.9390952059419311,
        "f1": 0.9390952059419311
    },
    "14": {
        "em": 0.9064145847400406,
        "prec": 0.9518568534773801,
        "recall": 0.9518568534773801,
        "f1": 0.9518568534773801
    },
    "16": {
        "em": 0.92491559756921,
        "prec": 0.9619851451721809,
        "recall": 0.9619851451721809,
        "f1": 0.9619851451721809
    },
    "18": {
        "em": 0.9438217420661715,
        "prec": 0.9719108710330857,
        "recall": 0.9719108710330857,
        "f1": 0.9719108710330857
    },
    "20": {
        "em": 0.9609723160027008,
        "prec": 0.9804861580013504,
        "recall": 0.9804861580013504,
        "f1": 0.9804861580013504
    }
}
```

## KE-T5 Ko Common-sense across top_k

```json
{
    "1": {
        "em": 0.346952296819788,
        "prec": 0.585136925795053,
        "recall": 0.5850633097762074,
        "f1": 0.5850927561837457
    },
    "2": {
        "em": 0.4712897526501767,
        "prec": 0.683303886925795,
        "recall": 0.6831566548881037,
        "f1": 0.6832155477031803
    },
    "4": {
        "em": 0.5896643109540636,
        "prec": 0.7699867491166078,
        "recall": 0.7698395170789164,
        "f1": 0.769898409893993
    },
    "6": {
        "em": 0.6561395759717314,
        "prec": 0.8122791519434629,
        "recall": 0.8121319199057714,
        "f1": 0.8121908127208481
    },
    "8": {
        "em": 0.7100265017667845,
        "prec": 0.8454063604240283,
        "recall": 0.8452591283863368,
        "f1": 0.8453180212014135
    },
    "10": {
        "em": 0.7519876325088339,
        "prec": 0.8699204946996466,
        "recall": 0.8696996466431094,
        "f1": 0.8697879858657245
    },
    "12": {
        "em": 0.7888692579505301,
        "prec": 0.8905697879858657,
        "recall": 0.8904225559481743,
        "f1": 0.890481448763251
    },
    "14": {
        "em": 0.818904593639576,
        "prec": 0.9077959363957597,
        "recall": 0.9077223203769139,
        "f1": 0.9077517667844524
    },
    "16": {
        "em": 0.8478356890459364,
        "prec": 0.9235865724381626,
        "recall": 0.9235129564193167,
        "f1": 0.9235424028268552
    },
    "18": {
        "em": 0.8714664310954063,
        "prec": 0.9358436395759717,
        "recall": 0.9357700235571259,
        "f1": 0.9357994699646643
    },
    "20": {
        "em": 0.8975265017667845,
        "prec": 0.9487632508833922,
        "recall": 0.9487632508833922,
        "f1": 0.9487632508833922
    }
}
```


