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

## Relation Extraction

URL: http://ketiair.com:10022/

## Google driver links

[ket5-base-ko](https://drive.google.com/file/d/1Xwd0xrp2eEbAbpnUWdLdG0U3kYeUd1q-/view?usp=sharing)

## Download models from Google drive

```bash
    python download_file_from_google_drive.py https://drive.google.com/file/d/1Xwd0xrp2eEbAbpnUWdLdG0U3kYeUd1q-/view?usp=sharing T5EncoderForSequenceClassificationFirstSubmeanObjmean_ket5-base-ko_weights.tgz
    tar xvzf T5EncoderForSequenceClassificationFirstSubmeanObjmean_ket5-base-ko_weights.tgz
```

## Build and run docker

```bash
docker build -t factual_reasoning/relation_extraction .
docker run --gpus all --rm -d -it -p 12345:5000 --name relation_extraction factual_reasoning/relation_extraction
```

## Test

after modify the URL variable in `test.py`

```bash
python test.py
```

