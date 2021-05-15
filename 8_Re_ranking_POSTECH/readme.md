## 8_Re_ranking_POSTECH
###input 형식
질문(question) 및 여러 모듈의 answer list, supporting_fact list가 들어갑니다.

supporting fact는 각 모듈당 하나의 문장(str)으로 concat해서 넣어주세요.
```{
        "question": "Were both of the following rock groups formed in California: Dig and Thinking Fellers Union Local 282?",
        "answers": ["yes",
                    "Califonia",
                    "no"],
        "supporting_facts": [
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco, California, though half of its members are from Iowa.",
            "Strangers from the Universe is the fifth album by Thinking Fellers Union Local 282, released on September 12, 1994 through Matador Records. Mother of All Saints is the fourth album by Thinking Fellers Union Local 282, released as a CD and double-LP on November 13, 1992 through Matador Records.",
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco"]
}
```

###output 형식
각 answer, supporting fact pair에 대한 점수(score)가 같이 나옵니다.

```{
        "question": "Were both of the following rock groups formed in California: Dig and Thinking Fellers Union Local 282?",
        "answers": ["yes",
                    "Califonia",
                    "no"],
        "supporting_facts": [
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco, California, though half of its members are from Iowa.",
            "Strangers from the Universe is the fifth album by Thinking Fellers Union Local 282, released on September 12, 1994 through Matador Records. Mother of All Saints is the fourth album by Thinking Fellers Union Local 282, released as a CD and double-LP on November 13, 1992 through Matador Records.",
            "Dig is an American alternative rock band from Los Angeles, California. Thinking Fellers Union Local 282 is an experimental indie rock group formed in 1986 in San Francisco"],
        "score": [0.4492, 0.1582,  0.3926]
}
```
