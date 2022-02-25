---
layout: single
title:  "[study] 정규표현식_Regex"
categories: study
tag: [python, re, linux, shell script]
toc: true
author_profile: false
---

# 들어가며. 
코딩, 리눅스, shell script를 짜다보면 종종 사용할일이 있는데,<br/>
이번 기회에 정리를 해야겠습니다.!!!
<br/><br/><br/><br/>

# Reference
<ul>
<li><a href="https://www.ibm.com/docs/ko/control-center/6.1.1?topic=reference-regular-expressions" target="_blank">IBM 정규표현식</a>
<li><a href="https://wikidocs.net/80567" target="_blank">토닥토닥 파이썬 정규표현식</a></li>
</ul>
<br/><br/><br/><br/>


# 연습사이트
<ul>
<li><a href="https://regexr.com/" target="_blank">https://regexr.com/</a>
<li><a href="https://regex101.com/" target="_blank">https://regex101.com/</a></li>
</ul>
<br/><br/><br/><br/>


# 문법
|특수문자|설명|
|---|----|
|[]| 대괄호 안 모든 문자와 일치합니다. 범위는 하이픈으로 지정됩니다[a-z], [0-9]) | 	ㅁㄴㅇ |
| [^] | 대괄호 사이에 표시되지 않은 어떤 문자와도 일치합니다. 범위는 하이픈으로 지정됩니다([a-z], [0-9]) | ㅁㄴㅇㄹ |
| . (마침표)|	모든 단일 문자와 일치합니다.|ㅁㄴㅇ|
| + (더하기 부호)|	더하기 부호 바로 앞에 하나 이상의 문자를 포함하는 문자열과 일치합니다.|ㅁㄴㅇ|
| * (별표)|	별표 앞에 0개 이상의 문자를 포함하는 문자열과 일치합니다. 검색에서는 별표 앞 문자를 선택사항으로 처리합니다.	|ㅁㄴㅇ|
| ? |	물음표 바로 앞에 0개 이상의 문자를 포함하는 문자열과 일치합니다.물음표 앞 문자는 검색 시 선택사항으로 처리됩니다.| ㅁㄴㅇㄹ |
| ㅣ(파이프) | 파이프의 각 측에 있는 문자와 일치합니다.|ㅁㄴㅇ|
| \	| 특수 문자를 보통 문자로 변환하는 이스케이프 문자입니다.	|ㅁㄴㅇ|
| (?i) | 대소문자 구분을 제어하는 데 사용됩니다. 이 인라인 수정자는 오른쪽에 있고 동일한 엔클로징 그룹에 있는 모든 문자에 영향을 미칩니다. 영향을 받는 문자는 대소문자를 구분하지 않을 수 있습니다.| ㅁㄴㅇ |

일단 예제를 보시죠.!!



- 기호 : []
- 문자들중 하나이상의 문자와 일치합니다.문자 클레스입니다.<br/>
대괄호 안 문자들의 or 가능한 문자들의 집합을 정의.<br/>
대괄호 안에는 어떤 것이든 들어갈 수 있다. 단, 엄밀하게 구분된다. <br/>즉, a와 A가 다르고, z와 Z가 다르다. 범위를 나타내려면 두 문자 사이에 - 기호를 넣습니다.<br/>
```
ex) c[abc]t : cat, cbt, cct
ex) ct[abc] : cta, ctb, ctc
ex) ct[a-c] : cta, ctb, ctc
ex) ct[^abc] : ctd, cte, ct[, ct], ct^, ct\ ,...
ex) ct[^a-c] : ctd, cte, ct[, ct], ct^, ct\ ,...
```




- 기호 : \
- 자주 사용되는 문자 클레스를 \알파벳 으로 축약해 사용할 수 있습니다.<br/>
보통 알파벳을 표현하기 위해서 [a-zA-Z]를 사용하고, 숫자를 표현하기 위해 [0-9]를 쓴다.<br/>
이게 너무 귀찮다보니 한 글자로 줄여버렸다.<br/>

```
\d : [0-9] 
ex) ct\d : ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8, ct9 
ex) \d\d\d : 000, 111, 123, ... 
ex) \d\d\d-\d\d\d-\d\d\d\d : 010-1234-2222, 053-1324-2556


\D : \d와 반대 (대문자는 반대의 의미) 
ex) ct\D : cta, ctb, ctc, ctd, cte, ct[, ct], ct^, ...


\w : [a-zA-Z0-9_] 워드 문자 
ex) \w\w\w : xyz, ABC, ...
ex) \w+ : Mozilla/5.0 (Linux; Android 6.0.1; SM-T375L Build/MMB29K; wv) -> Mozilla, 5, 0, Linux, Android, 6, 0, 1, SM, T375L, Build, MMB29K, wv


\W : [^a-zA-Z0-9_] \w와 반대 
ex) \W+ : Mozilla/5.0 (Linux; Android 6.0.1; SM-T375L Build/MMB29K; wv) -> /, ., (, ; ,' ' ,' ' ,. ,. ,; ,' ' ,- ,' ' ,/ ,; ,' ' ,)

[a-zA-Z0-9] \w  문자+숫자인 것을 찾는다. (특수문자는 제외. 단, 언더스코어 포함)  텍스트 + 숫자
[^a-zA-Z0-9]    \W  문자+숫자가 아닌 것을 찾는다.   특수문자 + 공백

\b : \s 경계 문자
ex) \bection : ' ection' -> 'ection'
\B : \S \b와 반대
```
<br/><br/><br/>

--------------------------------
제가 너무 어렵게 설명하고 있네요.<br/>
솔직히 설명 계속 해봐야 이해 안될것 같습니다.<br/>
지금 저도 제대로 이해하고 있지 않다고 생각됩니다.<br/>
더 공부한 후에 다시 업데이트 하겠습니다.<br/>
- 2022년 02월 25일(마지막 작성)


