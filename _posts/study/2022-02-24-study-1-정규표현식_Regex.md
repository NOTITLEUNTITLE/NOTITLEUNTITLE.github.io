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
이번 기회에 정리를 해야겠습니다.!!!<br/>
사실 SQL을 포함하여 Frontend, Backend, Data 작업을 하다보면 종종 쓰입니다!!
<br/><br/><br/><br/>



# Regex란..
Regular Expression이란 Text에서 원하는 문자열(패턴)을 쉽게 찾아낼수 있는 문법입니다.<br/>
보통은 형식이 있는 문자열들을 구별해 내고자 할때 주로 사용합니다.<br/>
예를들면, 이메일(@와.이 포함),전화번호(3자리-4자리-4자리),웹사이트주소(www.abcd.ab) 등등<br/>
단순히 찾는것에만 끝나는것이 아니라, 변환도 가능합니다.<br/><br/>

개인적으로는 CTF대회를 나가면서 Shell Script를 작성하거나, Linux를 사용하면서 Regex라는것이 필요해서 사용해본 경험이 있습니다.<br/>
(python으로 Web-site 제작할때도 사용했습니다.)<br/>
<br/><br/>

<hr/>
다른 주제의 이야기이지만, 네트워크, 리눅스, 시스템, 데이터 등 해야할 공부가 많아 Regex를 포스팅하는데에 시간을 할애하는것이 맞는 것인지 고민해보았지만,<br/>

`어차피 리눅스 쓰면 쓰게 될텐데` 라는 생각으로 글을 작성하고 있습니다..ㅋㅋ
<hr/>
<br/><br/>


# 개념
## 형태
`/Regex/ i` <br/>
위와 같은 Regular Expression 형태가 있을때, 
- `/` 기호의 의미는 `/` 사이에 있는 문자들은 Regular Expression이라는 의미입니다.
- `Regex` 의 의미는 Regular Expression이 되겠습니다.<br/>
저희가 찾고자 하는 패턴을 규칙에 맞춰서 입력해 주시면 됩니다.
- `i` 는 옵션을 의미합니다.
  - i : 대소문자 구분를 하지 않습니다.
  - g : 패턴과 일치하는 모든 경우를 검색합니다.<br/>
  (주의사항_미입력시 패턴과 일치하는 첫 번째 결과만 반환됩니다.)<br/>
  - m : 전체 문자열의 처음과 끝뿐 아니라 각 행의 시작과 끝에도 대응합니다.
  (주의사항 `^`와 `$`에 대해서만 작동을 합니다!)<br/>
  - flag에 대한 자세한 설명이 필요하신분들은 아래 링크를 참고해주세요.<br/>
  <a href="https://ko.javascript.info/regexp-introduction" target="_blank">https://ko.javascript.info/regexp-introduction</a>

<br/><br/><br/>

## 문법
문법은 제가 알려드린 사이트에서 직접 해보시면서 이해하시기를 추천드립니다.<br/>(사이트는 아래에 링크되어 있습니다. 첫번째 사이트를 추천드립니다!!)<br/><br/>
첫번째로는 패턴에 문자열을 그대로 넣어서 검색해보겠습니다.
```
Regex : /Hi/gmi

String : Hi there, Nice to meet you. And Hello there and hi.
```
총 2개를 찾아낼수 있는데요.<br/>
첫번째는 pattern과 일치한 Hi, 그리고 제일 마지막에 있는 hi가 되겠습니다.<br/>제일 마지막에 있는 hi를 찾아낸 이유는 flag `i` 때문입니다.<br/>
<br/>

두번째로는 그룹의 개념입니다.
```
Regex : /N(i|a)ce/gmi

String : Hi there, Nice to meet you. And Nace And Ntce.
```
Nice와 Nace가 검색되는것을 확인하실수 있으실텐데요.<br/>
여기서 처음 접하는 개념은 `|`입니다. pipeline이라고 읽으며, OR(또는)의 기능을 한다고 생각하시면 됩니다.<br/>
즉, `N`으로 시작을 하며 `i` 또는 `a`가 두번째 문자이며, `ce`로 끝나는 문자열을 찾으라는 정규표현식이였습니다.<br/>
보통은, `/N[ai]ce/gmi`로 많이들 사용하는데요.<br/>
여기서 []는 []안에 있는 문자열들 중 한개라도 만족한다면 다 찾아주는 기능을 해주고 있습니다.<br/>
<br/>

세번째로 설명드릴것은 








<br/><br/><br/><br/>

# Reference
<ul>
<li><a href="https://www.ibm.com/docs/ko/control-center/6.1.1?topic=reference-regular-expressions" target="_blank">IBM 정규표현식</a>
<li><a href="https://wikidocs.net/80567" target="_blank">토닥토닥 파이썬 정규표현식</a></li>
<li><a href="https://www.youtube.com/watch?v=t3M6toIflyQ" target="_blank">https://www.youtube.com/watch?v=t3M6toIflyQ</a></li>

</ul>
<br/><br/><br/><br/>


# 연습사이트
<ul>
<li><a href="https://regexr.com/" target="_blank">https://regexr.com/</a></li>
<li><a href="https://regex101.com/" target="_blank">https://regex101.com/</a></li>
<li><a href="https://www.hackerrank.com/" target="_blank">https://www.hackerrank.com/</a></li>
</ul>

