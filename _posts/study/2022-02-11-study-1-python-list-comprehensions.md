---
layout: single
title:  "[study] Python list comprehensions"
categories: study
tag: [python]
toc: true
author_profile: false
---

# &nbsp;&nbsp;&nbsp;&nbsp;Ah-Jji?
처음 파이썬을 공부하기 시작했을때, set(), dictionary(), list(), tuple() 이 4개의 개념이 헷갈렸다.<br/>
(위의 4가지의 개념도 포스팅을 작성하면 좋을것 같다.!!)<br/>
list를 접하면서 comprehension이라는 개념을 접했는데, 
블로그를 포스팅을 하기전이라서 따로 포스팅을 안했지만, 이런 개념들을 포스팅 하는것도 괜찮겠다 싶어, 하나씩 포스팅해보겠습니다.<br/>



<br/><br/>

# &nbsp;&nbsp;&nbsp;&nbsp;필요한 개념 및 용어정리.
list는 array라고 생각해보자.<br/>
그러면, comprehension은 무엇인가?<br/>
<img src="../../images/2022-02-11/study-1.png">
<br/>이해력이라는 뜻이 있습니다.<br/>
여기서부터 헷갈리기 시작합니다..<br/>
배열의 이해(?)라는 괴상한 단어뜻으로 유추가 가능해집니다....<br/>
어떻게 보면 맞는말이긴 하지만, 정확한 표현법은 아닐것같습니다.<br/>
list comprehension은 그냥 리스트 컴프리핸션입니다!!!!!(콩글리쉬 ㅋㅋ)<br/>
저희가 영어자료들을 공부할때(python과 같은 대부분의 자료들!!!), 한국어로 번역된 번역본을 가지고 공부를 하면, 나중에 확 꼬이게 되는 수많은 경우를 경험해봤습니다...<br/>
컴퓨터 분야(특히, 인공지능 논문.)는 다 영어입니다.<br/>
더이상 회피할수없습니다.<br/>
<br/>
영어의 중요성을 어필하는게 본 포스팅의 목적은 아니므로, 여기까지만 하겠습니다.<br/>
<br/>
결론은 list comprehension은 그냥 list comprehension으로 사용하겠습니다!!
<!-- <h2 style="text-align:center">앞으로 작성될 article은 영어원문을 그대로 쓰는경우가 많겠습니다</h2> -->



<br/><br/><br/>

# 예제
보통 파이썬으로 배열을 선언하고 배열안에 값을 넣는경우의 코드를 살펴봅시다!<br/>
```python
test_list = list()
for i in range(10):
  test_list.append(i)
```
위의 코드를 해석하면, <br/>
0~9번 인덱스까지 0~9가 입력되는것을 확인할수있습니다.<br/>
위의 코드를 list comprehension하게 짜본다면,
```python
test_list = [ i for i in range(10)]
```
굉장히 파이썬스럽게 바뀌어졌네요.ㅎㅎ<br/>
코드를 뜯어보면,<br/>
```
test_list = [ (변수의 값) for (변수이름) in (iterator)]
```
제일 중요한부분은 iterator입니다.
<ul>
<li>변수의 값은 선언한 list([])안에 어떤 값을 넣은건지에 대한 부분입니다.</li>
<li>변수이름은 "알맞게" 아무렇게나 작성해주시면 됩니다.</li>
<li>iterator는 이전 코드에서 작성한 range(10)을 대체하는 부분인데요.<br/>
제가 예시로 설명해드린 2개의 파이썬코드 둘다 range()를 사용했습니다.<br/>
그 말은 range()는 iterator하다라고 볼수있습니다!!<br/>
iterator는 순회가 가능한 객체를 말하는데요.<br/>
여기에는 range()만 해당하는것이아니라, set, dictionary 등 순회가 가능한 객체들은 전부 다 가능합니다.!!</li>
</ul>
일반적으로 이렇게 간단한 list comprehension보다는 조건문을 추가한 list comprehension을 더 많이 사용합니다.

```python
test_list = [ i for i in test_list if i % 2 == 0]
```
iterator 객체로 위에서 정의한 1~9까지의 리스트를 사용했습니다.</li>
마저 해석해보면, 2로 나누었을때 나머지가 0인 ,즉 짝수들은 다시 filtering해서 넣어준다는것이다.<br/>
대부분 이런식으로 사용합니다.<br/>

Comprehension 자체가  효율성과 생산성을 따지는 python스러운 코드를 만들어주는데, 크게 기여하는것같다!


# References
<ul>
  <li><a href="https://docs.python.org/3/tutorial/datastructures.html" target="_blank">https://docs.python.org/3/tutorial/datastructures.html</a></li>
</ul>  
<br/>


