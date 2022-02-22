---
layout: single
title:  "[study] Linear Algebra - 0"
categories: study
tag: [linear algebra]
toc: true
author_profile: false
---

# Linear Algebra를 시작하며
대학교 전공이 정보통신이여서, Linear Algebra 수업을 들었었다.
그때 당시에는 별로 흥미가 없어서 공부하지 않았지만, 이런 내가 데이터(인공지능)를 다루면서 Linear Algebra를 다시 복습하고 있다.
(복습보다는 처음 접한다는게 맞는 표현같다...ㅋ)
<br/><br/>

개인적으로 <strong>네트워크,</strong><strong>시스템,</strong><strong>리눅스</strong>쪽을 공부하려고 하고있고, 포스팅도 계획하고 있다.

그전에 Probability, Statistics and LInear Algebra로 포스팅을 시작하고, 글 작성능력이 어느정도 올라가면,
메인 주제인 네트워크, 시스템, 리눅스을 공부 및 포스팅을 하려고 합니다.
<del>(아마 의식의 흐름대로 진행할것같습니다.)</del>
<br/><br/>
## Reference
<ul>
<li><a href="https://www.youtube.com/watch?v=3wcELGwS4rU&list=PL9k2wIz8VsfOYIp4fxbbwgXVeql2vv7uw" target="_blank">Linear Algebra Lecture</a>
</li>
</ul>
<br/><br/>

## OT(기본 개념?)

Algebra 는 대수학이라는 입니다.
대수학은 equation(방정식)문제를 푸는것입니다.<br/>
Linear Algebra는 linear equation을 푸는 학문으로 설명할 수 있습니다.

equation은 y = f(x)에서 y를 주고,  x를 맞추는 유형의 문제들입니다.

equation은 중고등학교에서도 많이 접해왔고, 문제를 풀어봤는데, 대학에서도 이러한 문제를 다룬다고 하면, 왜 배우는 거지라고 의문을 갖습니다.

그래서 보통 linear algebra를 공부한다고 하면 vector와 matrix를 공부한다고 말하면, `아 그렇구나` 하게 되는 아이러니한 상황...ㅎㅎ


### 증명의 방법

#### 방정식의 증명방법
(x-1)<sup>2</sup>+1  &nbsp;=&nbsp;  x<sup>2</sup>-2x+2<br/>
시험삼아 위의 문제를 증명해보세요.<br/><br/><br/>
pf 1)<br/>
(x-1)<sup>2</sup>+1  &nbsp;=&nbsp;  x<sup>2</sup>-2x+2<br/>
x<sup>2</sup>-2x+2  &nbsp;=&nbsp;  x<sup>2</sup>-2x+2<br/>
0 = 0<sub>QED</sub>


pf 2)<br/>
(x-1)<sup>2</sup>+1  &nbsp;=&nbsp;  x<sup>2</sup>-2x+2<br/>
(x-1)<sup>2</sup>+1  - x<sup>2</sup>-2x+2 = 0<sub>QED</sub>

pf2가 올바른 방법이며, pf1은 증명이라고 할 수가 없습니다!!!<br/>
이유는, proof1은 2개의 식이 같다는것을 증명하라는 문제에서 이미 <strong>같다는 전제하에 증명을 시작하고 있습니다.</strong><br/>
따라서 할 말이 없는거죠..
주장은 하지만, 근거가 없는 경우라고 할 수 있습니다.
<br/><br/>
proof2의 경우에는 2개의 식이 같다라고 전제하지 않고 있습니다.<br/>
그렇기 때문에 빼보는 시도를 할 수 있었습니다.
그러한 시도를 했기 때문에, 같다는 결론을 도출할 수 있었으며, 
주장과 근거가 잘 성립된다고 볼 수 있습니다.
<br/><br/><br/>


증명을 하는 방법들은 다양하게 있을 수 있습니다.
- 복잡한 식을 간단하게 만드는 방법.
- 부등식을 이용한 방법.
- 기타 등등

본인에게 잘 맞는 방법, 문제에 따라서 선택하시면 됩니다.(물론 공부는 해야합니다!!)<br/><br/><br/>
#### 집합의 증명방법
이제는 집합의 증명에 대해서도 알아보도록 하시죠!<br/>
A = B (A,B = 집합) 일 때,<br/>
A - B =... = ... = ... =  &empty; 이러시면 안됩니다!!)<br/>
A가 {1,2,3}, B가 {1,2,3,4}인 경우 모순이 발생합니다.<br/>
<br/>
보통 집합이 같다라는 증명을 하려면,<br/>
A &sub; B를 한번 증명해주고,<br/>
B &sub; A도 한번 증명해주면, 끝입니다!!<br/>
<br/><br/>
A &sub; B,<br/>
x &isin; A = ... = ... = ... = x &isin; B<br/>
위와 같은 방법으로 접근해야지만 증명할 수 있습니다.<br/>
집합의 포함관계를 증명하라는 문제는,<br/>
먼저 A에서 임의의 한  원소를 가져와보고, 쭉쭉쭉 써내려가면 A에서 가져온 원소가 B에 포함된다.<br/>
(드 모르간의 법칙을 공부해 보시는것도 좋습니다.)

<br/><br/><br/>

#### p -> q 증명방법
p는 가정, q는 결론
- ~q -> ~p이다. (대우명제)
-  p, ~q


p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub>, p<sub>4</sub> -> q 를 증명하시오<br/>
가정이 여러개이고, 결론은 1개인 경우입니다.<br/>
위의 증명 문제를 대우명제로 만약 접근하게 되면,<br/>
~q -> ~p<sub>1</sub> <sub>or</sub> ~p<sub>2</sub> <sub>or</sub> ~p<sub>3</sub>   <sub>or</sub> ~p<sub>4</sub> 
어질어질합니다.<br/><br/>
그래서 조금 더 쉽게 하는 방법을 알려드리겠습니다.<br/>
p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub>, p<sub>4</sub>,  ~q
위와 같이 결론을 부정한 후, 한꺼번에 생각하는 것입니다.<br/>
그렇게 증명을 이어나가다 보면 0=5와 같은 이상한 식이 나오게 되면, 그 곳에서 증명 끝을 외치면 끝납니다.<br/><br/><br/>



### 단사함수, 전사함수, 전단사함수
- injective function, one-to-one function : 단사함수
- surjective function, onto function : 전사함수
- bijection, one-to-one correspondence : 전단사함수
<br/><br/>

one-to-one function 은 일대일 함수이고,<br/>
one-to-one correspondence은 일대일 대응입니다.!!<br/>

일대일대응이면 일대일 함수인데,<br/>
일대일함수라고 해서 일대일대응이다라고는 할 수 없습니다!!<br/><br/>

단사함수 : x &ne; y &rArr; f(x) &ne; f(y)<br/>
단사함수 증명의 형태:<br/>
pf) suppose <br/><br/>
f(x) = f(y) <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...<br/>
&nbsp;&nbsp;&nbsp;  x= y<br/><br/>
 
전사함수 : 






