---
layout: single
title:  "[Hackerrank] Apple and Orange"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Sam's house has an apple tree and an orange tree that yield an abundance of fruit. Using the information given below, determine the number of apples and oranges that land on Sam's house.<br/>
<br/>
In the diagram below:<br/>
<br/>
- The red region denotes the house, where s is the start point, and t is the endpoint. The apple tree is to the left of the house, and the orange tree is to its right.
- Assume the trees are located on a single point, where the apple tree is at point a, and the orange tree is at point b.
- When a fruit falls from its tree, it lands d units of distance from its tree of origin along the x-axis. *A negative value of d means the fruit fell d units to the tree's left, and a positive value of d means it falls d units to the tree's right. *

<br/><br/><br/>
s와 t의 입력이 주어지면, s와 t사이에 있는 오렌지와 사과의 개수를 리턴해주면 된다.<br/>
ex)<br/>
s=7, t=10이면, 7,8,9,10에 있는 사과와 오렌지의 갯수를 구해서 출력해주면 끝!<br/>




<br/><br/><br/>

# 제한사항

## Input Format
- The first line contains two space-separated integers denoting the respective values of s and t.
- The second line contains two space-separated integers denoting the respective values of a and b.
- The third line contains two space-separated integers denoting the respective values of m and n.
- The fourth line contains m space-separated integers denoting the respective distances that each apple falls from point a.
- The fifth line contains n space-separated integers denoting the respective distances that each orange falls from point b.

## Output Format

Print two integers on two different lines:

- The first integer: the number of apples that fall on Sam's house.
- The second integer: the number of oranges that fall on Sam's house.



## Others
- 1 &le; s,t,a,b,m,n &le; 10<sup>5</sup>
- -10<sup>5</sup> &le; d &le; 10<sup>5</sup>
- a &lt; s &lt; t &lt; b 


<br/><br/><br/>


# 입출력 예
Sample Input
```
7 11
5 15
3 2
-2 2 1
5 -6
```
Sample Output
```
1
1
```

# Idea

<p>
내일부터는 어려운 문제를 하나씩 풀어봐야 겠다.<br/>
문제가 쉬워진다.<br/>


</p>
<br/><br/><br/>

# Code

```python
def countApplesAndOranges(s, t, a, b, apples, oranges):
    # Write your code here
    apple_count = [ apple for apple in apples if a+apple >= s and a+apple <= t]
    orange_count = [ orange for orange in oranges if b+orange >= s and b+orange <= t]
    print(len(apple_count))
    print(len(orange_count))
```


# Explain
apple에는 a를 더해준다.(a가 apple tree)<br/>
orange에는 b를 더해준다.(b가 orange tree)<br/>
그래서 길이를 구하고, 구한 길이가, s와 t사이에 있는 과일들의 갯수를 출력해준다.<br/>
끝<br/>


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->


