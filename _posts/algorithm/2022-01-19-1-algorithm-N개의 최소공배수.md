---
layout: single
title:  "[programmers] N개의 최소공배수"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
<p>두 수의 최소공배수(Least Common Multiple)란 입력된 두 수의 배수 중 공통이 되는 가장 작은 숫자를 의미합니다. 예를 들어 2와 7의 최소공배수는 14가 됩니다. 정의를 확장해서, n개의 수의 최소공배수는 n 개의 수들의 배수 중 공통이 되는 가장 작은 숫자가 됩니다. n개의 숫자를 담은 배열 arr이 입력되었을 때 이 수들의 최소공배수를 반환하는 함수, solution을 완성해 주세요.</p>

# 제한사항
<p>arr은 길이 1이상, 15이하인 배열입니다.
arr의 원소는 100 이하인 자연수입니다.</p>

# Idea
<p>이 문제를 보고서 Euclidean GCD가 떠올랐다.</p>

```python
def hcfnaive(a,b):
    if(b==0):
        return a
    else:
        return hcfnaive(b,a%b)
```
<p>보통은 저렇게 코딩하지만 파이썬답게 수정했다.(사실 다를건 없다.ㅋㅋ)<br/>
또다른 특징은, 리스트내의 자연수들의 곱은 리스트내의 자연수들의 LCM*GCD이다.<br/>
그리고 나서 동영상을 하나 보고, 해결했다.<br/>
<a href="https://ko.khanacademy.org/math/cc-sixth-grade-math/cc-6th-factors-and-multiples/cc-6th-lcm/v/least-common-multiple-lcm" target="_blank">참고동영상</a>
</p>


# Code
```python
#Euclidean Algorithm
def findGCD(x, y):
    while(y):
        x, y = y, x % y
    return x

def findLCM(arr):
    lcm = arr[0]
    for i in range(1, len(arr)):
        lcm = lcm*arr[i] // findGCD(lcm, arr[i])
    return lcm

def solution(arr):
    return findLCM(arr)
```

# Explain
GCD는    
``` "2개의 자연수 a,b(a>b)에 대해서 a를 b로 나눈 나머지가 r일때, a와b의 최대공약수는 b와 r의 최대공약수와 같다.```      
0일 나올때까지 반복하게 되면 최대공약수는 구해지게 된다.     
다른 특징 하나인 ```리스트내의 모든 원소의 곱``` == ```GCD * LCM``` 을 이용해서,   
LCM을 구하기 위해, 양변에 <sup>1</sup>&frasl;<sub>GCD</sub>을 곱하면, 반복문안의 식을 구할수있다.    
끝!    