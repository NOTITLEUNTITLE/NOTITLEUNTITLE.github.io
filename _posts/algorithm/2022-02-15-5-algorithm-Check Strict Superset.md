---
layout: single
title:  "[Hackerrank] Check Strict Superset"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
You are given a set A and n other sets.
Your job is to find whether set A is a strict superset of each of the N sets.

Print True, if A is a strict superset of each of the N sets. Otherwise, print False.

A strict superset has at least one element that does not exist in its subset.


<br/><br/>


<br/><br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line contains the space separated elements of set A.<br/>
The second line contains integer n, the number of other sets.<br/>
The next n lines contains the space separated elements of the other sets.<br/>
<h2>Output Format</h2>
Print True if set A is a strict superset of all other N sets. Otherwise, print False.
<h2>Others</h2>

- 0 $\lt$ len(set(A)) $\lt$ 501
- 0 $\lt$ N $\lt$ 21
- 0 $\lt$ len(othersets) $\lt$ 101



<br/><br/><br/>



# 입출력 예
Sample Input 0
```
1 2 3 4 5 6 7 8 9 10 11 12 23 45 84 78
2
1 2 3 4 5
100 11 12
```
Sample Output 0
```
False
```

# Idea
<p>
issubset() method로 풀어도 됬겠는데.. 차이점도 지난문제와 차이점도 알수있고.
</p>
<br/><br/><br/>

# Code
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == "__main__":
    result = True
    A = list(map(str, input().split()))
    for _ in range(int(input())):
        escape = False
        other_set = list(map(str, input().split()))
        for i in range(len(other_set)):
            if other_set[i] not in A:
                result = False
                escape = True
                break
        if escape:
            break
    print(result)
```

# Explain
쉬운문제들도 다 포스팅 하려니깐 시간이 많이 걸리네요 ..<br/>
그래도 유용한 method들을 알수있어서 좋습니다.<br/>
그리고 문제설명에는 길이를 고려해야하는걸로 나오는데,<br/>
저는 길이를고려하지 않아도 패스가 됬네요 ;;ㅋㅋ<br/>





<br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
arr = input().split()
n = int(input())
result = True
for _ in range(n):
    if len(set(arr).union(set(input().split())))!=len(set(arr)) and result == True:
        result = False
print(result)
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/>

다른사람의 풀이 #2
<hr align="left" style="border: solid 10px gray;">

```python
A = set(map(int,input().split()))
N = int(input())
print(all([A.issuperset(set(map(int,input().split()))) for _ in range(N)]))
```
<hr align="left" style="border: solid 10px gray;">
superset() method도 존재하네요 ㅋㅋㅋ<br/><br/><br/>





<!-- # References
<ul>
  <li><a href="https://docs.python.org/3.6/library/stdtypes.html?highlight=issubset#frozenset.issubset" target="_blank">https://docs.python.org/3.6/library/stdtypes.html?highlight=issubset#frozenset.issubset</a></li>
  <li><a href="https://docs.python.org/3.6/library/stdtypes.html?highlight=difference#frozenset.difference" target="_blank">https://docs.python.org/3.6/library/stdtypes.html?highlight=difference#frozenset.difference</a></li>
  
</ul> -->

