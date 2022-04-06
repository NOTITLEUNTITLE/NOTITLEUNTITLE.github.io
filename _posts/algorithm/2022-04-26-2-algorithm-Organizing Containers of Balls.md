---
layout: single
title:  "[Hackerrank] Apple and Orange"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
David has several containers, each with a number of balls in it. He has just enough containers to sort each type of ball he has into its own container. David wants to sort the balls using his sort method.

David wants to perform some number of swap operations such that:

- Each container contains only balls of the same type.
- No two balls of the same type are located in different containers.

<br/><br/><br/>



# 제한사항

## Input Format
The first line contains an integer q, the number of queries.

Each of the next q sets of lines is as follows:

- The first line contains an integer n, the number of containers (rows) and ball types (columns).
- Each of the next n lines contains n space-separated integers describing row containers[i].


## Output Format

- string: either Possible or Impossible



## Others
- 1 &le; q &le; 10
- 1 &le; n &le; 100
- 0 &le; containers[i][j] &le; 10<sup>9</sup>


<br/><br/><br/>


# 입출력 예
Sample Input
```
2
2
1 1
1 1
2
0 2
1 1
```
Sample Output
```
Possible
Impossible
```

# Idea

<p>
컨테이너별로 공의 갯수를 세고나서, 타입별로 공의 갯수를 센다.<br/>
만약 갯수가 같다면, 타입별로 공을 제거해준다.<br/>
타입별로 공이 남아있지 않다면, 타입별로 공을 컨테이너에 위치시킬수 있으므로, possible을 출력, 그렇지 않으면 impossible을 출력<br/>


</p>
<br/><br/><br/>

# Code

```python
def organizingContainers(container):
    n = len(container[0])
    rows=[0]*n
    cols=[0]*n
    for i in range(n):
        for j in range(n):
            rows[i] +=container[i][j]
            cols[i] +=container[j][i]
 
    if sorted(rows)==sorted(cols):
            return "Possible"
    else:
            return "Impossible"
```


# Explain
코드를 보면 컨테이너별로 타입별로 주어진 값이 입력된 matrix가 주어지는데,<br/>
matrix에서 rows가 container별 공의 갯수이고, cols가 type별 공의 갯수이다.<br/>
누적시켜서 갯수를 다 구한다음에 갯수가 같으면 possible, 그러니깐 바꿀수가있는것이고,<br/>
갯수가 다르면 바꿔도 컨테이너별로 타입을 모을수가 없다.<br/>
끝<br/>


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->


