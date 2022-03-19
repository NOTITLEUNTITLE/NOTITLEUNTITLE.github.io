---
layout: single
title:  "[Hackerrank] Jesse and Cookies"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Jesse loves cookies and wants the sweetness of some cookies to be greater than value k. To do this, two cookies with the least sweetness are repeatedly mixed. This creates a special combined cookie with:

sweetness = (1 x Least sweet cookie + 2nd least sweet cookie).

This occurs until all the cookies have a sweetness &ge;k.

Given the sweetness of a number of cookies, determine the minimum number of operations required. If it is not possible, return -1.


<br/><br/>
Jesse가 달콤한 쿠키를 좋아하는데, 달콤함의 정도가 최소 k보다 크거나 같아야 한다고 합니다.<br/>
그렇게 만들기 위해서, k보다 작은 달콤함을 가진 쿠키 2개를 믹스한다고 하네요.<br/>
만약 쿠키를 전부 다 믹스했는데도 k보다 작다면 -1를 리턴하면 되겠습니다.<br/>



<br/><br/><br/>

# 제한사항

## Input Format
- A의 리스트를 넘겨주고, A는 각 쿠키별 달콤함의 수치를 가지고 있습니다.
- k는 쿠키가 가져야 할 최소한의 달콤함의 수치입니다.

## Output Format

- 만약 모든 쿠키를 다 믹스했는데도 k보다 작으면 -1을 리턴
- 그렇지 않고, 쿠키를 n번동안 믹스 했더니 모든 쿠키가 k보다 크거나 같으면 n번을 리턴



## Others
- 1 &le; n &le; 10<sup>6</sup>
- 0 &le; k &le; 10<sup>9</sup>
- 0 &le; A[i] &le; 10<sup>6</sup>

<br/><br/><br/>


# 입출력 예
Sample Input
```
6 7          
1 2 3 9 10 12
```
Sample Output
```
2
```

# Idea

<p>
그냥 막 풀려고 했는데, 숫자가 백만단위가 있어서, 시간을 고려해야할것 같았습니다.<br/>
저는 정렬을 고려했기에 heap을 사용했습니다.<br/>
코드는 아래와 같습니다.<br/>

</p>
<br/><br/><br/>

# Code

```python
def cookies(k, A):
    # Write your code here
    heapq.heapify(A)
    count = 0
    while A[0] < k:
        if len(A) == 1:
            return -1
        min_num1=heapq.heappop(A)
        min_num2=heapq.heappop(A)
        heapq.heappush(A,min_num1+min_num2*2)
        count += 1
    return count
```


# Explain
heapify() method로 정렬을 한 후, 가장 작은 달콤함의 수치를 확인하고,<br/>그 수치가 k보다 작으면, 커질때까지 반복문을 수행합니다.<br/>
다만, 반복문안에서 리스트의 길이가 1개이면, 믹스를 할 수 없기에, -1를 리턴합니다.<br/>


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->

