---
layout: single
title:  "[Hackerrank] list comprehension"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Let's learn about list comprehensions! You are given three integers x,y and z representing the dimensions of a cuboid along with an integer n.

Print a list of all possible coordinates given by (i, j, k) on a 3D grid where the sum of i+j+k is not equal to n.

 Here, 0 <= i <= x; 0 <= j <= y; 0 <= k <= z . Please use list comprehensions rather than multiple loops, as a learning exercise.

<br/><br/>

# 입출력 예
Sample input 0
```
1
1
1
2
```
Sample output 0
```
[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
```

# Idea
<p>
python의 list comprehension을 적용해서 문제를 풀라고 합니다.
이 문제 덕분에 comprehension에 대해서 좀더 공부하게 되었네요 ㅎㅎ

</p>
<br/><br/><br/>

# Code
```python
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print(list([i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1)  if i+j+k !=n))
```

# Explain
코드 설명을 간단히 하자면, 주석처리된 출력문을 실행해보면,<br/>
연속된 자연수의 합으로 n을 나타낼수있는만큼 "Success"가 출력되게 된다.<br/><br/>이중 포문을 써서 1~15까지, 2~15까지.... 마지막은 15가 되면 포문을 빠져나오고, 15까지 조건에 부합하니 한번더 증가시켜준다.<br/>
정상적으로 실행은 되지만, 효율성 테스트에서 실패했다.. 그리고 곰곰히 생각해 봤다.<br/> 최종코드는 아래와 같다!!<br/><br/><br/>
<hr align="left" style="border: solid 10px gray;">

```python
def solution(n):
    answer = 1
    for i in range(1,n//2 + 1):
        temp = i
        for j in range(i+1, n+1):
            if temp > n:
                break
            if temp == n:
                answer += 1
                break
            temp += j
    return answer
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/><br/><br/><br/>

다른 풀이 #1<br/>
```python
def solution(n):
    return len([i  for i in range(1,n+1,2) if n % i is 0])
```

설명은 아래 링크를 클릭해주세요!!<br/><br/>
<a href="https://gkalstn000.github.io/2021/01/21/%EC%88%AB%EC%9E%90%EC%9D%98-%ED%91%9C%ED%98%84/" target="_blank">https://gkalstn000.github.io/2021/01/21/%EC%88%AB%EC%9E%90%EC%9D%98-%ED%91%9C%ED%98%84/</a>

<!-- # References
<ul>
  <li><a href="https://www.geeksforgeeks.org/matrix-exponentiation/" target="_blank">https://www.geeksforgeeks.org/matrix-exponentiation/</a></li>
  <li><a href="https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/?ref=lbp" target="_blank">https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/?ref=lbp</a></li>
  <li><a href="https://myjamong.tistory.com/305" target="_blank">https://myjamong.tistory.com/305</a></li>
  
</ul>  
<br/> -->
