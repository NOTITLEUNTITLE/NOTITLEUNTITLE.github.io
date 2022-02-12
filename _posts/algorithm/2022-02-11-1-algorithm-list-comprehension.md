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
평상시에 굉장히 많이보던 코드유형이였는데, 이러한 코드들을 comprehension이라고 한다네요 ㅎㅎ.<br/>
다양한 예제들도 접해보고, 좋았습니다.!!!
참고로 위의 코드를 파이썬스럽지 않게(?)짜면 아래와 같습니다!!



<hr align="left" style="border: solid 10px gray;">

```python
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    result = []
    for i in range(x+1):
      for j in range(y+1):
        for k in range(z+1):
          if i+j+k != n:
            result.append([i,j,k])
    print(result)
            
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/><br/><br/><br/>


<!-- # References
<ul>
  <li><a href="https://www.geeksforgeeks.org/matrix-exponentiation/" target="_blank">https://www.geeksforgeeks.org/matrix-exponentiation/</a></li>
  <li><a href="https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/?ref=lbp" target="_blank">https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/?ref=lbp</a></li>
  <li><a href="https://myjamong.tistory.com/305" target="_blank">https://myjamong.tistory.com/305</a></li>
  
</ul>  
<br/> -->
