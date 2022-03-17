---
layout: single
title:  "[Hackerrank] Queen's Attack II"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
You will be given a square chess board with one queen and a number of obstacles placed on it. Determine how many squares the queen can attack.
<br/><br/>
Complete the queensAttack function in the editor below.

queensAttack has the following parameters:
- int n: the number of rows and columns in the board
- nt k: the number of obstacles on the board
- int r_q: the row number of the queen's position
- int c_q: the column number of the queen's position
- int obstacles[k][2]: each element is an array of 2 integers, the row and column of an obstacle
<br/><br/><br/>

# 제한사항

## Input Format

The first line contains two space-separated integers n and k, the length of the board's sides and the number of obstacles.
The next line contains two space-separated integers r<sub>q</sub> and c<sub>q</sub>, the queen's row and column position.
Each of the next k lines contains two space-separated integers r[i] and c[i], the row and column position of obstacle[i].

## Output Format

int: the number of squares the queen can attack


## Others
- 0 &lt; n &lt;= 10<sup>5</sup>
- 0 &lt; k &lt;= 10<sup>5</sup>
- A single cell may contain more than one obstacle.
- There will never be an obstacle at the position where the queen is located.

<br/><br/><br/>


# 입출력 예

Sample Input 0
```
4 0
4 4
```
Sample Output 0
```
9
```

# Idea

<p>
이 문제를 brute focre 방식으로 접근하게 되면, 시간초과가 발생하게 됩니다.<br/>
(모든 부분집합을 구하고, 부분집합에서 2개의 합을 또 구하고, 비교하는 방식입니다.)<br/>
따라서 나머지 전략을 취하기로 했습니다.<br/>
k개의 만큼 리스트를 만들어 주고, 입력받은 리스트의 각 원소의 값을 k로 나누어서 k개의 리스트에 각각 추가해 줍니다.<br/>

```
k를 9이라고 하자

어떤 수를 k로 나눈 나머지는 0~8이다

그 수들중에 1은 8을 제외하고 더했을 때 9로 나눈 나머지가 0이 되지 않는다.

마찬가지로 2도 7, 3은 6, 4는 5가 더해졌을때를 제외하고는 9로나눈 나머지가 0이 되지 않는다.

그래서 나눈 나머지가 1과8인 것들의 개수중 더 많은 것, (2,7), (3,6) .... 도 마찬가지로 더 많은 것을 골라서 집합에 포함시키면 된다.

0은 예외다. 두 수의 합이기 때문에 0과 다른 모든 수를 더해도 나머지가 0이 아닐 것 같지만

k의 배수 두 개를 더하면 k로 나눈 나머지가 0이기 때문에 k로 나눈 나머지가 0인 수가 있다면 최대 1개만 집합에 포함시킬 수 있다. k가 짝수일 때도 (k/2)인 수 두개 이상을 허용하지 않도록 한다. 
```

</p>
<br/><br/><br/>

# Code

```python
def nonDivisibleSubset(k, s):
    # Write your code here
    count = [0] * k

    for i in s:
        remainder = i % k
        count[remainder] +=1
    
    ans = min( count[0] , 1)          

    if k % 2 == 0:                    
        ans += min(count[k//2] ,1 )

    for i in range( 1 , k//2 + 1):    
        if i != k - i:           
            ans += max(count[i] , count[k-i])
    return ans
```


# Explain
자세한 설명은 Reference를 참고해 주세요!


<br/><br/><br/>





# References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul>

