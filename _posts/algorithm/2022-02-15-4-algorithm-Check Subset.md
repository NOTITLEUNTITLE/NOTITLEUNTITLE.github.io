---
layout: single
title:  "[Hackerrank] Check Subset"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
You are given two sets, A and B.<br/>
Your job is to find whether set A is a subset of set B.<br/><br/>

If set A is subset of set B, print True.<br/>
If set A is not a subset of set B, print False.<br/>


<br/><br/>
set과 관련된 문제이다.<br/>
A가 B의 subset이면 True<br/>
A가 B의 subset이 아니면 False<br/>

<br/><br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line will contain the number of test cases, T.<br/>
The first line of each test case contains the number of elements in set A.<br/>
The second line of each test case contains the space separated elements of set A.<br/>
The third line of each test case contains the number of elements in set B.<br/>
The fourth line of each test case contains the space separated elements of set B.<br/>

<h2>Output Format</h2>
Output True or False for each test case on separate lines.
<h2>Others</h2>

- 0 $\lt$ T $\lt$ 21
- 0 $\lt$ Number of elements in each set $\lt$ 1001


<br/><br/><br/>



# 입출력 예
Sample Input 0
```
3
5
1 2 3 5 6
9
9 8 5 6 3 2 1 4 7
1
2
5
3 6 5 4 1
7
1 2 3 5 6 8 9
3
9 8 2
```
Sample Output 0
```
True 
False
False
```

# Idea
<p>
음...구현의 문제???
</p>
<br/><br/><br/>

# Code
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT

if __name__ == "__main__":
    for i in range(int(input())):
        len_A = int(input())
        ele_A = list(map(int, input().split()))
        len_B = int(input())
        ele_B = list(map(int, input().split()))
        ele_A.sort()
        ele_B.sort()
        count = 0
        for i in range(len_A):
            if ele_A[i] in ele_B:
                count += 1
        if count == len_A:
            print("True")
        else:
            print("False")
```

# Explain
처음에는 리스트 A가 B안에 있는지 확인해보려했는데, 리스트를 정렬한다고 하더라도, 반례들이 존재해서, 일일이 카운트하는 방법을 택했습니다.<br/>
다른 코드를 보니 부분집합이 되는 조건중에 하나인 리스트 길이에 대한 조건문을 선행하는게 있어서 견문을 넓히기위해 가져와봤습니다.<br/>
제가 맨처음에 풀려고 했던 방법은 <strong>다른사람의 풀이 #3</strong>과 동일하네요!!


<br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
for _ in range(int(input())):
    a,A,b,B = [set(map(int,input().split())) for i in range(4)][0::]
    isSubset = True
    if(a > b):
        print('False')
        continue
    
    for i in A:
        if(i not in B):
            print('False')
            isSubset = False
            break
						
    if(isSubset):
        print('True')
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/>

다른사람의 풀이 #2
<hr align="left" style="border: solid 10px gray;">

```python
number_tests = int(input())
for _ in range(number_tests):
    len_a = int(input())
    set_a = set(map(int, input().split()))
    
    len_b = int(input())
    set_b = set(map(int, input().split()))
    
    print(set_a.issubset(set_b))
```
<hr align="left" style="border: solid 10px gray;">
이분은 아예 issubset()이라는 methode를 사용하셨네요;;<br/>
처음 알았습니다. ㅎㅎ<br/><br/><br/>


다른사람의 풀이 #3
<hr align="left" style="border: solid 10px gray;">

```python
n = int(input())
for i in range(0,n):
    t = int(input())
    t1 = set(input().split())
    T = int(input())
    t2 = set(input().split())
    t3 = t1.difference(t2)
    if (len(t3)==0):
        print("True")
    else:
        print("False")
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/><br/>



# References
<ul>
  <li><a href="https://docs.python.org/3.6/library/stdtypes.html?highlight=issubset#frozenset.issubset" target="_blank">https://docs.python.org/3.6/library/stdtypes.html?highlight=issubset#frozenset.issubset</a></li>
  <li><a href="https://docs.python.org/3.6/library/stdtypes.html?highlight=difference#frozenset.difference" target="_blank">https://docs.python.org/3.6/library/stdtypes.html?highlight=difference#frozenset.difference</a></li>
</ul>

