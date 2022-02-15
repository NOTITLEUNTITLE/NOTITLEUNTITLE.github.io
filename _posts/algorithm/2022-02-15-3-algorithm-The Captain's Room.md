---
layout: single
title:  "[Hackerrank] The Captain's Room"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Mr. Anant Asankhya is the manager at the INFINITE hotel. The hotel has an infinite amount of rooms.<br/>
<br/>
One fine day, a finite number of tourists come to stay at the hotel.
The tourists consist of:<br/>
→ A Captain.<br/>
→ An unknown group of families consisting of K members per group where K ≠ 1.<br/>
<br/>
The Captain was given a separate room, and the rest were given one room per group.<br/>
<br/>
Mr. Anant has an unordered list of randomly arranged room entries. The list consists of the room numbers for all of the tourists. The room numbers will appear K times per group except for the Captain's room.<br/>
<br/>
Mr. Anant needs you to help him find the Captain's room number.
The total number of tourists or the total number of groups of families is not known to you.<br/>
You only know the value of K and the room number list.<br/>

매니저가 호텔을 관리하는데 호텔에 방의 개수가 무한하다고 합니다.<br/>
무한명의 숙박객이 호텔에 옵니다.<br/>
캡틴은 한명이고, k=1이 아닌, k명으로 구성된 그룹들이다.<br/>
캡틴은 독방이고, 그룹당 한개의 방이 주어진다.<br/>
k의 값과 방의번호를 캡틴의 방을 찾아라...<br/>







<br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line consists of an integer, K, the size of each group.
The second line contains the unordered elements of the room number list.<br/>

<h2>Output Format</h2>
Output the Captain's room number.
<h2>Others</h2>

- 1 $\lt$ K $\lt$ 1000


<br/><br/><br/>



# 입출력 예
Sample Input 0
```
5
1 2 3 6 5 4 4 2 5 3 6 1 6 5 3 2 4 1 2 5 1 4 3 6 8 4 3 1 5 6 2 
```
Sample Output 0
```
8
```

# Idea
<p>
collections의 Counter class를 사용했습니다.
</p>
<br/><br/><br/>

# Code
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
if __name__ == "__main__":
    N = int(input())
    num_list = list(map(str, input().split()))
    result = Counter(num_list)
    for key, value in result.items():
        if value == 1:
            print(key)
```

# Explain
일단 첫번째 입력인 k를 받는데, 모든 구성원들이 k개의 그룹으로 나뉘게 됩니다(캡틴 제외).<br/>그리고 k=1이 안된다는 조건이 있습니다.<br/> 캡틴만 독방을 씁니다.<br/>최종적으로 방번호가 있고, 각 방번호마다 k개의 사람들이 들어가 있을텐데, 이중에서 1명이 방번호 한개를 사용(독방)하는 경우가 캡틴의 방입니다.!!!<br/>


<br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
from collections import Counter
k = int(input())
room_list = Counter(input().split())
print(*[k for k,v in room_list.items() if v == 1])

```
<hr align="left" style="border: solid 10px gray;">
<br/><br/>




# References
<ul>
  <li><a href="https://docs.python.org/3.6/library/collections.html?highlight=counter#collections.Counter" target="_blank">https://docs.python.org/3.6/library/collections.html?highlight=counter#collections.Counter</a></li>
</ul>

