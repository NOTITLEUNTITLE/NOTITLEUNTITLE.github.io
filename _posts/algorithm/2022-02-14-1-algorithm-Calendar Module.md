---
layout: single
title:  "[Hackerrank] Calendar Module"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
You are given a date. Your task is to find what the day is on that date.



<br/><br/>

# 제한사항
<h2>Input Format</h2>
A single line of input containing the space separated month, day and year, respectively, in MM DD YYYY format.
<h2>Output Format</h2>
Output the correct day in capital letters.
<h2>Others</h2>

 - 2000 $\lt$ year $\lt$ 3000




# 입출력 예
Sample Input 0
```
08 05 2015
```
Sample Output 0
```
WEDNESDAY
```

# Idea
<p>
알고리즘 문제같지는 않고, 내가 풀고있는게, 파이썬 문법문제여서 그런것 같다.<br/> 음... 계속 해야되나 고민된다.. 알고리즘 문제를 푸는게 맞는것같은데,,<br/>
또 어떻게 보면, 파이썬의 내장함수들에 대해서 아는것도 도움이되니, 잘 모르겠다.<br/>

</p>
<br/><br/><br/>

# Code
```python
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
# print(calendar.TextCalendar(firstweekday=6).formatyear(2022))
query_date = input()
month, day, year = query_date.split()
# print(month, day, year)
month, day, year = int(month), int(day), int(year)
result = {
    0: "MONDAY",
    1: "TUESDAY",
    2: "WEDNESDAY",
    3: "THURSDAY",
    4: "FRIDAY",
    5: "SATURDAY",
    6: "SUNDAY",
    
}
print(result[calendar.weekday(year, month, day)])
```

# Explain
문제가 calendar의 함수에 다 정의되어있었다!!<br/>
그냥 불러다가 쓰면된다.<br/>
output만 capital letters로 출력해주었다.<br/><br/><br/><br/>


<!-- <br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
score_list = []
for _ in range(int(input())):
    name = input()
    score = float(input())
    score_list.append([name, score])
second_highest = sorted(set([score for name, score in score_list]))[1]
print('\n'.join(sorted([name for name, score in score_list if score == second_highest])))
```
<hr align="left" style="border: solid 10px gray;">
<br/><br/><br/><br/><br/> -->

# References
<ul>
  <li><a href="https://docs.python.org/2/library/calendar.html#calendar.setfirstweekday" target="_blank">https://docs.python.org/2/library/calendar.html#calendar.setfirstweekday</a></li>
  
</ul>

