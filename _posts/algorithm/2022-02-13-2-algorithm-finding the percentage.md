---
layout: single
title:  "[Hackerrank] Finding the percentage"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
The provided code stub will read in a dictionary containing key/value pairs of name:[marks] for a list of students. Print the average of the marks array for the student name provided, showing 2 places after the decimal.



<br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line contains the integer n, the number of students' records. The next n lines contain the names and marks obtained by a student, each value separated by a space. The final line contains query_name, the name of a student to query.
<h2>Output Format</h2>
Print one line: The average of the marks obtained by the particular student correct to 2 decimal places.
<h2>Others</h2>

 - 2 $\leq$  n $\leq$ 10
 - 0 $\leq$ marks[i] $\leq$ 100
 - length of marks arrays = 3



# 입출력 예
Sample Input 0
```
3
Krishna 67 68 69
Arjun 70 98 63
Malika 52 56 60
Malika
```
Sample Output 0
```
56.00
```

# Idea
<p>
백준이랑 프로그래머스 같은 국내사이트들도 훌륭하지만,<br/>
영어공부의 필요성을 느껴서, hackerrank site의 문제들을 풀고있다.<br/>
순서대로 풀고있어서, 쉬운문제를 풀고있지만, 점점 어려워질것같다!<br/>
아래와 같이 풀어보았다..<br/>
코드잼도 풀어서 올려야겠다.

</p>
<br/><br/><br/>

# Code
```python

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    select_score = list()
    for key, scores in student_marks.items():
        if key == query_name:
            select_score = scores
    
    
    print(f"{(sum(select_score) / len(select_score)):.2f}")
```

# Explain
baseline code들을 최대한 안 건드리고 푸는식으로 계속 진행하고 있다.<br/>
설명 패스!!<br/>


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

<!-- # References
<ul>
  <li><a href="https://docs.python.org/3/howto/sorting.html" target="_blank">https://docs.python.org/3/howto/sorting.html</a></li>
  <li><a href="https://www.programiz.com/python-programming/methods/list/sort" target="_blank">https://www.programiz.com/python-programming/methods/list/sort</a></li>
</ul>   -->

