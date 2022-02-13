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
알고리즘도 필요없고, 구현만 하면 되는 문제이다.<br/>
아래와 같이 풀어보았다 ㅎ

</p>
<br/><br/><br/>

# Code
```python
class Record:
    def __init__(self,name, score):
        self.name = name
        self.score = score
    def __repr__(self):
        return repr((self.name, self.score))
    def value_return(self):
        return self.score
    def name_return(self):
        return self.name


if __name__ == '__main__':
    n = int(input())
    record_list = list()
    for i in range(n):
        name = input()
        score = float(input())
        record_list.append(Record(name, score))
    
    record_list = list(sorted(record_list, key=lambda x:x.score))
    score_list = list()
    
    for i in range(n):
        score_list.append(record_list[i].value_return())

    second_score = sorted(set(score_list))[1]
    name_list = list()
    for i in range(n):
        if second_score == record_list[i].value_return():
            name_list.append(record_list[i].name_return())
    name_list.sort()
    for name in name_list:
        print(name)
```

# Explain
파이썬스럽게 못 풀었다...아쉽다.<br/>
리스트에 name과 score를 입력해주고, score를 기준으로 2번째값을 구해낸 다음,
2번째 값을 가지고있는 name들을 리스트에 넣어주고, 리스트를 알파벳순으로 정렬해서 출력한다!<br/>
나도 밑에 풀이처럼 list comprehension을 적용했다면, 파이썬스럽게 코딩할수있었는데 아쉽다 ㅎ...<br/>


<br/><br/><br/>
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
<br/><br/><br/><br/><br/>

# References
<ul>
  <li><a href="https://docs.python.org/3/howto/sorting.html" target="_blank">https://docs.python.org/3/howto/sorting.html</a></li>
  <li><a href="https://www.programiz.com/python-programming/methods/list/sort" target="_blank">https://www.programiz.com/python-programming/methods/list/sort</a></li>
</ul>  

