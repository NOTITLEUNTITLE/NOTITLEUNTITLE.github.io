---
layout: single
title:  "[Hackerrank] Grading Students"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
HackerLand University has the following grading policy:

- Every student receives a grade in the inclusive range from 0 to 100.
- Any grade less than 40 is a failing grade.

Sam is a professor at the university and likes to round each student's  according to these rules:
- If the difference between the grade and the next multiple of 5 is less than 3, round grade up to the next multiple of 5.
- If the value of grade is less than 38, no rounding occurs as the result will still be a failing grade.

<br/><br/><br/>
성적을 입력받고, 입력받은 성적을 5또는10을 뺏을때 3보다 작으면 반올림하고, 3보다 같거나 크면, 반올림 없다.
40점미만은 반올림 없다.




<br/><br/><br/>

# 제한사항

## Input Format
- The first line contains a single integer, n, the number of students.
- Each line i of the n subsequent lines contains a single integer, grades[i].

## Output Format

- int[n]: the grades after rounding as appropriate



## Others
- 1 &le; n &le; 60
- 0 &le; grades[i] &le; 100


<br/><br/><br/>


# 입출력 예
Sample Input
```
haveaniceday
```
Sample Output
```
hae and via ecy
```

# Idea

<p>
처음에 5의 배수를 어떻게 구해야 하나 고민했는데, 생각해보니깐 나머지 개념을 이용해도 바로 풀리는걸로<br/>


</p>
<br/><br/><br/>

# Code

```python
def gradingStudents(grades):
    # Write your code here
    result = list()
    for data in grades:
        temp = math.ceil(data / 5)
        # print(temp)
        if data < 38:
            print(f"temp = {temp*5} input data = {data}")
            result.append(data)
        elif data % 5 not in [3,4]:
            print(f"temp = {temp*5} input data = {data}")
            result.append(data)
        else:
            #  temp - data >= 3:
            print(f"temp = {temp*5} input data = {data}")
            result.append((temp)*5)
    return result

```


# Explain
파이썬스럽지는 않지만 문제를 푼거에 만족


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->


