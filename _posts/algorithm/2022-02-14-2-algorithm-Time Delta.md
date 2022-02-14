---
layout: single
title:  "[Hackerrank] Time Delta"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
When users post an update on social media,such as a URL, image, status update etc., other users in their network are able to view this new post on their news feed. Users can also see exactly when the post was published, i.e, how many hours, minutes or seconds ago.

Since sometimes posts are published and viewed in different time zones, this can be confusing. You are given two timestamps of one such post that a user can see on his newsfeed in the following format:

Day dd Mon yyyy hh:mm:ss +xxxx

Here +xxxx represents the time zone. Your task is to print the absolute difference (in seconds) between them.

문제가 길어서 약간 해석하자면,   
유저가 소셜미디에서 포스팅을 업로드하면, 다른 유저와의 time zone에 의해 다르게 표시될수있으므로, 혼돈될수있다.   
그러므로, 두 타임존의 절대차이(second)를 구하라!!



<br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line contains T, the number of testcases.

Each testcase contains 2 lines, representing time $ t_1 $ and time $t_2$.
<h2>Output Format</h2>

Print the absolute difference ($t_1$ - $t_2$)  in seconds.

<h2>Others</h2>

- Input contains only valid timestamps
- year $\leq$ 3000




# 입출력 예
Sample Input 0
```
2
Sun 10 May 2015 13:54:36 -0700
Sun 10 May 2015 13:54:36 -0000
Sat 02 May 2015 19:54:36 +0530
Fri 01 May 2015 13:54:36 -0000
```
Sample Output 0
```
25200
88200
```

# Idea
<p>
python docs를 꼭 찾아봐야합니다.<br/> 그렇지 않으면 굉장히 힘들고 긴 코딩이 준비되어있습니다....<br/>날짜 관련문제를 풀때는 도움이 될것같습니다.
<br/>
참고로 baseline code를 건드릴거는 없고, 코드에 있는 주석을 참고하시면 됩니다.!!

</p>
<br/><br/><br/>

# Code
```python
import math
import os
import random
import re
import sys

# Complete the time_delta function below.

# Time Delta in Python - Hacker Rank Solution START
from datetime import datetime
def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    return str(int(abs((t1-t2).total_seconds())))
# Time Delta in Python - Hacker Rank Solution END

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    print(os.getcwd())
    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()
```

# Explain
아래에 있는 reference 링크를 클릭하면 docs로 이동하게 됩니다. 정말 다양한 내장함수들을 사용해보실수있습니다.<br/>
이 문제를 풀기위해서 datetime.strptime()를 사용했으며, 간략하게 설명드리면,<br/>
```python
strptime(date_string, format)
```
%a = Weekday as locale’s abbreviated name.<br/>
%d = Day of the month as a zero-padded decimal number.<br/>
%b = Month as locale’s abbreviated name.<br/>
%Y = Year with century as a decimal number.<br/>
%H = Hour (24-hour clock) as a zero-padded decimal number.<br/>
%M = Minute as a zero-padded decimal number.<br/>
%S = Second as a zero-padded decimal number.<br/>
%z = UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive).<br/>
<br/>
total_second()<br/>
Return the total number of seconds contained in the duration. Equivalent to td / timedelta(seconds=1).
<br/><br/>



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
  <li><a href="https://docs.python.org/3.6/library/datetime.html" target="_blank">https://docs.python.org/3.6/library/datetime.html</a></li>
  
</ul>

