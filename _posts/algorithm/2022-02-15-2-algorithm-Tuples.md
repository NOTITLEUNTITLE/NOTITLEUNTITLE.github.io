---
layout: single
title:  "[Hackerrank] Tuples"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Given an integer, n, and n space-separated integers as input, create a tuple, t, of those n integers. Then compute and print the result of hash(t).

정수 n으로 튜플 t를 만드는것과 최종 결과물인 hash(t)라는것만 알면 문제는 풀릴것같다<br/>
아래에 있는 제한사항을 읽어보니 hash()라는 내장함수가 존재하므로, 쉽게 될것같다!<br/>




<br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line contains an integer, n, denoting the number of elements in the tuple.<br/>
The second line contains n space-separated integers describing the elements in tuple t.
<h2>Output Format</h2>
Print the result of hash(t).
<h2>Others</h2>

- hash() is one of the functions in the __builtins__ module, so it need not be imported.

<br/><br/><br/>



# 입출력 예
Sample Input 0
```
2
1 2
```
Sample Output 0
```
3713081631934410656
```

# Idea
<p>
어려운 문제인줄 알았는데 너무 쉽게 해결했다..ㅋㅋ

</p>
<br/><br/><br/>

# Code
```python
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    tuple_list = tuple(integer_list)
    print(hash(tuple_list))
    
```

# Explain
설명은 생략하겠습니다.<br/>
제일 밑에있는 링크를 참조해주세요<br/>


<br/><br/><br/>
<!-- 
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
if __name__ == '__main__':
    L = []
    for _ in range(0, int(raw_input())):
        user_input = raw_input().split(' ')
        command = user_input.pop(0)
        if len(user_input) > 0:
            if 'insert' == command:
                eval("L.{0}({1}, {2})".format(command, user_input[0], user_input[1]))
            else:
                eval("L.{0}({1})".format(command, user_input[0]))
        elif command == 'print':
            print L
        else:
            eval("L.{0}()".format(command))

```
<hr align="left" style="border: solid 10px gray;">

<br/>
이 코드는 eval()이라는 함수를 사용했다.
<br/>

```
eval is evil
```
이라는 말이있는데, 확실히 위험하긴하다!!<br/>
```python
x = input()
print eval(x)
```
위와 같은 코드가 있을때...
```
__import__('os').system('rm -rf *')
```
폭탄이 터졌다 ㅋㅋㅋ

<br/><br/>
다른사람의 풀이 #2
<hr align="left" style="border: solid 10px gray;">

```python
if __name__ == '__main__':
    import builtins
    
    N = int(input())
    list_ = list()
    for _ in range(N):
        
        func_ = str(input()).split()
        getattr(builtins,'print')(list_) if func_[0] == 'print' else getattr(list_,func_[0])(*map(int,func_[1:]))
```

<hr align="left" style="border: solid 10px gray;">
<br/><br/> -->



# References
<ul>
  <li><a href="https://www.geeksforgeeks.org/python-hash-method/#:~:text=Python%20hash()%20function%20is,while%20looking%20at%20a%20dictionary." target="_blank">https://www.geeksforgeeks.org/python-hash-method/#:~:text=Python%20hash()%20function%20is,while%20looking%20at%20a%20dictionary.</a></li>
  <li><a href="https://docs.python.org/3.6/library/functions.html?highlight=hash#hash" target="_blank">https://docs.python.org/3.6/library/functions.html?highlight=hash#hash</a></li>
  
</ul>

