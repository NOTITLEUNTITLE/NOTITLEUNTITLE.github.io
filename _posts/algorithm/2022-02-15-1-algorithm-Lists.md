---
layout: single
title:  "[Hackerrank] Lists"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Consider a list (list = []). You can perform the following commands:<br/>

insert i e: Insert integer e at position i.<br/>
print: Print the list.<br/>
remove e: Delete the first occurrence of integer e.<br/>
append e: Insert integer e at the end of the list.<br/>
sort: Sort the list.<br/>
pop: Pop the last element from the list.<br/>
reverse: Reverse the list.<br/><br/>
Initialize your list and read in the value of n followed by n lines of commands where each command will be of the 7 types listed above. Iterate through each command in order and perform the corresponding operation on your list.<br/>

stack이라는 자료구조를 직접 구현해보는 문제처럼,<br/>
list라는 자료구조를 한번 구현해보라는 문제인것같다!!<br/>



<br/><br/>

# 제한사항
<h2>Input Format</h2>
The first line contains an integer, n, denoting the number of commands.
Each line i of the n subsequent lines contains one of the commands described above.
<h2>Output Format</h2>

For each command of type print, print the list on a new line.

<h2>Others</h2>

- The elements added to the list must be integers.

<br/><br/><br/>



# 입출력 예
Sample Input 0
```
12
insert 0 5
insert 1 10
insert 0 6
print
remove 6
append 9
append 1
sort
print
pop
reverse
print
```
Sample Output 0
```
[6, 5, 10]
[1, 5, 9, 10]
[9, 5, 1]
```

# Idea
<p>
class 정의하고 무작정 해보자!!

</p>
<br/><br/><br/>

# Code
```python
class List:
    def __init__(self):
        self.arr = list()
        
    def insert(self, position, value):
        self.arr.insert(position, value)
    
    def append(self, value):
        self.arr.append(value)
    
    def print(self):
        print(self.arr)
    
    def remove(self, value):
        if value not in self.arr:
            print(f"It doesn't exist in the list.---->[{value}]")
            return -1
        self.arr.remove(value)
    
    def sort(self):
        self.arr.sort()
        
    def pop(self):
        if len(self.arr) == 0:
            print("Empty")
            return -1
        self.arr.pop()
    
    def reverse(self):
        if len(self.arr) == 0:
            print("Empty")
            return -1
        self.arr = self.arr[::-1]
        


if __name__ == '__main__':
    arr = List()
    for i in range(int(input())):
        query = input()
        if query == "print":
            arr.print()
        elif query == "pop":
            arr.pop()
        elif query == "reverse":
            arr.reverse()
        elif query == "sort":
            arr.sort()
        elif query.startswith("insert"):
            query, position, value = query.split()
            arr.insert(int(position), int(value))
        elif query.startswith("append"):
            query, value = query.split()
            arr.append(int(value))
        elif query.startswith("remove"):
            query, value = query.split()
            arr.remove(int(value))
```

# Explain
어려운건 하나도 없지만, 흥미로운점을 발견했다.<br/>
hackerrank의 code edit는 ascii character만 작성해서 제출할수있다.<br/>
error 출력을 한글로 할려고 했더니, 제출이 불가했다.<br/>
저의 코드를 보시는것보다, 밑에 코드를 보는게 더 흥미로울것같습니다!!<br/>


<br/><br/><br/>
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
<br/><br/>



# References
<ul>
  <li><a href="https://docs.python.org/3.6/library/functions.html#getattr" target="_blank">https://docs.python.org/3.6/library/functions.html#getattr</a></li>
  <li><a href="https://docs.python.org/3.6/library/functions.html?highlight=eval#eval" target="_blank">https://docs.python.org/3.6/library/functions.html?highlight=eval#eval</a></li>
  
</ul>

