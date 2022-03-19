---
layout: single
title:  "[Hackerrank] Simple Text Editor"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Implement a simple text editor. The editor initially contains an empty string, S. Perform Q operations of the following 4 types:

append(W) - Append string W to the end of S.
delete(k) - Delete the last k characters of S.
print(k) - Print the K character of S.
undo() - Undo the last (not previously undone) operation of type 1 or 2, reverting S to the state it was in prior to that operation.


<br/><br/>
역시 제일 중요한 언어는 영어다..<br/>
undo 해석을 하다가 1,2 type에 대해서만 해야하는데 3번 type에 대해서도 실행하는 바람에 시간이 꽤 많이 걸렸다..<br/>


<br/><br/><br/>

# 제한사항

## Input Format

- 제일 먼저 실행할 명령어의 갯수가 주어집니다.
- 위에서 입력한 숫자만큼 명령어들이 쭉 나열됩니다.

## Output Format

- 3번 타입(출력)에 대해서만 출력하면 되겠습니다!!


## Others
- 중요하지 않아 보입니다!

<br/><br/><br/>


# 입출력 예
Sample Input
```
8       
1 abc   
3 3     
2 3     
1 xy
3 2
4 
4 
3 1
```
Sample Output
```
c
y
a
```

# Idea

<p>
undo를 1,2,3번의 전체 타입에 대해서 실행하는 바람에 문제를 다시 읽고 해결해 보았습니다.<br/>

</p>
<br/><br/><br/>

# 잘못 이해한 코드..

```python
text = ""
previous_op = 0
previous = ""
for i in range(int(input())):
    line = input()
    if int(line.split(" ")[0]) == 4:
        if previous_op == 1:
            text += previous
        elif previous_op == 2:
            text = text[:-int(previous)]
        elif previous_op == 3:
            index = int(previous) - 1
            print(text[index])
            
    if len(line.split(" ")) == 2:
        current_op = int(line.split(" ")[0])
        current = line.split(" ")[1]
    
    if int(line.split(" ")[0]) == 1:
        text += line.split(" ")[1]
    elif int(line.split(" ")[0]) == 2:
        text = text[:-int(line.split(" ")[1])]
    elif int(line.split(" ")[0]) == 3:
        index = int(line.split(" ")[1]) - 1
        print(text[index])
    previous_op = current_op
    previous = current
    
```

# 정답코드

```python
text = ""
memory = list()
for i in range(int(input())):
    line = input().strip().split(" ")
    op = int(line[0])
    if op == 1:
        memory.append(text)
        text += line[1]
    if op == 2:
        memory.append(text)
        text = text[:-int(line[1])]
    if op == 3:
        print(text[int(line[1])-1])
    if op == 4:
        text = memory.pop()
```

# Explain
다른분의 코드와 동일했습니다.<br/>
한 번 삽질을 크게 하니깐, 중요한게 어떤건지 경험해서 그런것 같아요 ..<br/>
앞으로는 지시사항 꼼꼼하게 읽어야 겠습니다.<br/>


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->

