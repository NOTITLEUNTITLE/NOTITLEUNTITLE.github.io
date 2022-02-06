---
layout: single
title:  "[programmers] JadenCase 문자열 만들기"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
<br/>JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다. 문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.
<br/>


# 제한사항
<ul>
<li>s는 길이 1 이상 200 이하인 문자열입니다.</li>
<li>s는 알파벳과 숫자, 공백문자(" ")로 이루어져 있습니다.</li>
<ul>
<li>숫자는 단어의 첫 문자로만 나옵니다. </li>
<li>숫자로만 이루어진 단어는 없습니다.</li>
<li>공백문자가 연속해서 나올 수 있습니다.</li>
</ul>
<li>첫 문자가 영문이 아닐때에는 이어지는 영문은 소문자로 씁니다. ( 첫번째 입출력 예 참고 )</li>
</ul>
<br/>




  
  
  





# 입출력 예

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">s</th>
    <th class="tg-0lax">return</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">"3people unFollowed me"</td>
    <td class="tg-0lax">"3people Unfollowed Me"</td>
  </tr>
  <tr>
    <td class="tg-0lax">"for the last week"</td>
    <td class="tg-0lax">"For The Last Week"</td>
  </tr>
  

</tbody>
</table>



<br/>


# Idea
<p>

```python
for i in range(len(s)):
  temp = s[i][0].upper() + s[i][1:].lower()
```
처음에 이렇게 짜봤는데 잘 space가 연속적으로 나오는 부분에 대해서 처리가 부족한것 같습니다.<br/>
split을 사용할때 주의해야 합니다.

</p>


# Code
```python
def solution(s):
    result = list()
    s = s.split(" ")
    print(s)
    for ch in s:
        print(ch)
        if ch == "":
            result.append(ch)
            continue
        word = list(ch.lower())
        word[0] = word[0].upper()
        temp = "".join(word)
        result.append(temp)
    print(result)
    return " ".join(result)
    
    


```

# Explain
<strong>설명 생략</strong><br/>
다른분들의 코드도 보도록 하죠!!<br/>
<br/>

다른 풀이 #1

```python
def solution(s):
    answer = ''
    s=s.split(' ')
    for i in range(len(s)):
        # capitalize 내장함수를 사용하면 첫 문자가 알파벳일 경우 대문자로 만들고
        # 두번째 문자부터는 자동으로 소문자로 만든다
        # 첫 문자가 알파벳이 아니면 그대로 리턴한다
        s[i]=s[i].capitalize()
    answer=' '.join(s)
    return answer
```


<!-- # References
<ul>
  <li><a href="https://www.geeksforgeeks.org/python-all-possible-n-combination-tuples/" target="_blank">https://www.geeksforgeeks.org/python-all-possible-n-combination-tuples/</a></li>
  <li><a href="https://www.geeksforgeeks.org/python-all-pair-combinations-of-2-tuples/" target="_blank">https://www.geeksforgeeks.org/python-all-pair-combinations-of-2-tuples/</a></li>
  
</ul>   -->
