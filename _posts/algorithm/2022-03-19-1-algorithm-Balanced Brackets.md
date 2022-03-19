---
layout: single
title:  "[Hackerrank] Balanced Brackets"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명

A bracket is considered to be any one of the following characters: (, ), {, }, [, or ].

Two brackets are considered to be a matched pair if the an opening bracket (i.e., (, [, or {) occurs to the left of a closing bracket (i.e., ), ], or }) of the exact same type. There are three types of matched pairs of brackets: [], {}, and ().

A matching pair of brackets is not balanced if the set of brackets it encloses are not matched. For example, {[(])} is not balanced because the contents in between { and } are not balanced. The pair of square brackets encloses a single, unbalanced opening bracket, (, and the pair of parentheses encloses a single, unbalanced closing square bracket, ].

By this logic, we say a sequence of brackets is balanced if the following conditions are met:

It contains no unmatched brackets.
The subset of brackets enclosed within the confines of a matched pair of brackets is also a matched pair of brackets.
Given n strings of brackets, determine whether each sequence of brackets is balanced. If a string is balanced, return YES. Otherwise, return NO.


<br/><br/>
스택으로 풀수있는 여러 유명한 문제들중 하나인 괄호검사 문제입니다!!


<br/><br/><br/>

# 제한사항

## Input Format

- 1개의 String이 주어집니다.

## Output Format

- 각각의 괄호들이 정상적이면 YES, 괄호의 순서 혹은 갯수가 맞지 않으면 NO를 리턴하면 됩니다.


## Others
- 중요하지 않아 보입니다!

<br/><br/><br/>


# 입출력 예

- 생략

# Idea

<p>
어렵지 않게 풀립니다.<br/>
다른분들의 코드를 보면서 힐링을 하게 됬네요<br/>

</p>
<br/><br/><br/>

# Code

```python
def isBalanced(s):
    # Write your code here
    stack1 = list() # {}
    
    for item in s:
        if item in '}])' and len(stack1) == 0:
            return "NO"
        if item == '}' and stack1[-1] != '{':
            return "NO"
        if item == ']' and stack1[-1] != '[':
            return "NO"
        if item == ')' and stack1[-1] != '(':
            return "NO"
        if item == '{':
            stack1.append(item)
        if item == '[':
            stack1.append(item)
        if item == '(':
            stack1.append(item)
        if item == '}':
            stack1.pop()
        if item == ']':
            stack1.pop()
        if item == ')':
            stack1.pop()
    if len(stack1) == 0:
        return "YES"
    else:
        return "NO"
```


# Explain
저는 굉장히 무식하게 풀었는데, dictionary를 활용했으면 아래의 코드처럼 깔끔하게 나왔을것 같은데 ㅠㅠ<br/>
그래도 나름 조건문의 순서를 신경써야 합니다!!<br/>


<br/><br/><br/>


## 다른사람의 풀이

```python
def isBalanced(s):
  if len(s) % 2 != 0:
    return 'NO'
  
  stack = []
  closing = ['}', ']', ')']
  pairs = {'}':'{', ']':'[', ')':'('}
  
  for bracket in s:
    if bracket in closing:
      if not stack or stack.pop() != pairs.get(bracket):
        return 'NO'
    else:
      stack.append(bracket)
      
  return 'YES' if not stack else 'NO'
```
위에 분은 dictionary를 이용해서 푸셨고, 나름 나이스한 풀이로 보입니다.
<br/><br/><br/>


```python
def isBalanced(s):
    # Write your code here
    while '{}' in s or '[]' in s or '()' in s:
        s = s.replace('{}', '' )
        s = s.replace('[]', '')
        s = s.replace('()', '')
    if len(s) == 0:
        return("YES")
    else:
        return("NO")

```
이거는 정말 할말이 없네요... 이렇게 푸셨네.. 스택 안쓰시고...
<br/><br/><br/>

<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->

