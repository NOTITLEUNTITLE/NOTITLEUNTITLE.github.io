---
layout: single
title:  "[Hackerrank] The Grid Search"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---



# Idea

<p>
CNN처럼 풀어보고자 했다.<br/>
1,2,3,4,5<br/>
1,2,3,4,5<br/>
1,2,3,4,5<br/>
1,2,3,4,5<br/>
1,2,3,4,5<br/>
위와 같은 배열에 일치하는 패턴이<br/>
2,3<br/>
2,3<br/>
일때 패턴의 row와 col의 길이만큼을 grid에서 완전탐색으로 Search를 하는것이다.<br/>

<a href="http://taewan.kim/post/cnn/" target="_blank">cnn 설명자료</a>를 클릭하시면 이해가 되실겁니다.<br/>


<br/>


</p>
<br/><br/><br/>

# Code

```python
def gridSearch(G, P):
    # Write your code here
    target = [[ P[i][j] for j in range(len(P[0]))]for i in range(len(P))]
    # Like CNN
    for i in range(len(G) - len(P)):
        for j in range(len(G[0]) - len(P[0]) ):
            arr = [[G[row+i][col+j] for col in range(len(P[0]))] for row in range(len(P))]
            if target == arr:
                return "YES"
    return "NO"
```


# Explain
시간초과는 그렇다 치더라도, 어째서 틀린걸까요?ㅜㅜ<br/>
다른분의 코드를 참고해봤습니다.<br/>
<br/>

다른 사람의 풀이 #1
```python
def gridSearch(G, P):
    rs="NO"
    for i in range(len(G)-len(P)+1):
        for j in range(len(G[0])-len(P[0])+1):
            tt=0
            for x in range(len(P)):
                if G[i+x][j:(j+len(P[0]))]==P[0+x][:]:
                    tt+=1
            if tt==len(P):
                rs="YES"
    return rs 
```

다른 사람의 풀이 #2
```python
def gridSearch(G, P):
    pat = P[0]; spot = []
    l = len(pat); t_len = len(P)
    for i, pattern in enumerate(G):
        for j in range(len(pattern)):
            if G[i][j:j+l] == pat:
                spot.append([i, [j, j+l]])
    count = 0
    # print(len(spot))
    print(spot)
    
    for r, c in spot:
        s = c[0]; end = c[1]; drow = r
        for each in P:
            if G[drow][s:end] == each:
                count += 1
            else:
                count = 0; break
            drow += 1
        if count == t_len:
            return 'YES'
    return 'NO'
```

일단 저처럼 코드를 짜면 시간초과가 다른 분들에게도 나오는것을 확인했습니다.<br/>
1번분의 풀이가 저와 비슷하다고 할 수 있습니다.<br/>
1번분도 시간초과가 뜨셧습니다.<br/>
2번째분은 spot이라는 list에 패턴이 일치한지 파악한뒤 인덱스 값을 넣어주었습니다.<br/>
그런다음 r은 row, c은 col으로 pattern과 일치하는지 확인한후, return하는 구조이네요.<br/>
저의 코드에서는 시간초과 문제만 나오게 고쳐보았습니다.<br/>
1번분의 코드처럼 각각의 for문에서 +1 만 시켜주면 됩니다.<br/>
```python
for i in range(len(G) - len(P) + 1):
    for j in range(len(G[0]) - len(P[0]) + 1):
```



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->


