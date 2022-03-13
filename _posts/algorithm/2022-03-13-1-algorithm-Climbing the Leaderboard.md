---
layout: single
title:  "[Hackerrank] Climbing the Leaderboard"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
An arcade game player wants to climb to the top of the leaderboard and track their ranking. The game uses <a href="https://en.wikipedia.org/wiki/Ranking#Dense_ranking_.28.221223.22_ranking.29">Dense Ranking</a>, so its leaderboard works like this:

- The player with the highest score is ranked number 1 on the leaderboard.
- Players who have equal scores receive the same ranking number, and the next player(s) receive the immediately following ranking number.
 



<br/><br/>


<br/><br/><br/>

# 제한사항
<h2>Input Format</h2>

- The first line contains an integer n, the number of players on the leaderboard.
- The next line contains n space-separated integers ranked[i], the leaderboard scores in decreasing order.
- The next line contains an integer, m, the number games the player plays.
- The last line contains m space-separated integers player[j], the game scores.

<h2>Output Format</h2>

- int[m]: the player's rank after each new score

<h2>Others</h2>
<ul>
<li>1 &leq; n &leq; 2 &times; 10<sup>5</li>
<li>1 &leq; m &leq; 2 &times; 10<sup>5</li>
<li>0 &leq; ranked[i] &leq; 10<sup>9</sup> for 0 &leq; i &lt; n</li>
<li>0 &leq; player[j] &leq; 10<sup>9</sup> for 0 &leq; j &lt; m</li>
<li>The existing leaderboard, ranked, is in descending order.</li>
<li>The player's scores, player, are in ascending order.</li>

</ul>

<br/><br/><br/>



# 입출력 예
Sample Input 0
```
7
100 100 50 40 40 20 10
4
5 25 50 120
```
Sample Output 0
```
6
4
2
1
```

# Idea
<p>
일단 이 문제가 처음으로 시간초과로 저를 반겨주었습니다.<br/>
입출력 예시 설명만으로도 문제가 이해가 되실것입니다.<br/>
player들의 ranking을 return해주면 되는데요.<br/>
동률의 rank에서는 1개로 인정합니다.<br/>
이게 무슨말이냐면, 1등 1명, 2등이 2명일경우 보통은 다음순위를 4등으로 표기하지만 여기서는 3등으로 표기합니다.<br/>
코드를 보시면서 시간초과를 어떻게 해결했는지 설명드리겠습니다.

</p>
<br/><br/><br/>

# Code
```python
def climbingLeaderboard(ranked, player):
    # Write your code here
    rank = list(set(ranked))
    rank.sort()
    result = list()
    
    for i in range(len(player)):
        num = 1
        for j in range(len(rank)):
            if player[i] < rank[j]:
                num += 1
        result.append(num)
    return result
```
위의 코드로 제출을 하니, 정상작동은 하지만 시간초과가 떳습니다.</br>
그래서 어떻게 하면 시간을 줄일수 있을지 고민을 해 보았고, </br>
이중 포문이 아무래도 시간을 많이 잡아먹는것 같았다.</br>
</br>
정렬이 되어있는 상태에서 숫자를 비교하는게 O(n)의 시간복잡도를 가지므로,
binary-search 알고리즘을 적용해 보기로 하였다.</br>
</br>
binary-search 알고리즘은





## 최종코드
```python

```


# Explain
생략.




<br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">


<hr align="left" style="border: solid 10px gray;">
<br/><br/>





# References
<ul>
  <li><a href="https://www.youtube.com/watch?v=FMxA_g9oQnA" target="_blank">https://www.youtube.com/watch?v=FMxA_g9oQnA</a></li>
  
  
</ul>

