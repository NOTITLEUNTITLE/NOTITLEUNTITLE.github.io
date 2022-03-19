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

<br/><br/><br/>

# 제한사항

## Input Format

- The first line contains an integer n, the number of players on the leaderboard.
- The next line contains n space-separated integers ranked[i], the leaderboard scores in decreasing order.
- The next line contains an integer, m, the number games the player plays.
- The last line contains m space-separated integers player[j], the game scores.

## Output Format

- int[m]: the player's rank after each new score

## Others
- 1 &le; n &le; 2 &times; 10<sup>5
- 1 &le; m &le; 2 &times; 10<sup>5
- 0 &le; ranked[i] &le; 10<sup>9</sup> for 0 &le; i &lt; n
- 0 &le; player[j] &le; 10<sup>9</sup> for 0 &le; j &lt; m
- The existing leaderboard, ranked, is in descending order.
- The player's scores, player, are in ascending order.

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

위의 코드로 제출을 하니, 정상작동은 하지만 시간초과가 떳습니다.   
그래서 어떻게 하면 시간을 줄일수 있을지 고민을 해 보았고,    
이중 포문이 아무래도 시간을 많이 잡아먹는것 같았다.   
   
정렬이 되어있는 상태에서 숫자를 비교하는게 O(n)의 시간복잡도를 가지므로,   
binary-search 알고리즘을 적용하려고 하였으나, 예전에 python package중에서 bisect라는것을 본적이 있었던것 같았다.   

찾아보았더니 내가 원했던 결과를 얻어낼 수 있었다.(역시 파이썬..대다나다)   






## 최종코드
```python

def climbingLeaderboard(ranked, player):
    # Write your code here
    rank = sorted(list(set(ranked)))
    result = list()
    import bisect
    for i in range(len(player)):
        rere = len(rank) - bisect.bisect(rank, player[i]) + 1
        result.append(rere)
    return result

```


# Explain
bisect에 관한 설명은 Reference를 참고해 주세요!




<br/><br/><br/>
다른사람의 풀이 #1
<hr align="left" style="border: solid 10px gray;">

```python
def climbingLeaderboard(ranked, player):
    # Write your code here
    import bisect
    ranked = list(set(ranked))
    ranked.sort()
    length = len(ranked)
    return [(length-bisect.bisect(ranked, i))+1 for i in player]
```
list comprehension으로 문제를 간단하게 풀었네요.. 대다나다

<hr align="left" style="border: solid 10px gray;">
<br/><br/>





# References

<ul>
  <li><a href="https://programming119.tistory.com/196" target="_blank">https://programming119.tistory.com/196</a></li>
  <li><a href="https://docs.python.org/3.8/library/bisect.html" target="_blank">https://docs.python.org/3.8/library/bisect.html</a></li>
  <li><a href="https://11001.tistory.com/71" target="_blank">https://11001.tistory.com/71</a></li>
</ul>

