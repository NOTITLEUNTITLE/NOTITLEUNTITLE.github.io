import os

os.chdir('/mnt/d/notitle-github-blog/NOTITLEUNTITLE.github.io/_posts/polymath/')
path = os.getcwd()
print(path)



filename1 = "2022-01-28-"
filename2 = "-bandit-"
seq = 29
#########################################################################
# for i in range(1, 6):
#   os.unlink(filename1 + str(i) + filename2 + str(seq) + ".md")
#   seq += 1
#########################################################################


for i in range(6,11):
  with open(filename1 + str(i) + filename2 + str(seq) + ".md", "w", encoding="UTF8") as f:
    print('''---
layout: single
title:  "[bandit] Bandit Level 21 → Level 22"
categories: bandit
tag: [wargame, linux]
toc: true
author_profile: false
---



# 문제설명
<hr size=10 noshade>

<br/>
<hr size=10 noshade>

# 문제풀이

<img src="../../images/2022-01-27/bandit23-2.PNG">

<p><br/>

</p>
''', file=f);
  seq += 1