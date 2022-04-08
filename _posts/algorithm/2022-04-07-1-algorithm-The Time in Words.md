---
layout: single
title:  "[Hackerrank] The Time in Words"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---



# Idea

<p>
print문 작성할줄 아는지 물어보는 문제 인것같다.<br/>
숫자를 문자로 변환할려면 미리 입력을 다 해야하는데, 귀찮아서 다른사람의 코드를 참고했습니다.<br/>


</p>
<br/><br/><br/>

# Code

```python
def timeInWords(h, m):
    # Write your code here
    minutes = ['zero','one','two','three','four','five',
           'six','seven','eight','nine','ten',
           'eleven','twelve','thirteen','fourteen',
           'fifteen','sixteen','seventeen','eighteen',
           'nineteen','twenty','twenty one', 'twenty two',
           'twenty three','twenty four','twenty five',
           'twenty six','twenty seven','twenty eight',
           'twenty nine', 'thirty']
    hours = ['pad','one','two','three','four','five',
         'six','seven','eight','nine','ten','eleven','twelve']
    if m == 0:
        return (f"{hours[h]} o' clock")
    elif m >0 and m < 30:
        if m == 15:
            return (f"quarter past {hours[h]}")
        elif m == 1:
            return (f"{minutes[m]} minute past {hours[h]}")
        else:
            return (f"{minutes[m]} minutes past {hours[h]}")
    elif m == 30:
        return (f"half past {hours[h]}")
    elif m>30 and m<60:
        new_minute = 60-m
        new_hour = h+1
        if m == 45:
            return (f"quarter to {hours[new_hour]}")
        elif new_minute == 1:
            return (f"{minutes[new_minute]} minute to {hours[new_hour]}")
        else:
            return (f"{minutes[new_minute]} minutes to {hours[new_hour]}")
        
```


# Explain
쉬운 문제 푸니깐, 금방 풀리네요<br/>
<br/>


<br/><br/><br/>



<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->


