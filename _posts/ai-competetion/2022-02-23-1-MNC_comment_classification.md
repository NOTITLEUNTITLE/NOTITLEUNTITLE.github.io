---
layout: single
title:  "[MNC] 뉴스 악성 댓글 분류 대회 - 준비"
categories: ML
tag: [python, pytorch, ML]
toc: true
author_profile: false
---

# 이번에는 NLP대회를 시작하면서 공부한 내용들을 정리해보겠습니다.

- 전체적인 흐름 
  - 대회의 목적은 댓글을 분류하는것이므로, 댓글들로만 학습된 모델들을 사용하는것이 좋아보입니다.
  - 최종적으로 결과물은 class별로 구분하는것이므로, fine tuning을 해야합니다.
  - 전처리에대한 고민을 해야합니다.!!
<br/><br/><br/>

# Reference
<ul>
<li><a href="https://bigwaveai.tistory.com/1?category=953606" target="_blank">https://bigwaveai.tistory.com/1?category=953606</a></li>
<li><a href="https://complexoftaste.tistory.com/2" target="_blank">https://complexoftaste.tistory.com/2</a></li>
<li><a href="https://drive.google.com/file/d/1EkymGGy-6Fzh-DobbplbZcMThG9TP58P/view" target="_blank">https://drive.google.com/file/d/1EkymGGy-6Fzh-DobbplbZcMThG9TP58P/view</a></li>
<li><a href="https://github.com/Seolini/KoBERT_Korean_multi_classification/blob/main/KoBERT_%ED%95%9C%EA%B5%AD%EC%96%B4_7%EA%B0%9C%EA%B0%90%EC%A0%95_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.ipynb" target="_blank">https://github.com/Seolini/KoBERT_Korean_multi_classification/blob/main/KoBERT_%ED%95%9C%EA%B5%AD%EC%96%B4_7%EA%B0%9C%EA%B0%90%EC%A0%95_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.ipynb</a></li>
<li><a href="https://drive.google.com/file/d/1EkymGGy-6Fzh-DobbplbZcMThG9TP58P/view" target="_blank">https://drive.google.com/file/d/1EkymGGy-6Fzh-DobbplbZcMThG9TP58P/view</a></li>
</ul>
<br/><br/><br/>


# 데이터 전처리
```python
print(len(train_data))

# 한글과 공백 제외하고 전체 다 제거.
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 만약 한글이 없는 댓글이 있다면, null값으로 변경한후,null값이 몇개인지 확인해본다.
train_data['comment'] = train_data['comment'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['comment'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

# null값이 있는데이터를 삭제해준다.
train_data = train_data.dropna(how = 'any')
print(len(train_data))

```



</ul>