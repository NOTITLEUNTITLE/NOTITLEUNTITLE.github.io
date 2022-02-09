---
layout: single
title:  "[MNC] 대출자 채무 불이행 여부 예측 모델[최종본]"
categories: ML
tag: [python, pytorch, ML]
toc: true
author_profile: false
---


<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


</br></br></br></br></br></br>

<h1 style="font-size:10vw"><strong><center>Ah-Jji?</center></strong></h1>

</br></br></br></br></br></br>



```python
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



import catboost
import xgboost
import lightgbm
# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.experimental import enable_hist_gradient_boosting 
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
# from tpot import TPOTClassifier
```

<pre>
/home/notitle/anaconda3/envs/loan/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/notitle/anaconda3/envs/loan/lib/python3.8/site-packages/tpot/builtins/__init__.py:36: UserWarning: Warning: optional dependency `torch` is not available. - skipping import of NN models.
  warnings.warn("Warning: optional dependency `torch` is not available. - skipping import of NN models.")
</pre>

```python
df = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_submit = pd.read_csv("./data/sample_submission.csv")
df_train = df
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>int_rate</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>total_acc</th>
      <th>collections_12_mths_ex_med</th>
      <th>acc_now_delinq</th>
      <th>...</th>
      <th>term1</th>
      <th>open_acc</th>
      <th>installment</th>
      <th>revol_util</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_rec_int</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>depvar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0824</td>
      <td>21000.0</td>
      <td>29.19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3016</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>18</td>
      <td>37.74</td>
      <td>0.076</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>157.94</td>
      <td>765</td>
      <td>769</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1299</td>
      <td>80000.0</td>
      <td>4.82</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5722</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>269.52</td>
      <td>0.447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1702.42</td>
      <td>665</td>
      <td>669</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1299</td>
      <td>38000.0</td>
      <td>23.66</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6511</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7</td>
      <td>168.45</td>
      <td>0.880</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1066.64</td>
      <td>670</td>
      <td>674</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1367</td>
      <td>100000.0</td>
      <td>16.27</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>6849</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>510.27</td>
      <td>0.457</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1256.24</td>
      <td>680</td>
      <td>684</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1269</td>
      <td>30000.0</td>
      <td>25.28</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>8197</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>335.45</td>
      <td>0.416</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>871.04</td>
      <td>660</td>
      <td>664</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>0.1757</td>
      <td>65000.0</td>
      <td>17.67</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>11255</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>13</td>
      <td>718.75</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5373.29</td>
      <td>660</td>
      <td>664</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0.0890</td>
      <td>65000.0</td>
      <td>2.88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2105</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7</td>
      <td>190.52</td>
      <td>0.120</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>835.66</td>
      <td>765</td>
      <td>769</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>0.1349</td>
      <td>46000.0</td>
      <td>32.12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8998</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>217.16</td>
      <td>0.643</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1261.67</td>
      <td>665</td>
      <td>669</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>0.2115</td>
      <td>31000.0</td>
      <td>4.53</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3875</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>207.64</td>
      <td>0.731</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1357.69</td>
      <td>710</td>
      <td>714</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>0.1599</td>
      <td>125000.0</td>
      <td>33.33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34580</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>1164.42</td>
      <td>0.499</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8882.58</td>
      <td>690</td>
      <td>694</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 76 columns</p>
</div>


따로 data preprocessing은 할 필요가 없습니다.   

미리 진행 하였습니다.   

모델 학습위주로 보시겠습니다.   




```python
# pd.options.display.max_rows = 80
```


```python
# df_train.nunique()
```


```python
# df_test.nunique()
```


```python
X = df_train.drop("depvar", axis=1)
y = df_train["depvar"]
```


```python
print(len(df_train.columns))
df_train.columns
```

<pre>
76
</pre>
<pre>
Index(['int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'pub_rec', 'revol_bal', 'total_acc', 'collections_12_mths_ex_med',
       'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',
       'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens', 'emp_length1',
       'emp_length2', 'emp_length3', 'emp_length4', 'emp_length5',
       'emp_length6', 'emp_length7', 'emp_length8', 'emp_length9',
       'emp_length10', 'emp_length11', 'emp_length12', 'home_ownership1',
       'home_ownership2', 'home_ownership3', 'home_ownership4',
       'home_ownership5', 'home_ownership6', 'verification_status1',
       'verification_status2', 'verification_status3', 'purpose1', 'purpose2',
       'purpose3', 'purpose4', 'purpose5', 'purpose6', 'purpose7', 'purpose8',
       'purpose9', 'purpose10', 'purpose11', 'purpose12', 'purpose13',
       'purpose14', 'initial_list_status1', 'initial_list_status2',
       'mths_since_last_delinq1', 'mths_since_last_delinq2',
       'mths_since_last_delinq3', 'mths_since_last_delinq4',
       'mths_since_last_delinq5', 'mths_since_last_delinq6',
       'mths_since_last_delinq7', 'mths_since_last_delinq8',
       'mths_since_last_delinq9', 'mths_since_last_delinq10',
       'mths_since_last_delinq11', 'funded_amnt', 'funded_amnt_inv',
       'total_rec_late_fee', 'term1', 'open_acc', 'installment', 'revol_util',
       'out_prncp', 'out_prncp_inv', 'total_rec_int', 'fico_range_low',
       'fico_range_high', 'depvar'],
      dtype='object')
</pre>

```python
df_train["emp_length"] = 0
emp_df = df_train.iloc[:, 15:27]

df_train["purpose"] = 0
pur_df = df_train.iloc[:, 36:50]

df_train["mths_since_last_delinq"] = 0
mths_df = df_train.iloc[:, 52:63]

df_train["home_ownership"] = 0
home_df = df_train.iloc[:, 27:33]
```


```python
temp = list()
for i in tqdm(range(0,100000)):
  str1 = "emp_length"
  for j in range(1, 13):
    str2 = str(j)
    if emp_df[str1+str2][i] == 1:
      temp.append(j)
      break

df_train["emp_length"] = temp

temp = list()
for i in tqdm(range(0,100000)):
  str1 = "purpose"
  for j in range(1, 15):
    str2 = str(j)
    if pur_df[str1+str2][i] == 1:
      temp.append(j)
      break

df_train["purpose"] = temp

temp = list()
for i in tqdm(range(0,100000)):
  str1 = "mths_since_last_delinq"
  for j in range(1, 12):
    str2 = str(j)
    if mths_df[str1+str2][i] == 1:
      temp.append(j)
      break

df_train["mths_since_last_delinq"] = temp

temp = list()
for i in tqdm(range(0,100000)):
  str1 = "home_ownership"
  for j in range(1, 7):
    str2 = str(j)
    if home_df[str1+str2][i] == 1:
      temp.append(j)
      break

df_train["home_ownership"] = temp
```

<pre>
100%|██████████| 100000/100000 [00:03<00:00, 29317.28it/s]
100%|██████████| 100000/100000 [00:02<00:00, 42521.94it/s]
100%|██████████| 100000/100000 [00:02<00:00, 39833.68it/s]
100%|██████████| 100000/100000 [00:03<00:00, 26846.15it/s]
</pre>
<h2>Feature Engineering입니다. one-hot encode를 lable encode로 변경해보았습니다.</h2>

<h2>트리기반 모델을 학습할것이기에 안해도 됩니다!!!</h2>



```python
df_train = df_train.drop(columns=[
  'emp_length1', 'emp_length2', 'emp_length3', 'emp_length4',
  'emp_length5', 'emp_length6', 'emp_length7', 'emp_length8',
  'emp_length9', 'emp_length10', 'emp_length11', 'emp_length12',
  'purpose1', 'purpose2', 'purpose3', 'purpose4', 
  'purpose5', 'purpose6', 'purpose7',
  'purpose8', 'purpose9', 'purpose10', 'purpose11', 'purpose12',
  'purpose13', 'purpose14',
  'mths_since_last_delinq1','mths_since_last_delinq2',
  'mths_since_last_delinq3', 'mths_since_last_delinq4',
  'mths_since_last_delinq5', 'mths_since_last_delinq6',
  'mths_since_last_delinq7', 'mths_since_last_delinq8',
  'mths_since_last_delinq9', 'mths_since_last_delinq10',
  'mths_since_last_delinq11',
  'home_ownership1', 'home_ownership2', 'home_ownership3',
  'home_ownership4', 'home_ownership5', 'home_ownership6',
  ])
```


```python
# 선택사항
# df_train = df_train.drop(columns=[
#   "initial_list_status2", "funded_amnt", "funded_amnt_inv", "out_prncp", "fico_range_high"
#   ])
```


```python
df_train
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>int_rate</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>total_acc</th>
      <th>collections_12_mths_ex_med</th>
      <th>acc_now_delinq</th>
      <th>...</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_rec_int</th>
      <th>fico_range_low</th>
      <th>fico_range_high</th>
      <th>depvar</th>
      <th>emp_length</th>
      <th>purpose</th>
      <th>mths_since_last_delinq</th>
      <th>home_ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0824</td>
      <td>21000.0</td>
      <td>29.19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3016</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>157.94</td>
      <td>765</td>
      <td>769</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1299</td>
      <td>80000.0</td>
      <td>4.82</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5722</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1702.42</td>
      <td>665</td>
      <td>669</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1299</td>
      <td>38000.0</td>
      <td>23.66</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6511</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1066.64</td>
      <td>670</td>
      <td>674</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>11</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1367</td>
      <td>100000.0</td>
      <td>16.27</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>6849</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1256.24</td>
      <td>680</td>
      <td>684</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1269</td>
      <td>30000.0</td>
      <td>25.28</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>8197</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>871.04</td>
      <td>660</td>
      <td>664</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>0.1757</td>
      <td>65000.0</td>
      <td>17.67</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>11255</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5373.29</td>
      <td>660</td>
      <td>664</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0.0890</td>
      <td>65000.0</td>
      <td>2.88</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2105</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>835.66</td>
      <td>765</td>
      <td>769</td>
      <td>0</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>0.1349</td>
      <td>46000.0</td>
      <td>32.12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8998</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1261.67</td>
      <td>665</td>
      <td>669</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>0.2115</td>
      <td>31000.0</td>
      <td>4.53</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3875</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1357.69</td>
      <td>710</td>
      <td>714</td>
      <td>1</td>
      <td>12</td>
      <td>10</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>0.1599</td>
      <td>125000.0</td>
      <td>33.33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34580</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8882.58</td>
      <td>690</td>
      <td>694</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 37 columns</p>
</div>



```python
X_train1 = X[:20000]
X_train2 = X[20000:40000]
X_train3 = X[40000:60000]
X_train4 = X[60000:80000]
X_train5 = X[80000:]

y_train1 = y[:20000]
y_train2 = y[20000:40000]
y_train3 = y[40000:60000]
y_train4 = y[60000:80000]
y_train5 = y[80000:]
```


```python
##########
frames = [X_train1, X_train2, X_train3, X_train4]
X_train_dataset1 = pd.concat(frames)

frames = [X_train1, X_train2, X_train3, X_train5]
X_train_dataset2 = pd.concat(frames)

frames = [X_train1, X_train2, X_train4, X_train5]
X_train_dataset3 = pd.concat(frames)

frames = [X_train1, X_train3, X_train4, X_train5]
X_train_dataset4 = pd.concat(frames)

frames = [X_train2, X_train3, X_train4, X_train5]
X_train_dataset5 = pd.concat(frames)
##########

frames = [y_train1, y_train2, y_train3, y_train4]
y_train_dataset1 = pd.concat(frames)

frames = [y_train1, y_train2, y_train3, y_train5]
y_train_dataset2 = pd.concat(frames)

frames = [y_train1, y_train2, y_train4, y_train5]
y_train_dataset3 = pd.concat(frames)

frames = [y_train1, y_train3, y_train4, y_train5]
y_train_dataset4 = pd.concat(frames)

frames = [y_train2, y_train3, y_train4, y_train5]
y_train_dataset5 = pd.concat(frames)
```

<h2>이번 과제를 진행함에 있어서, 핵심적인 방법으로써, 모델의 학습용 데이터를 섞어주는것입니다.<br/></h2>

<h2>위의 코드로 설명을 하자면, 먼저 전체 데이터를 5개의 학습데이터로 나누어 줍니다.<br/></h2>

<h2>1번모델에서는 1,2,3,4의 학습데이터(전체의 80%)로 학습을 합니다.<br/></h2>

<h2>1번모델에서는 1,2,3,5의 학습데이터(전체의 80%)로 학습을 합니다.<br/></h2>

<h2>1번모델에서는 1,2,4,5의 학습데이터(전체의 80%)로 학습을 합니다.<br/></h2>

<h2>1번모델에서는 1,3,4,5의 학습데이터(전체의 80%)로 학습을 합니다.<br/></h2>

<h2>1번모델에서는 2,3,4,5의 학습데이터(전체의 80%)로 학습을 합니다.<br/></h2>

<h2>이렇게 모델을 학습시키면,<br/></h2>

<h2>1번모델은 5번데이터로 검증을 하고,<br/></h2>

<h2>2번모델은 4번데이터로 검증을 하고,<br/></h2>

<h2>3번모델은 3번데이터로 검증을 하고,<br/></h2>

<h2>4번모델은 2번데이터로 검증을 하고,<br/></h2>

<h2>5번모델은 1번데이터로 검증을 하고,<br/></h2>

<h2>최종 제출시에는 동일한 테스트 데이터들을 각 모델별로 학습을 시켜준 다음,<br/></h2>

<h2>과반이상(3개)의 동일한 label값을 제출해주게 되면 됩니다.<br/></h2>

<h2><br/></h2>

<h2><br/></h2>



```python
model1 = xgboost.XGBClassifier()
model2 = xgboost.XGBClassifier()
model3 = xgboost.XGBClassifier()
model4 = xgboost.XGBClassifier()
model5 = xgboost.XGBClassifier()

model1.fit(X_train_dataset1, y_train_dataset1)
model2.fit(X_train_dataset2, y_train_dataset2)
model3.fit(X_train_dataset3, y_train_dataset3)
model4.fit(X_train_dataset4, y_train_dataset4)
model5.fit(X_train_dataset5, y_train_dataset5)
```

<pre>
/home/notitle/anaconda3/envs/loan/lib/python3.8/site-packages/xgboost/sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
/home/notitle/anaconda3/envs/loan/lib/python3.8/site-packages/xgboost/data.py:250: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  elif isinstance(data.columns, (pd.Int64Index, pd.RangeIndex)):
</pre>
<pre>
[10:54:49] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[10:55:04] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[10:55:17] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[10:55:29] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
[10:55:41] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
</pre>
<pre>
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
</pre>

```python
model6 = lightgbm.LGBMClassifier()
model7 = lightgbm.LGBMClassifier()
model8 = lightgbm.LGBMClassifier()
model9 = lightgbm.LGBMClassifier()
model10 = lightgbm.LGBMClassifier()

model6.fit(X_train_dataset1, y_train_dataset1)
model7.fit(X_train_dataset2, y_train_dataset2)
model8.fit(X_train_dataset3, y_train_dataset3)
model9.fit(X_train_dataset4, y_train_dataset4)
model10.fit(X_train_dataset5, y_train_dataset5)
```

<pre>
LGBMClassifier()
</pre>

```python
model11 = catboost.CatBoostClassifier()
model12 = catboost.CatBoostClassifier()
model13 = catboost.CatBoostClassifier()
model14 = catboost.CatBoostClassifier()
model15 = catboost.CatBoostClassifier()

model11.fit(X_train_dataset1, y_train_dataset1)
model12.fit(X_train_dataset2, y_train_dataset2)
model13.fit(X_train_dataset3, y_train_dataset3)
model14.fit(X_train_dataset4, y_train_dataset4)
model15.fit(X_train_dataset5, y_train_dataset5)
```

<pre>
Learning rate set to 0.06692
0:	learn: 0.6759286	total: 66.1ms	remaining: 1m 6s
1:	learn: 0.6610865	total: 77.6ms	remaining: 38.7s
2:	learn: 0.6486346	total: 88.8ms	remaining: 29.5s
3:	learn: 0.6368109	total: 99.2ms	remaining: 24.7s
4:	learn: 0.6277035	total: 108ms	remaining: 21.6s
5:	learn: 0.6184388	total: 120ms	remaining: 19.8s
6:	learn: 0.6111359	total: 130ms	remaining: 18.5s
7:	learn: 0.6042963	total: 150ms	remaining: 18.7s
8:	learn: 0.5985637	total: 163ms	remaining: 18s
9:	learn: 0.5937757	total: 177ms	remaining: 17.5s
991:	learn: 0.4435676	total: 17.9s	remaining: 144ms
992:	learn: 0.4435011	total: 17.9s	remaining: 126ms
993:	learn: 0.4434435	total: 18s	remaining: 108ms
994:	learn: 0.4433911	total: 18s	remaining: 90.3ms
995:	learn: 0.4433295	total: 18s	remaining: 72.2ms
996:	learn: 0.4432811	total: 18s	remaining: 54.2ms
997:	learn: 0.4432344	total: 18s	remaining: 36.1ms
998:	learn: 0.4431728	total: 18.1s	remaining: 18.1ms
999:	learn: 0.4431009	total: 18.1s	remaining: 0us
Learning rate set to 0.06692
0:	learn: 0.6751357	total: 16.8ms	remaining: 16.8s
1:	learn: 0.6608088	total: 27.9ms	remaining: 13.9s
2:	learn: 0.6477290	total: 40.3ms	remaining: 13.4s
3:	learn: 0.6361654	total: 53.2ms	remaining: 13.3s
4:	learn: 0.6251300	total: 65.5ms	remaining: 13s
5:	learn: 0.6157266	total: 82.4ms	remaining: 13.7s
6:	learn: 0.6074684	total: 98ms	remaining: 13.9s
7:	learn: 0.6013841	total: 112ms	remaining: 13.9s
8:	learn: 0.5958427	total: 128ms	remaining: 14.1s
9:	learn: 0.5910737	total: 149ms	remaining: 14.7s
991:	learn: 0.4405092	total: 16.2s	remaining: 130ms
992:	learn: 0.4404866	total: 16.2s	remaining: 114ms
993:	learn: 0.4404314	total: 16.2s	remaining: 97.7ms
994:	learn: 0.4404005	total: 16.2s	remaining: 81.4ms
995:	learn: 0.4403511	total: 16.2s	remaining: 65.1ms
996:	learn: 0.4403181	total: 16.2s	remaining: 48.8ms
997:	learn: 0.4402590	total: 16.2s	remaining: 32.6ms
998:	learn: 0.4401813	total: 16.3s	remaining: 16.3ms
999:	learn: 0.4401328	total: 16.3s	remaining: 0us
Learning rate set to 0.06692
0:	learn: 0.6755781	total: 14.9ms	remaining: 14.9s
1:	learn: 0.6608919	total: 33.3ms	remaining: 16.6s
2:	learn: 0.6476158	total: 46.2ms	remaining: 15.4s
3:	learn: 0.6364682	total: 64.6ms	remaining: 16.1s
4:	learn: 0.6266676	total: 76.9ms	remaining: 15.3s
5:	learn: 0.6181876	total: 91.6ms	remaining: 15.2s
6:	learn: 0.6101398	total: 104ms	remaining: 14.8s
7:	learn: 0.6032476	total: 118ms	remaining: 14.6s
8:	learn: 0.5970564	total: 135ms	remaining: 14.9s
9:	learn: 0.5921347	total: 148ms	remaining: 14.7s
991:	learn: 0.4417088	total: 17.3s	remaining: 139ms
992:	learn: 0.4416633	total: 17.3s	remaining: 122ms
993:	learn: 0.4415901	total: 17.3s	remaining: 105ms
994:	learn: 0.4415245	total: 17.3s	remaining: 87.1ms
995:	learn: 0.4414774	total: 17.4s	remaining: 69.7ms
996:	learn: 0.4414339	total: 17.4s	remaining: 52.3ms
997:	learn: 0.4414069	total: 17.4s	remaining: 34.9ms
998:	learn: 0.4413760	total: 17.4s	remaining: 17.4ms
999:	learn: 0.4413206	total: 17.5s	remaining: 0us
Learning rate set to 0.06692
0:	learn: 0.6759031	total: 17.7ms	remaining: 17.7s
1:	learn: 0.6600829	total: 36ms	remaining: 18s
2:	learn: 0.6472842	total: 54.7ms	remaining: 18.2s
3:	learn: 0.6368099	total: 69.3ms	remaining: 17.3s
4:	learn: 0.6275614	total: 85.9ms	remaining: 17.1s
5:	learn: 0.6191886	total: 103ms	remaining: 17s
6:	learn: 0.6113553	total: 118ms	remaining: 16.8s
7:	learn: 0.6043434	total: 133ms	remaining: 16.5s
8:	learn: 0.5986823	total: 155ms	remaining: 17.1s
9:	learn: 0.5940700	total: 184ms	remaining: 18.2s
991:	learn: 0.4424352	total: 20.3s	remaining: 164ms
992:	learn: 0.4424006	total: 20.3s	remaining: 143ms
993:	learn: 0.4423326	total: 20.4s	remaining: 123ms
994:	learn: 0.4422714	total: 20.4s	remaining: 102ms
995:	learn: 0.4422291	total: 20.4s	remaining: 81.8ms
996:	learn: 0.4421637	total: 20.4s	remaining: 61.4ms
997:	learn: 0.4421127	total: 20.4s	remaining: 40.9ms
998:	learn: 0.4420725	total: 20.4s	remaining: 20.4ms
999:	learn: 0.4420401	total: 20.4s	remaining: 0us
Learning rate set to 0.06692
0:	learn: 0.6754843	total: 15.9ms	remaining: 15.9s
1:	learn: 0.6597966	total: 27.4ms	remaining: 13.7s
2:	learn: 0.6469985	total: 40.2ms	remaining: 13.4s
3:	learn: 0.6359868	total: 53.8ms	remaining: 13.4s
4:	learn: 0.6266373	total: 66.6ms	remaining: 13.3s
5:	learn: 0.6176368	total: 79.1ms	remaining: 13.1s
6:	learn: 0.6098391	total: 90.6ms	remaining: 12.9s
7:	learn: 0.6026141	total: 101ms	remaining: 12.6s
8:	learn: 0.5968272	total: 121ms	remaining: 13.3s
9:	learn: 0.5918948	total: 133ms	remaining: 13.1s
991:	learn: 0.4425485	total: 18.9s	remaining: 152ms
992:	learn: 0.4424977	total: 18.9s	remaining: 133ms
993:	learn: 0.4424367	total: 18.9s	remaining: 114ms
994:	learn: 0.4423775	total: 18.9s	remaining: 95.2ms
995:	learn: 0.4423239	total: 19s	remaining: 76.1ms
996:	learn: 0.4422579	total: 19s	remaining: 57.1ms
997:	learn: 0.4422072	total: 19s	remaining: 38ms
998:	learn: 0.4421429	total: 19s	remaining: 19ms
999:	learn: 0.4420807	total: 19s	remaining: 0us
</pre>
<pre>
<catboost.core.CatBoostClassifier at 0x7fc76e585a60>
</pre>

```python
y_prob1 = model1.predict_proba(X_train5)
y_prob2 = model2.predict_proba(X_train4)
y_prob3 = model3.predict_proba(X_train3)
y_prob4 = model4.predict_proba(X_train2)
y_prob5 = model5.predict_proba(X_train1)

y_prob6 = model6.predict_proba(X_train5)
y_prob7 = model7.predict_proba(X_train4)
y_prob8 = model8.predict_proba(X_train3)
y_prob9 = model9.predict_proba(X_train2)
y_prob10 = model10.predict_proba(X_train1)

y_prob11 = model11.predict_proba(X_train5)
y_prob12 = model12.predict_proba(X_train4)
y_prob13 = model13.predict_proba(X_train3)
y_prob14 = model14.predict_proba(X_train2)
y_prob15 = model15.predict_proba(X_train1)
```


```python
# thr 0.3보다 큰 경우
thr = 0.35
print(f"thr > {thr}")
pred1 = (y_prob1[:,1] >= thr).astype(np.int64)
pred2 = (y_prob2[:,1] >= thr).astype(np.int64)
pred3 = (y_prob3[:,1] >= thr).astype(np.int64)
pred4 = (y_prob4[:,1] >= thr).astype(np.int64)
pred5 = (y_prob5[:,1] >= thr).astype(np.int64)
pred6 = (y_prob6[:,1] >= thr).astype(np.int64)
pred7 = (y_prob7[:,1] >= thr).astype(np.int64)
pred8 = (y_prob8[:,1] >= thr).astype(np.int64)
pred9 = (y_prob9[:,1] >= thr).astype(np.int64)
pred10 = (y_prob10[:,1] >= thr).astype(np.int64)
pred11 = (y_prob11[:,1] >= thr).astype(np.int64)
pred12 = (y_prob12[:,1] >= thr).astype(np.int64)
pred13 = (y_prob13[:,1] >= thr).astype(np.int64)
pred14 = (y_prob14[:,1] >= thr).astype(np.int64)
pred15 = (y_prob15[:,1] >= thr).astype(np.int64)

print(f" model1의 예측비율 : {pred1.sum() / len(pred1)}")
print(f" model2의 예측비율 : {pred2.sum() / len(pred2)}")
print(f" model3의 예측비율 : {pred3.sum() / len(pred3)}")
print(f" model4의 예측비율 : {pred4.sum() / len(pred4)}")
print(f" model5의 예측비율 : {pred5.sum() / len(pred5)}")
print(f" model6의 예측비율 : {pred6.sum() / len(pred6)}")
print(f" model7의 예측비율 : {pred7.sum() / len(pred7)}")
print(f" model8의 예측비율 : {pred8.sum() / len(pred8)}")
print(f" model9의 예측비율 : {pred9.sum() / len(pred9)}")
print(f" model10의 예측비율 : {pred10.sum() / len(pred10)}")
print(f" model11의 예측비율 : {pred11.sum() / len(pred11)}")
print(f" model12의 예측비율 : {pred12.sum() / len(pred12)}")
print(f" model13의 예측비율 : {pred13.sum() / len(pred13)}")
print(f" model14의 예측비율 : {pred14.sum() / len(pred14)}")
print(f" model15의 예측비율 : {pred15.sum() / len(pred15)}")
```

<pre>
thr > 0.35
 model1의 예측비율 : 0.37745
 model2의 예측비율 : 0.38155
 model3의 예측비율 : 0.3768
 model4의 예측비율 : 0.3799
 model5의 예측비율 : 0.37465
 model6의 예측비율 : 0.37305
 model7의 예측비율 : 0.3825
 model8의 예측비율 : 0.3756
 model9의 예측비율 : 0.3755
 model10의 예측비율 : 0.3704
 model11의 예측비율 : 0.376
 model12의 예측비율 : 0.38415
 model13의 예측비율 : 0.37595
 model14의 예측비율 : 0.37735
 model15의 예측비율 : 0.37725
</pre>
<h2>제출하기전 threshold 값을 적절히 조절하여, acu와 f1값이 높아지게 해주면 됩니다.<br/>

아래와 같이 함수로 작성해주어도 됩니다.<br/>

저는 함수를 사용하지 않았습니다!!</h2>



```python
def calc_score_model(model, name, X_train, y_train, X_val, y_val):
    model1 = model
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_val)
    y_prob1 = model1.predict_proba(X_val)
    thr_result = 0.35
    max_val = 0.0
    scale = 1000
    for thr in range(1, scale):
        val = calc_sum_f1_and_accuracy(y_val, (y_prob1[:,1] >= thr / scale))
        if val > max_val:
            thr_result = thr / scale
            max_val = val
    
    return [name, thr_result, max_val, model]
    
# function test
print(calc_score_model(XGBClassifier(), 'XGB', X_train_dataset1, y_train_dataset1, X_train5, y_train5))
```


```python
def get_clf_eval(y_actual, y_pred, model):
    # accuracy = accuracy_score(y_actual, y_pred)
    # precision = precision_score(y_actual, y_pred)
    # recall = recall_score(y_actual, y_pred)
    AUC = roc_auc_score(y_actual, y_pred)
    F1 = f1_score(y_actual, y_pred)
    # print('\n정확도: {:.4f}'.format(accuracy))
    # print('정밀도: {:.4f}'.format(precision))
    # print('재현율: {:.4f}'.format(recall))
    print(model)
    print('AUC: {:.4f}'.format(AUC))
    print('F1: {:.4f}'.format(F1))
    print()
```


```python
# xgboost 성능 확인
get_clf_eval(y_train5, pred1, "model 1")
get_clf_eval(y_train4, pred2, "model 2")
get_clf_eval(y_train3, pred3, "model 3")
get_clf_eval(y_train2, pred4, "model 4")
get_clf_eval(y_train1, pred5, "model 5")
```

<pre>
model 1
AUC: 0.7240
F1: 0.6246

model 2
AUC: 0.7208
F1: 0.6301

model 3
AUC: 0.7230
F1: 0.6305

model 4
AUC: 0.7215
F1: 0.6245

model 5
AUC: 0.7156
F1: 0.6188

</pre>

```python
get_clf_eval(y_train5, pred6, "model 6")
get_clf_eval(y_train4, pred7, "model 7")
get_clf_eval(y_train3, pred8, "model 8")
get_clf_eval(y_train2, pred9, "model 9")
get_clf_eval(y_train1, pred10, "model 10")
```

<pre>
model 6
AUC: 0.7262
F1: 0.6273

model 7
AUC: 0.7223
F1: 0.6320

model 8
AUC: 0.7249
F1: 0.6329

model 9
AUC: 0.7244
F1: 0.6280

model 10
AUC: 0.7223
F1: 0.6272

</pre>

```python
get_clf_eval(y_train5, pred11, "model 11")
get_clf_eval(y_train4, pred12, "model 12")
get_clf_eval(y_train3, pred13, "model 13")
get_clf_eval(y_train2, pred14, "model 14")
get_clf_eval(y_train1, pred15, "model 15")
```

<pre>
model 11
AUC: 0.7296
F1: 0.6317

model 12
AUC: 0.7275
F1: 0.6386

model 13
AUC: 0.7277
F1: 0.6364

model 14
AUC: 0.7266
F1: 0.6308

model 15
AUC: 0.7257
F1: 0.6316

</pre>

```python
# 제출 양식 다운로드
submit = pd.read_csv('./data/sample_submission.csv')

# prediction 수행
df_test = pd.read_csv('./data/test.csv')
```


```python
test = df_test.drop('ID', axis=1)
```


```python
prob1 = model1.predict_proba(test)
out1 = (prob1[:,1] >= thr).astype(np.int64)
prob2 = model2.predict_proba(test)
out2 = (prob2[:,1] >= thr).astype(np.int64)
prob3 = model3.predict_proba(test)
out3 = (prob3[:,1] >= thr).astype(np.int64)
prob4 = model4.predict_proba(test)
out4 = (prob4[:,1] >= thr).astype(np.int64)
prob5 = model5.predict_proba(test)
out5 = (prob5[:,1] >= thr).astype(np.int64)

prob6 = model6.predict_proba(test)
out6 = (prob6[:,1] >= thr).astype(np.int64)
prob7 = model7.predict_proba(test)
out7 = (prob7[:,1] >= thr).astype(np.int64)
prob8 = model8.predict_proba(test)
out8 = (prob8[:,1] >= thr).astype(np.int64)
prob9 = model9.predict_proba(test)
out9 = (prob9[:,1] >= thr).astype(np.int64)
prob10 = model10.predict_proba(test)
out10 = (prob10[:,1] >= thr).astype(np.int64)

prob11 = model11.predict_proba(test)
out11 = (prob11[:,1] >= thr).astype(np.int64)
prob12 = model12.predict_proba(test)
out12 = (prob12[:,1] >= thr).astype(np.int64)
prob13 = model13.predict_proba(test)
out13 = (prob13[:,1] >= thr).astype(np.int64)
prob14 = model14.predict_proba(test)
out14 = (prob14[:,1] >= thr).astype(np.int64)
prob15 = model15.predict_proba(test)
out15 = (prob15[:,1] >= thr).astype(np.int64)

print(out1.sum() / len(out1))
print(out2.sum() / len(out2))
print(out3.sum() / len(out3))
print(out4.sum() / len(out4))
print(out5.sum() / len(out5))
print(out6.sum() / len(out6))
print(out7.sum() / len(out7))
print(out8.sum() / len(out8))
print(out9.sum() / len(out9))
print(out10.sum() / len(out10))
print(out11.sum() / len(out11))
print(out12.sum() / len(out12))
print(out13.sum() / len(out13))
print(out14.sum() / len(out14))
print(out15.sum() / len(out15))
```

<pre>
0.3856097833370561
0.3853305785123967
0.385637703819522
0.38658700022336384
0.3859169086441814
0.38128210855483585
0.38172883627429083
0.381198347107438
0.38175675675675674
0.378992628992629
0.3858052267143176
0.3860285905740451
0.3832365423274514
0.3866428411882957
0.38418583873129325
</pre>

```python
result = []
for i in range(len(test)):
  temp = 0
  for j in range(out_list.shape[0]):
    temp += out_list[j][i]
  if 6 <= temp:
    result.append(1)
  else:
    result.append(0)
```


```python
print(f" 최종 결과물!!! ")
print(f" {sum(result) / len(result)} ")
print(len(result))
print(result.count(1))

submit["answer"] = result  
```

<pre>
 최종 결과물!!! 
 0.40037971856153676 
35816
14340
</pre>

```python
# 제출 파일 저장
submit.to_csv('./submit/submit.csv', index=False)
```
