---
layout: single
title:  "[PyTorch] Softmax and Cross Entropy"
categories: coding
tag: [python, pytorch]
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



```python
import torch
import torch.nn as nn
import numpy as np
```


```python
def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)
```


```python
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
outputs
```

<pre>
array([0.65900114, 0.24243297, 0.09856589])
</pre>

```python
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
outputs
```

<pre>
tensor([0.6590, 0.2424, 0.0986])
</pre>

```python
def cross_entropy(actual, predicted):
  loss = -np.sum(actual * np.log(predicted))
  return loss
```


```python
y = np.array([2,0,1])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y,y_pred_good)
l2 = cross_entropy(y,y_pred_bad)
print(f"loss 1 numpy : {l1:.4f}")
print(f"loss 2 numpy : {l2:.4f}")
```

<pre>
loss 1 numpy : 3.0159
loss 2 numpy : 5.1160
</pre>




```python

```


```python

```


```python

```


```python
loss = nn.CrossEntropyLoss()
```


```python
y = torch.tensor([2,0,1])
y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1] ,[0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 5.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(f"loss 1 numpy : {l1:.4f}")
print(f"loss 2 numpy : {l2:.4f}")
```

<pre>
loss 1 numpy : 0.3018
loss 2 numpy : 1.5943
</pre>

```python
_, pred1 = torch.max(y_pred_good, 1)
_, pred2 = torch.max(y_pred_bad, 1)
print(pred1, pred2)
```

<pre>
tensor([2, 0, 1]) tensor([0, 2, 1])
</pre>
{\color{Blue}x^2}+{\color{Red}2x}-{\color{Green}1}

\

\alpha

×



```python

```
