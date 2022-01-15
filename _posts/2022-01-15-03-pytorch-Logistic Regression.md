---
layout: single
title:  "[PyTorch] Logistic Regression[미완성]"
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


# Logistic Regression





> sklearn libaray의 dataset을 기준으로 코드를 작성하고 연습을 진행하였다.   



Linear Regression이 예측 모델이라면,

Logistic Regression은 분류 모델이다.









큰 흐름을 말하자면, 



1.   Dataset and Data Preprocessing

2.   Model Declaration and Initialization.

3.   Cost function Declaration and Gradient(Optimizer) Declaration

4.   Training

5.   Visualization


# 아직 다듬을게 많다.



```python
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
```


```python
# Prepare data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
```


```python
dir(bc)
```

<pre>
['DESCR',
 'data',
 'data_module',
 'feature_names',
 'filename',
 'frame',
 'target',
 'target_names']
</pre>

```python
bc.data.shape
```

<pre>
(569, 30)
</pre>

```python
bc.feature_names
```

<pre>
array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'], dtype='<U23')
</pre>

```python
bc.target_names
```

<pre>
array(['malignant', 'benign'], dtype='<U9')
</pre>

```python
bc.target
```

<pre>
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
</pre>

```python

```


```python

```


```python
n_samples, n_features = x.shape
```


```python
n_samples, n_features
```

<pre>
(569, 30)
</pre>

```python
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
```


```python
x_train
```

<pre>
array([[9.029e+00, 1.733e+01, 5.879e+01, ..., 1.750e-01, 4.228e-01,
        1.175e-01],
       [2.109e+01, 2.657e+01, 1.427e+02, ..., 2.903e-01, 4.098e-01,
        1.284e-01],
       [9.173e+00, 1.386e+01, 5.920e+01, ..., 5.087e-02, 3.282e-01,
        8.490e-02],
       ...,
       [1.429e+01, 1.682e+01, 9.030e+01, ..., 3.333e-02, 2.458e-01,
        6.120e-02],
       [1.398e+01, 1.962e+01, 9.112e+01, ..., 1.827e-01, 3.179e-01,
        1.055e-01],
       [1.218e+01, 2.052e+01, 7.722e+01, ..., 7.431e-02, 2.694e-01,
        6.878e-02]])
</pre>

```python

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
```


```python
# model
# f = wx * b, sigmoid at the end

class LogisticRegression(nn.Module):
  def __init__(self, input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_features, 1)

  def forward(self, x):
    y_pred = torch.sigmoid(self.linear(x))
    return y_pred
  
model = LogisticRegression(n_features)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```


```python
# training loop

epochs = 100
for epoch in range(epochs):
  # forward pass and loss
  y_pred = model(x_train)
  loss = criterion(y_pred, y_train)

  # backward pass
  loss.backward()

  # update
  optimizer.step()

  optimizer.zero_grad()

  if (epoch + 1) % 10 == 0:
    print(f"epoch = {epoch+1} loss = {loss.item():.4f} ")


```

<pre>
epoch = 10 loss = 0.5061 
epoch = 20 loss = 0.4237 
epoch = 30 loss = 0.3709 
epoch = 40 loss = 0.3341 
epoch = 50 loss = 0.3070 
epoch = 60 loss = 0.2862 
epoch = 70 loss = 0.2695 
epoch = 80 loss = 0.2559 
epoch = 90 loss = 0.2445 
epoch = 100 loss = 0.2348 
</pre>

```python
with torch.no_grad():
  y_pred = model(x_test)
  y_pred_cls = y_pred.round()
  acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f"accuracy = {acc:.4f}")
```

<pre>
accuracy = 0.9561
</pre>

```python
y_test.shape, y_pred_cls.shape, y_pred.shape
```

<pre>
(torch.Size([114, 1]), torch.Size([114, 1]), torch.Size([114, 1]))
</pre>

```python
# plt.plot(y_pred_cls, y_test)
plt.plot(x_test, y_pred_cls, 'ro', alpha=0.3)
# plt.scatter(x_test, y_pred_cls, color='green', marker='o', markersize=10, alpha=0.5)
plt.plot(x_test, y_test, 'bo', alpha=0.1)

# plt.scatter(x=y_test, y=y_pred_cls, c='blueviolet', alpha=0.5)
# plt.plot(x_train, y_train, 'ro')
# plt.plot(x_train, y_train, 'bo')

# plt.plot(y_pred_cls, x_test, 'ro')
# plt.plot(y_test, x_test, 'bo')


plt.show()
```

<img src="data:image/png;base64,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"/>


```python

```
