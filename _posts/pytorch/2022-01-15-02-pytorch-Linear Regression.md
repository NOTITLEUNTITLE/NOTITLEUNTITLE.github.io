---
layout: single
title:  "[PyTorch] Linear Regression"
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


# Linear Regression (선형 회귀)





> sklearn libaray의 dataset을 기준으로 코드를 작성하고 연습을 진행하였다.   





큰 흐름을 말하자면, 



1.   Dataset and Data Preprocessing

2.   Model Declaration and Initialization.

3.   Cost function Declaration and Gradient(Optimizer) Declaration

4.   Training

5.   Visualization







# CODE




```python
# Declare the required library

import numpy as np                 # Data libaray
import matplotlib.pyplot as plt    # Visualization


import torch                       # torch libaray
import torch.nn as nn              # torch neural network libaray
from sklearn import datasets       # Dataset Load
```


```python
# Data Load
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
```


```python
x = torch.from_numpy(x_numpy.astype(np.float32))      # neural network에 올리기 위한 형변환.
y = torch.from_numpy(y_numpy.astype(np.float32))      # neural network에 올리기 위한 형변환.
y = y.view(y.shape[0],1)                              # tensor의 size를 100,1로 변경한다.

# 와... y_numpy에 x_numpy 넣어줘서 1시간걸려서 찾았네;;
# 생각해보면 y가 이상하다는거 알았으면 바로 여기부터 볼것이지....어이가 없네
```


```python
x.shape, y.shape
```

<pre>
(torch.Size([100, 1]), torch.Size([100, 1]))
</pre>

```python
# model
n_samples, n_features = x.shape        
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)      # model은 nn.Linear모델을 사용하였다. 지금은 내가 답을 알고 선택하기에 이렇지만, 차후에는 데이터의 특성을 파악하고 해야한다.
```


```python
# loss and optimizer

criterion = nn.MSELoss()                                        # cost fucntion은 MSE(Mean Square Error를 선택했다. 특정 값이상에서는 오차가 커지므로 학습하기 무난하다고 생가했는데. 사실 그냥 골랐디.)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)        # 최적화(optimizer)로는 SGD(Stochastic gradient descent)를 선택했는데 Adam을 하는것도 무난한데, Linear이고 처음이니깐 이거 했다. 별 이유는 없다.
```


```python
# training loop
epochs = 100
for epoch in range(epochs):
  # forward pass and loss
  y_pred = model(x)                      # 모델을 학습시켜서 예측값을 뱉어낸다.
  loss = criterion(y_pred, y)            # 정답과 학습시켜서 나온 예측값의 오차를 측정한다.

  # backward pass
  loss.backward()                        # 오차를 줄인다.

  # update
  optimizer.step()                       # 예측값을 업데이트한다.

  optimizer.zero_grad()                  # 다음학습에 영향을 받으면 안되니깐 다시 초기화.

  if (epoch+1) % 10 == 0:
    print(f"epoch: {epoch+1}, loss = {loss.item():.4f} ")
  
# plot
predicted = model(x).detach().numpy()
# plt.subplot(121)
plt.plot(x_numpy, y_numpy, 'ro')
# plt.subplot(122)
plt.plot(x_numpy, predicted, 'b')
plt.show()
```

<pre>
epoch: 10, loss = 4349.8877 
epoch: 20, loss = 3245.0125 
epoch: 30, loss = 2445.9539 
epoch: 40, loss = 1867.4298 
epoch: 50, loss = 1448.1460 
epoch: 60, loss = 1143.9843 
epoch: 70, loss = 923.1429 
epoch: 80, loss = 762.6686 
epoch: 90, loss = 645.9738 
epoch: 100, loss = 561.0574 
</pre>
<img src="data:image/png;base64,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"/>


```python

```
