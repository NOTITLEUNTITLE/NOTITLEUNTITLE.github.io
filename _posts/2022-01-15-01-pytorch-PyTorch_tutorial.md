---
layout: single
title:  "[PyTorch] PyTorch Tutorial"
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


# PyTorch Tutorial





---



연습하는 페이지.



```python
import numpy as np
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
```


```python
weights = torch.ones(4, requires_grad=True)
```


```python
num = 10
```


```python
for epoch in range(num):
  # weights.grad.zero_()
  model_output = (weights*3).sum()
  model_output.backward()
  print(weights.grad)
  # weights.grad.zero_()
```

<pre>
tensor([6., 6., 6., 6.])
tensor([9., 9., 9., 9.])
tensor([12., 12., 12., 12.])
tensor([15., 15., 15., 15.])
tensor([18., 18., 18., 18.])
tensor([21., 21., 21., 21.])
tensor([24., 24., 24., 24.])
tensor([27., 27., 27., 27.])
tensor([30., 30., 30., 30.])
tensor([33., 33., 33., 33.])
</pre>

```python
weights.grad
```

<pre>
tensor([33., 33., 33., 33.])
</pre>

```python
# Backpropagation
```

# Backpropagation






```python
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_pred = w * x
loss = (y_pred - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)

```

<pre>
tensor(1., grad_fn=<PowBackward0>)
tensor(-2.)
</pre>

```python

```

-Prediction: PyTorch Model   

-Gradients Computation: Autograd   

-Loss Computation: PyTorch Loss   

-Parameter updates: PyTorch Optimizer   





# Numpy를 활용한 Machine Learning



```python
# f = 2 * x
x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# Model Prediction
def forward(x):
  return w *x 

# loss = MSE
def loss(y, y_pred):
  return ((y_pred - y) ** 2).mean()

# Gradient
# MSE = 1/N * (w*x)**2
# dJ/dw = 1/N * 2x * (w*x - y)
def gradient(x,y,y_pred):
  return np.dot(2*x, y_pred-y).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f} ")

# Training
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
  # Prediction = forward pass
  y_pred = forward(x)

  # Loss
  l = loss(y, y_pred)

  # Gradients
  dw = gradient(x,y,y_pred)

  # Update weights
  w -= learning_rate * dw

  if epoch % 1 == 0:
    print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f} ")


```

<pre>
Prediction before training: f(5) = 0.000 
epoch 1: w = 1.200, loss = 30.00000000
epoch 2: w = 1.680, loss = 4.79999924
epoch 3: w = 1.872, loss = 0.76800019
epoch 4: w = 1.949, loss = 0.12288000
epoch 5: w = 1.980, loss = 0.01966083
epoch 6: w = 1.992, loss = 0.00314574
epoch 7: w = 1.997, loss = 0.00050331
epoch 8: w = 1.999, loss = 0.00008053
epoch 9: w = 1.999, loss = 0.00001288
epoch 10: w = 2.000, loss = 0.00000206
Prediction after training: f(5) = 9.999 
</pre>
# PyTorch를 활용한 Machine Learning( 미분기능)



```python
# f = 2 * x
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

# 초기값은 0으로 했다.
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model Prediction
def forward(x):
  return w *x 

# loss = MSE
def loss(y, y_pred):
  return ((y_pred - y) ** 2).mean()

# Gradient
# MSE = 1/N * (w*x)**2
# dJ/dw = 1/N * 2x * (w*x - y)
def gradient(x,y,y_pred):
  return np.dot(2*x, y_pred-y).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f} ")

# Training
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
  # Prediction = forward pass
  y_pred = forward(x)

  # Loss
  l = loss(y, y_pred)

  # Gradients = Backward pass
  l.backward()  # dl/dw

  # Update weights
  with torch.no_grad():
    w -= learning_rate * w.grad

  # zero gradients
  w.grad.zero_()


  if epoch % 5 == 0:
    print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f} ")


```

<pre>
Prediction before training: f(5) = 0.000 
epoch 1: w = 0.300, loss = 30.00000000
epoch 6: w = 1.246, loss = 5.90623236
epoch 11: w = 1.665, loss = 1.16278565
epoch 16: w = 1.851, loss = 0.22892261
epoch 21: w = 1.934, loss = 0.04506890
epoch 26: w = 1.971, loss = 0.00887291
epoch 31: w = 1.987, loss = 0.00174685
epoch 36: w = 1.994, loss = 0.00034392
epoch 41: w = 1.997, loss = 0.00006770
epoch 46: w = 1.999, loss = 0.00001333
epoch 51: w = 1.999, loss = 0.00000262
epoch 56: w = 2.000, loss = 0.00000052
epoch 61: w = 2.000, loss = 0.00000010
epoch 66: w = 2.000, loss = 0.00000002
epoch 71: w = 2.000, loss = 0.00000000
epoch 76: w = 2.000, loss = 0.00000000
epoch 81: w = 2.000, loss = 0.00000000
epoch 86: w = 2.000, loss = 0.00000000
epoch 91: w = 2.000, loss = 0.00000000
epoch 96: w = 2.000, loss = 0.00000000
Prediction after training: f(5) = 10.000 
</pre>

```python
test = torch.tensor([4, 16, 36, 64], dtype=torch.float32)
test.mean()
```

<pre>
tensor(30.)
</pre>
# PyTorch를 활용한 Machine Learning( cost function, update, initialize)



```python
# f = 2 * x
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

# 초기값은 0으로 했다.
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model Prediction
def forward(x):
  return w *x 


print(f"Prediction before training: f(5) = {forward(5):.3f} ")

# Training
learning_rate = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(epochs):
  # Prediction = forward pass
  y_pred = forward(x)

  # Loss
  l = loss(y, y_pred)

  # Gradients = Backward pass
  l.backward()  # dl/dw

  # Update weights
  optimizer.step()

  # zero gradients
  optimizer.zero_grad()


  if epoch % 5 == 0:
    print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f} ")


```

<pre>
Prediction before training: f(5) = 0.000 
epoch 1: w = 0.300, loss = 30.00000000
epoch 6: w = 1.246, loss = 5.90623236
epoch 11: w = 1.665, loss = 1.16278565
epoch 16: w = 1.851, loss = 0.22892261
epoch 21: w = 1.934, loss = 0.04506890
epoch 26: w = 1.971, loss = 0.00887291
epoch 31: w = 1.987, loss = 0.00174685
epoch 36: w = 1.994, loss = 0.00034392
epoch 41: w = 1.997, loss = 0.00006770
epoch 46: w = 1.999, loss = 0.00001333
epoch 51: w = 1.999, loss = 0.00000262
epoch 56: w = 2.000, loss = 0.00000052
epoch 61: w = 2.000, loss = 0.00000010
epoch 66: w = 2.000, loss = 0.00000002
epoch 71: w = 2.000, loss = 0.00000000
epoch 76: w = 2.000, loss = 0.00000000
epoch 81: w = 2.000, loss = 0.00000000
epoch 86: w = 2.000, loss = 0.00000000
epoch 91: w = 2.000, loss = 0.00000000
epoch 96: w = 2.000, loss = 0.00000000
Prediction after training: f(5) = 10.000 
</pre>
# PyTorch를 활용한 Machine Learning(모델선언)



```python
# # f = 2 * x
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
# x = torch.tensor([1,2,3,4], dtype=torch.float32)
# y = torch.tensor([2,4,6,8], dtype=torch.float32)


x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
input_size = n_features
output_size = n_features


model = nn.Linear(input_size, output_size)


print(f"Prediction before training: f(5) = {model(x_test).item():.3f} ")

# Training
learning_rate = 0.01
epochs = 101

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  # Prediction = forward pass
  y_pred = model(x)

  # Loss
  l = loss(y, y_pred)

  # Gradients = Backward pass
  l.backward()  # dl/dw

  # Update weights
  optimizer.step()

  # zero gradients
  optimizer.zero_grad()


  if epoch % 5 == 0:
    [w, b] = model.parameters()
    print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {model(x_test).item():.3f} ")


```

<pre>
Prediction before training: f(5) = -3.405 
epoch 1: w = -0.418, loss = 50.64212799
epoch 6: w = 0.749, loss = 8.41928101
epoch 11: w = 1.221, loss = 1.61969924
epoch 16: w = 1.414, loss = 0.51815248
epoch 21: w = 1.496, loss = 0.33335984
epoch 26: w = 1.532, loss = 0.29625094
epoch 31: w = 1.551, loss = 0.28311497
epoch 36: w = 1.562, loss = 0.27404681
epoch 41: w = 1.571, loss = 0.26583838
epoch 46: w = 1.578, loss = 0.25796744
epoch 51: w = 1.585, loss = 0.25034457
epoch 56: w = 1.591, loss = 0.24294901
epoch 61: w = 1.597, loss = 0.23577258
epoch 66: w = 1.603, loss = 0.22880816
epoch 71: w = 1.609, loss = 0.22204940
epoch 76: w = 1.615, loss = 0.21549028
epoch 81: w = 1.621, loss = 0.20912491
epoch 86: w = 1.626, loss = 0.20294756
epoch 91: w = 1.632, loss = 0.19695282
epoch 96: w = 1.637, loss = 0.19113493
epoch 101: w = 1.643, loss = 0.18548904
Prediction after training: f(5) = 9.264 
</pre>
# PyTorch를 활용한 Machine Learning(Customize Model)



```python
# f = 2 * x
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
# x = torch.tensor([1,2,3,4], dtype=torch.float32)
# y = torch.tensor([2,4,6,8], dtype=torch.float32)


x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
input_size = n_features
output_size = n_features




class LinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearRegression, self).__init__()
    # define layers
    self.lin = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.lin(x)


model = LinearRegression(input_size, output_size)


print(f"Prediction before training: f(5) = {model(x_test).item():.3f} ")

# Training
learning_rate = 0.01
epochs = 101

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  # Prediction = forward pass
  y_pred = model(x)

  # Loss
  l = loss(y, y_pred)

  # Gradients = Backward pass
  l.backward()  # dl/dw

  # Update weights
  optimizer.step()

  # zero gradients
  optimizer.zero_grad()


  if epoch % 5 == 0:
    [w, b] = model.parameters()
    print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {model(x_test).item():.3f} ")


```

<pre>
Prediction before training: f(5) = 2.496 
epoch 1: w = 0.724, loss = 16.88318253
epoch 6: w = 1.397, loss = 2.74425197
epoch 11: w = 1.668, loss = 0.46917462
epoch 16: w = 1.779, loss = 0.10241038
epoch 21: w = 1.824, loss = 0.04262101
epoch 26: w = 1.844, loss = 0.03223152
epoch 31: w = 1.853, loss = 0.02981082
epoch 36: w = 1.858, loss = 0.02869401
epoch 41: w = 1.861, loss = 0.02780840
epoch 46: w = 1.863, loss = 0.02698089
epoch 51: w = 1.866, loss = 0.02618290
epoch 56: w = 1.868, loss = 0.02540932
epoch 61: w = 1.870, loss = 0.02465874
epoch 66: w = 1.872, loss = 0.02393033
epoch 71: w = 1.874, loss = 0.02322345
epoch 76: w = 1.875, loss = 0.02253745
epoch 81: w = 1.877, loss = 0.02187172
epoch 86: w = 1.879, loss = 0.02122569
epoch 91: w = 1.881, loss = 0.02059869
epoch 96: w = 1.883, loss = 0.01999025
epoch 101: w = 1.884, loss = 0.01939976
Prediction after training: f(5) = 9.762 
</pre>

```python

```

# 마무리



## Design model (input_size, output_size, forward pass)

## Construct loss and optimizer

## Training loop   

### 1. forward pass : compute prediction and loss   

### 2. backward pass : Gradients   

### 3. update weights

    



```python

```
