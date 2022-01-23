---
layout: single
title:  "[study] tutorial Dataset, DataLoader"
categories: study
tag: [pytorch, python]
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
import numpy as np
```


```python
data = [[1,2], [3,4]]
t_data = torch.tensor(data)
```


```python
t_data
```

<pre>
tensor([[1, 2],
        [3, 4]])
</pre>

```python
t_data.dtype
```

<pre>
torch.int64
</pre>

```python
t_data.size()
```

<pre>
torch.Size([2, 2])
</pre>

```python

```


```python
np_array = np.array(data)
t_np = torch.from_numpy(np_array)
```


```python
t_np
```

<pre>
tensor([[1, 2],
        [3, 4]])
</pre>

```python
t_np.size()
```

<pre>
torch.Size([2, 2])
</pre>

```python
t_np.shape
```

<pre>
torch.Size([2, 2])
</pre>

```python
x_ones = torch.ones_like(t_data)
x_ones
```

<pre>
tensor([[1, 1],
        [1, 1]])
</pre>

```python
x_rand = torch.rand_like(t_data, dtype=torch.float)
x_rand
```

<pre>
tensor([[0.4474, 0.7949],
        [0.4205, 0.8668]])
</pre>

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)
```

<pre>
tensor([[0.8367, 0.4293, 0.0447],
        [0.2943, 0.1218, 0.8405]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
</pre>

```python
tensor = torch.rand(3,4)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
```

<pre>
torch.Size([3, 4])
torch.float32
cpu
</pre>

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```


```python
device
```

<pre>
'cpu'
</pre>

```python
tensor = torch.ones(4,4)
print(tensor)
print("First row", tensor[0])
print("First column", tensor[:,0])
print("Last Column", tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

<pre>
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
First row tensor([1., 1., 1., 1.])
First column tensor([1., 1., 1., 1.])
Last Column tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
</pre>

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

<pre>
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
</pre>

```python
t2 = torch.cat([tensor,tensor,tensor], dim=0)
t2
```

<pre>
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
</pre>

```python
t2.shape
```

<pre>
torch.Size([12, 4])
</pre>

```python
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
print(y3)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)
```

<pre>
tensor([[0.3516, 0.7082, 0.2357, 0.0189],
        [0.5164, 0.8764, 0.1845, 0.9110],
        [0.4364, 0.2171, 0.0746, 0.9275],
        [0.3004, 0.5628, 0.6385, 0.0230]])
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
</pre>

```python
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)
```

<pre>
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
</pre>

```python
agg = tensor.sum()

agg = agg.type(torch.int32)
agg.item()
```

<pre>
12
</pre>

```python
tensor = tensor.sub(1)
tensor
```

<pre>
tensor([[ 0., -1.,  0.,  0.],
        [ 0., -1.,  0.,  0.],
        [ 0., -1.,  0.,  0.],
        [ 0., -1.,  0.,  0.]])
</pre>

```python
tensor = tensor.add(2)
tensor
```

<pre>
tensor([[2., 1., 2., 2.],
        [2., 1., 2., 2.],
        [2., 1., 2., 2.],
        [2., 1., 2., 2.]])
</pre>

```python
t = torch.ones(5)
print(t)
print(type(t))
n = t.numpy()
print(n)
print(type(n))
at = torch.from_numpy(n)
print(at)
print(type(at))
```

<pre>
tensor([1., 1., 1., 1., 1.])
<class 'torch.Tensor'>
[1. 1. 1. 1. 1.]
<class 'numpy.ndarray'>
tensor([1., 1., 1., 1., 1.])
<class 'torch.Tensor'>
</pre>

```python
t.add_(1)
print(t)
print(n)
print(at)
```

<pre>
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.])
</pre>

```python
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```


```python
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```

<pre>
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
</pre>
<pre>
  0%|          | 0/26421880 [00:00<?, ?it/s]
</pre>
<pre>
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
</pre>
<pre>
  0%|          | 0/29515 [00:00<?, ?it/s]
</pre>
<pre>
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
</pre>
<pre>
  0%|          | 0/4422102 [00:00<?, ?it/s]
</pre>
<pre>
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
</pre>
<pre>
  0%|          | 0/5148 [00:00<?, ?it/s]
</pre>
<pre>
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw

</pre>

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(9,9))
cols, rows = 4,4
for i in range(1, cols*rows + 1):
  sample_idx = torch.randint(len(train_data), size=(1,)).item()
  img, label = train_data[sample_idx]
  figure.add_subplot(rows,cols, i)
  plt.title(labels_map[label])
  print(img.shape, labels_map[label])
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

<pre>
torch.Size([1, 28, 28]) Sneaker
torch.Size([1, 28, 28]) T-Shirt
torch.Size([1, 28, 28]) Shirt
torch.Size([1, 28, 28]) Bag
torch.Size([1, 28, 28]) Pullover
torch.Size([1, 28, 28]) Bag
torch.Size([1, 28, 28]) Trouser
torch.Size([1, 28, 28]) Trouser
torch.Size([1, 28, 28]) T-Shirt
torch.Size([1, 28, 28]) Shirt
torch.Size([1, 28, 28]) Ankle Boot
torch.Size([1, 28, 28]) Coat
torch.Size([1, 28, 28]) Pullover
torch.Size([1, 28, 28]) Sandal
torch.Size([1, 28, 28]) Shirt
torch.Size([1, 28, 28]) Shirt
</pre>
<img src="data:image/png;base64,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"/>


```python
sample_idx = torch.randint(len(train_data), size=(1,)).item()
sample_idx
```

<pre>
10819
</pre>

```python
len(train_data)
```

<pre>
60000
</pre>

```python

train_data[sample_idx]
```

<pre>
(tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
           0.0000, 0.0000, 0.0000, 0.0000, 0.8078, 0.8314, 0.8118, 1.0000,
           0.1529, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0392, 0.0000, 0.2588, 0.7255, 0.9804, 1.0000, 0.8549,
           0.4824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0627,
           0.1059, 0.0000, 0.0000, 0.6000, 0.5608, 0.8196, 0.9333, 0.6667,
           0.7333, 0.3922, 0.0000, 0.0588, 0.0431, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.0980,
           0.0000, 0.0353, 0.8667, 0.9098, 0.8196, 0.7098, 0.6000, 0.8039,
           0.7961, 0.9647, 0.5647, 0.0000, 0.0863, 0.0745, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0235, 0.0784, 0.0000,
           0.0000, 0.8000, 0.8667, 0.7059, 0.7569, 0.7373, 0.6667, 0.7765,
           0.7216, 0.7294, 0.9686, 0.3608, 0.0000, 0.1059, 0.0314, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0588, 0.0863, 0.0000,
           0.3686, 0.9333, 0.7059, 0.7490, 0.7490, 0.7294, 0.6627, 0.7804,
           0.7725, 0.7490, 0.8196, 0.8706, 0.0000, 0.0314, 0.0863, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0510, 0.0824, 0.0000,
           0.6824, 0.7686, 0.7216, 0.7412, 0.7529, 0.7216, 0.6784, 0.7686,
           0.5882, 0.5608, 0.5098, 0.8824, 0.2000, 0.0000, 0.1176, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0549, 0.0510, 0.0196,
           0.8431, 0.7686, 0.7294, 0.7412, 0.7490, 0.7176, 0.6902, 0.7725,
           0.7255, 0.7804, 0.7216, 0.8824, 0.3490, 0.0000, 0.1059, 0.0039,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.0275, 0.1451,
           0.8235, 0.7804, 0.7569, 0.7529, 0.7647, 0.7333, 0.6941, 0.7725,
           0.7608, 0.7686, 0.7255, 0.9059, 0.2941, 0.0000, 0.0902, 0.0078,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0902, 0.0196, 0.0980,
           0.7216, 0.7843, 0.7608, 0.7529, 0.7569, 0.7451, 0.7098, 0.7765,
           0.7608, 0.8000, 0.7216, 0.8784, 0.2196, 0.0039, 0.0784, 0.0275,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0039, 0.0824, 0.0235, 0.1373,
           0.7255, 0.7608, 0.7608, 0.7529, 0.7569, 0.7451, 0.7137, 0.7765,
           0.7647, 0.7725, 0.7294, 0.8431, 0.2824, 0.0000, 0.0863, 0.0353,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.0627, 0.1725,
           0.8431, 0.7294, 0.7608, 0.7529, 0.7529, 0.7451, 0.7176, 0.7765,
           0.7765, 0.7725, 0.7294, 0.8196, 0.4078, 0.0000, 0.1059, 0.0549,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1098, 0.0941, 0.1686,
           0.7529, 0.7725, 0.7569, 0.7569, 0.7569, 0.7490, 0.7176, 0.7843,
           0.7725, 0.7843, 0.7216, 0.7843, 0.5176, 0.0000, 0.1373, 0.0667,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1098, 0.0980, 0.2902,
           0.4706, 0.8078, 0.7529, 0.7569, 0.7569, 0.7529, 0.7216, 0.7804,
           0.7569, 0.7961, 0.7294, 0.7529, 0.7059, 0.0000, 0.1961, 0.0431,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0157, 0.0784, 0.1608, 0.3216,
           0.5176, 0.7804, 0.7647, 0.7529, 0.7608, 0.7647, 0.7294, 0.7725,
           0.7647, 0.7725, 0.7686, 0.7882, 0.6745, 0.0235, 0.2588, 0.0118,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0235, 0.0118, 0.2431, 0.2588,
           0.3255, 0.8196, 0.7569, 0.7529, 0.7608, 0.7725, 0.7255, 0.7647,
           0.7725, 0.7569, 0.7961, 0.7451, 0.3373, 0.1961, 0.2039, 0.0078,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0353, 0.0157, 0.1608, 0.4235,
           0.2078, 0.8745, 0.7412, 0.7529, 0.7608, 0.7686, 0.7059, 0.7686,
           0.7608, 0.7647, 0.8039, 0.6980, 0.3725, 0.3490, 0.1216, 0.0235,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0353, 0.0275, 0.0588, 0.6078,
           0.2902, 0.8549, 0.7412, 0.7569, 0.7608, 0.7647, 0.7098, 0.7804,
           0.7529, 0.7686, 0.8157, 0.6784, 0.3725, 0.4353, 0.0706, 0.0196,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0353, 0.0353, 0.0745, 0.5333,
           0.2902, 0.8588, 0.7451, 0.7529, 0.7608, 0.7608, 0.7098, 0.7843,
           0.7569, 0.7490, 0.8196, 0.6863, 0.3059, 0.5137, 0.0235, 0.0275,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0275, 0.0157, 0.1490, 0.3922,
           0.2824, 0.8784, 0.7373, 0.7569, 0.7569, 0.7608, 0.7020, 0.7804,
           0.7569, 0.7490, 0.8078, 0.6941, 0.2667, 0.5725, 0.0000, 0.0235,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0078, 0.2392, 0.2980,
           0.3216, 0.8980, 0.7412, 0.7686, 0.7686, 0.7725, 0.7020, 0.7843,
           0.7569, 0.7569, 0.7882, 0.7529, 0.2824, 0.5922, 0.0000, 0.0314,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0275, 0.0000, 0.3412, 0.2588,
           0.3569, 0.8784, 0.7255, 0.7529, 0.7569, 0.7608, 0.7020, 0.7686,
           0.7373, 0.7451, 0.7569, 0.7804, 0.2667, 0.5725, 0.0000, 0.0431,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0471, 0.0000, 0.4000, 0.1922,
           0.3843, 0.9294, 0.7647, 0.7804, 0.7686, 0.7608, 0.6902, 0.7765,
           0.7451, 0.7804, 0.8118, 0.8745, 0.2157, 0.5647, 0.0039, 0.0431,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2588, 0.1059,
           0.3294, 0.8235, 0.7451, 0.7608, 0.7569, 0.7882, 0.7529, 0.7725,
           0.7608, 0.7529, 0.7451, 0.8549, 0.1373, 0.4824, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.1333, 0.3255, 0.6196, 0.0392,
           0.1647, 0.7647, 0.5608, 0.6784, 0.8706, 0.8549, 0.7412, 0.9294,
           0.8902, 0.6235, 0.6078, 0.6235, 0.0039, 0.6078, 0.2196, 0.2235,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.3882, 0.7922, 0.8549, 0.1569,
           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0078, 0.0000, 0.0000, 0.9765, 0.8706, 0.7765,
           0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.1098, 0.4941, 0.5098, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4235, 0.5569, 0.2627,
           0.0000, 0.0000, 0.0000, 0.0000]]]), 4)
</pre>

```python
import os
import pandas as pd
from torchvision.io import read_image

class myDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file, names=["file_name", "label"])
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
  
  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_trasnform:
      label = self.target_trasnform(label)
    return image, label
```


```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

 일반적으로 샘플들을 “미니배치(minibatch)”로 전달하고, 매 에폭(epoch)마다 데이터를 다시 섞어서 과적합(overfit)을 막고, Python의 multiprocessing 을 사용하여 데이터 검색 속도를 높이려고 합니다.



```python
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

<pre>
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
</pre>
<img src="data:image/png;base64,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"/>

<pre>
Label: 9
</pre>

```python

```
