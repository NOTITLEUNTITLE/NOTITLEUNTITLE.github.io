---
layout: single
title:  "[PyTorch] Convolutional Neural Net"
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```


```python
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
#Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001
```


```python
# dataset has PILImage images of range[0, 1]
# We transform them to Tensors of normalized range[-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root = "/content/drive/MyDrive/Study/data", train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="/content/drive/MyDrive/Study/data", train=False,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
```

<pre>
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /content/drive/MyDrive/Study/data/cifar-10-python.tar.gz
</pre>
<pre>
  0%|          | 0/170498071 [00:00<?, ?it/s]
</pre>
<pre>
Extracting /content/drive/MyDrive/Study/data/cifar-10-python.tar.gz to /content/drive/MyDrive/Study/data
Files already downloaded and verified
</pre>

```python
# implement conv net
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3,6,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6,16,5)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
    
```


```python
model = ConvNet().to(device)
```


```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    # origin shape: [4, 3, 32, 32] = 4, 3, 1024
    # input_layer: 3 input channels, 6 output channels, 5 kernel size
    images = images.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 2000 == 0:
      print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f} ")

print("finished training")
```

<pre>
Epoch [1/4], Step [2000/12500], Loss: 2.2975 
Epoch [1/4], Step [4000/12500], Loss: 2.2932 
Epoch [1/4], Step [6000/12500], Loss: 2.2696 
Epoch [1/4], Step [8000/12500], Loss: 2.2891 
Epoch [1/4], Step [10000/12500], Loss: 2.2334 
Epoch [1/4], Step [12000/12500], Loss: 2.5101 
Epoch [2/4], Step [2000/12500], Loss: 2.2816 
Epoch [2/4], Step [4000/12500], Loss: 2.0126 
Epoch [2/4], Step [6000/12500], Loss: 1.7955 
Epoch [2/4], Step [8000/12500], Loss: 1.4909 
Epoch [2/4], Step [10000/12500], Loss: 1.9605 
Epoch [2/4], Step [12000/12500], Loss: 1.6538 
Epoch [3/4], Step [2000/12500], Loss: 1.3560 
Epoch [3/4], Step [4000/12500], Loss: 1.9046 
Epoch [3/4], Step [6000/12500], Loss: 1.2195 
Epoch [3/4], Step [8000/12500], Loss: 1.8773 
Epoch [3/4], Step [10000/12500], Loss: 1.3472 
Epoch [3/4], Step [12000/12500], Loss: 1.3043 
Epoch [4/4], Step [2000/12500], Loss: 1.5261 
Epoch [4/4], Step [4000/12500], Loss: 0.9752 
Epoch [4/4], Step [6000/12500], Loss: 0.6960 
Epoch [4/4], Step [8000/12500], Loss: 2.1438 
Epoch [4/4], Step [10000/12500], Loss: 2.0787 
Epoch [4/4], Step [12000/12500], Loss: 1.2792 
finished training
</pre>

```python
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  n_class_correct = [0 for i in range(10)]
  n_class_samples = [0 for i in range(10)]
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

    # max returns (value, index)
    _, prediction = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (prediction == labels).sum().item()
    
    for i in range(batch_size):
      label = labels[i]
      pred = prediction[i]
      if (label == pred):
        n_class_correct[label] += 1
      n_class_samples[label] += 1
  acc = 100 * n_correct / n_samples
  print(f"Accuracy of the network: {acc}%")

  for i in range(10):
    acc = 100 * n_class_correct[i] / n_class_samples[i]
    print(f"Accuracy of {classes[i]}: {acc}%")
```

<pre>
Accuracy of the network: 46.75%
Accuracy of plane: 48.2%
Accuracy of car: 67.8%
Accuracy of bird: 27.7%
Accuracy of cat: 31.2%
Accuracy of deer: 27.3%
Accuracy of dog: 33.2%
Accuracy of frog: 71.4%
Accuracy of horse: 60.9%
Accuracy of ship: 60.1%
Accuracy of truck: 39.7%
</pre>

```python

```


```python

```


```python

```


```python

```
