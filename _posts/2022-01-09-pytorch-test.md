---
layout: single
title:  "Pytorch test-1"
categories: coding
tag: [python, pytorch]
toc: true
author_profile: false
---

# 데이터 전처리
---
PyTorch에서는 데이터 작업을 위한 기본요소 두가지인  
`torch.utils.data.DataLoader`와 `torch.utils.data.Dataset`가 있습니다.  
`Dataset`은  `Sample`과 `label`을 저장하고,  
`DataLoader`은 `Dataset`을 `iterable` 로 감쌉니다.


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# vision data set을 library를 통해서 다운로드 받는다.
# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)



batch_size = 64
# 다운된 data set을 dataloader의 parameter로 넘겨준다.
# 데이터로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

```

# 모델 만들기