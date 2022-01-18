---
layout: single
title:  "[PyTorch] Forward Net"
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

```


```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
input_size = 784
hidden_size = 100
output_size = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
```


```python
train_dataset = torchvision.datasets.MNIST(root='/content/drive/MyDrive/Study/data', train=True, transform=transforms.ToTensor(), download=True)
```


```python
test_dataset = torchvision.datasets.MNIST(root="/content/drive/MyDrive/Study/data", train=False, transform=transforms.ToTensor())
```


```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```


```python
examples = iter(train_loader)
samples, labels = examples.next()
samples.shape, labels.shape
```

<pre>
(torch.Size([100, 1, 28, 28]), torch.Size([100]))
</pre>

```python
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.imshow(samples[i][0])
  # plt.imshow(samples[i][0], cmap="gray")  #흑백
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXAAAAD6CAYAAAC4RRw1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXhV1bkG8PfLSSAgMkuIDIZZsVVpuQoqaotWHCoOVaBWhqI44GyraNvb3t620lZRqzikBREvaqsoUmuvClcL4gQqRSCEgIggIYAiIHOSdf/guPf+NtknhzOvfd7f8/jwrbN2zl7mSxabddYgxhgQEZF9CrLdACIiSgw7cCIiS7EDJyKyFDtwIiJLsQMnIrIUO3AiIksl1YGLyBARqRSRVSIyIVWNouxiXsOLuQ0XSXQeuIhEAKwEcBaA9QAWAhhhjFmeuuZRpjGv4cXchk9hEl97IoBVxpiPAUBEngEwFEDgD0MTaWqKcVgSt6RU2IOd2Gf2SkA182qpRvIKHGJumdfcsQNbtxhjjvC/nkwH3gnAOk95PYCT/BeJyDgA4wCgGM1xkgxO4paUCu+aubGqmVdLNZJXII7cMq+5aY55bm1Dr6f9Q0xjTLkxpr8xpn8Rmqb7dpQhzGs4Ma92SaYD/wxAF0+5c/Q1shvzGl7Mbcgk04EvBNBLRLqJSBMAwwHMTk2zKIuY1/BibkMm4TFwY0ytiFwP4BUAEQBTjTHLUtYyygrmNbyY2/BJ5kNMGGNeBvByitpCOYJ5DS/mNly4EpOIyFLswImILMUOnIjIUuzAiYgsxQ6ciMhS7MCJiCyV1DRCIopt94UnOvG8h8tVXfdXxzpxr9HvZ6xNYVB/6glOvL9VUdxft254rSoXNqlz4rL2X6i6l4921zjN2tla1f364R85ccf734r7/qnGJ3AiIkuxAycishQ7cCIiS3EMnCiFCo/qospld6xw4v2mTtVNO22KE999wg9VXf1iHpJT0Ly5E/edv0fV3XLEQ05cGmmu6uqR2CljBdBnYXjf54LDtqq602/7oxMPxk9VXSbHxPkETkRkKXbgRESWsmIIZddF+kSv+ZMfC7x20Pirnbj5C+9mvC09/nqNE/e85Z2U359y26qrOqvyrK6zAq8t33iGExds/lLV1ae0VXbwDpkAwOfPdnLiiR2f8V3dLKF7rKnVQzFXrnCnA+55qqOqq/McSPSHO/QU0NOK3Xj3gK8Saksq8AmciMhS7MCJiCzFDpyIyFJWjIFvOE0av6iBa3u+kPq2dLu9Imb96mGPOnEPXKPqOCYeDpGe3VS57OlqJ55Zep/vaneZ93XrT1M1S5/p68Qln2VvOXbOiERU8cgW2514r9mv6pqK+30duHiYqmt3p/s+BZ9vV3WmVi+lb1azxo2xRtUVHHe0Ex/fRL8P4A6C79/eFNnCJ3AiIkuxAycispQVQygHDT0Ma/g6ADhlgLuCrSYNbZl+1Lw0vCvZ5LPzSlX5xSOf85SCd8Zb+D/Hq3LJgxw28arfsUOVd5/uli/43vX62kJ3qPSI5fo3vfaTT93rDuH+hWVdVXng/yx24lYFxaquYr87pHP0wzt12w7hnsniEzgRkaXYgRMRWYodOBGRpawYA/cbudadjuUfk/aWz8YJSDXvUnlATxv0847HA+kZk6f0iLRrq8oVv+vhxK8M+aPv6mIE6TPrOifutTB7S65tV/TqosC62sCaxnnHvS/+53uqbnTLDU68eJ/eSfKnV41327Y4e6cp8QmciMhSjXbgIjJVRDaJyFLPa21F5DURqYr+2Sa9zaRUY17Di7nNH/EMoUwD8BCA6Z7XJgCYa4yZKCITouU7Ut+8hi14x13Bhhye1ucf3knHkE4SpiHH8ppNW0cNVOXjr1uiyi929g6V6SGTr+r3OvHp9/9E1fV+wP1nuX8VYBpNA3ML4OAdDr+88DhVPm/CG07sHTIB9IEOYyfdrOpK5uTGFNBGn8CNMfMAfOF7eSiAJ6LxEwAuTHG7KM2Y1/BibvNHomPgJcaYrzeA2AigJEXtoexiXsOLuQ2hpD/ENMYYIPgQOhEZJyKLRGTRfuwNuoxyDPMaXrFyy7zaJdFphDUiUmqMqRaRUgCbgi40xpQDKAeAltI2sdNGE7TqvgGqzN0AG2VFXtPhnl8+osoDm9YFXAnU1O1W5cFPuIfalt2rx0Zz6BsTV25tzGvdGd9S5a193N0BB41bqOr+2HFyjHcK3vV0X6uEmpZ2iT6BzwYwKhqPAvBiappDWca8hhdzG0LxTCN8GsDbAPqIyHoRGQtgIoCzRKQKwJnRMlmEeQ0v5jZ/NDqEYowZEVA1OMVtiZt3KGTkAL1Jvnfqnn+V5KB56T3w2Ca5mNdM++oyd4itrPBNX23wobnXrfmBKpf94u1UNitpYcxt4VFdVHnVle7h0fNH36Pq2hQEr4xN1Lyr9erbK2aNdeL6pStSfr94cSUmEZGl2IETEVmKHTgRkaWs3I3Qq2ag77DRDQ1fBwDzJz/mxGe/kFPL2ikDvrxCL5d/9rfuuGZpJHjMGwCmbHN3rdt/Q2tf7cak20axnfXyR6o8q/UsJy7wfV5RH2Py5rDVQ1T5i4llTrz5eH2a0uLrH3Ri/4k8H49wt5Ip+1ng7dKOT+BERJZiB05EZCnrh1D8vAcuxDps4ZUNi1V50PjgKYa7LjrJif2HNFBu23a5O1Xwgf96SNXFGjaZvVPvtjpr5Hec2CxZ6r+c0mzRtjJVLmj9sRNHRD+Hztje3okf+8Ulqq7Fs/p3uyk2O3Hnl/U9p1zhDptd1WrdIbU3U/gETkRkKXbgRESWYgdORGSp0I2Bx7vM3s87xRAHbVi22P8C5aivLj1JlWdPvNeJD2WJ9R/++3JVbr0ot5bL55vPv7tHlc/7xsjAayM1Xzpxi3Xxb5kR6d1Dlb9Z/IETx5qamE18AicishQ7cCIiS7EDJyKyVOjGwL38y+x73OfOEY81nzvWWHkyvCcE8XSg1Im0do9LqRvzuaqLd9z7nMuvUuW27+u53vUJto1So36PHgPHouC5+LUJ3qP6TH1M6IlNg8e9D8uRaeF8AicishQ7cCIiS4V6CMXPO2xRE+O6QRddrcpqiqGPd+k+EHv5vnfYJtb96dDsO8Gd/jX/+D8HXudfHn/3H92pgkcsXKLq6nfuTFHryBbDr30tsG5NrR7C6fjCaicOPv46/fgETkRkKXbgRESWYgdORGSpvBoDj5d/O9lYp/f0hG864LDg9/VOTzwbPBEoUWbg8ao85rEX4/q6ST/Th7W3e9ZdHs9pgsn5/Er3tKN9rUTVdXxnlxPLgtzalkK+fawT/6Ttk6rO+zNx9j9vUXW9a95LZ7PixidwIiJLsQMnIrIUh1BSLN4TgUrebqnKBx3OnOcibfSUv69O6+XEd9w7XdV9rxmn/GXamon6gOgXh7u7PvYu0qtfZ3qmb/5lzEWqLtNDKpGW+vdu62/c6YH+k32+qt/txF1fSm+7EsUncCIiS7EDJyKyVKMduIh0EZHXRWS5iCwTkZuir7cVkddEpCr6Z5vG3otyB/MaTsxrfolnDLwWwG3GmA9E5HAA74vIawBGA5hrjJkoIhMATABwR/qaage1y2CcUwoBvVNihnYqzOm87v9GmSrPffiRTDfBVhnJa8UV+tiqejR14pm+LQse/8E5TixLMjDmXRBxwrrT9ZTTlcN0l7fyOPfnqs7o6Y/DVv7AiYtfyo1pg36NPoEbY6qNMR9E4x0AKgB0AjAUwBPRy54AcGG6Gkmpx7yGE/OaXw5pFoqIlAHoB+BdACXGmOpo1UYAJQFfMw7AOAAoRvNE20lpxLyGE/MafnF34CLSAsBMADcbY7aLuP/cMMYYEWlw93NjTDmAcgBoKW1z82TQNDmUnQqzhXkNp3Tn9Zj5o1V52aDHnfiiw75QdeufcocfHp8+RNU125T8j87mAXo/wGP6rHfil3oH707pt3ifPgqi4Gp3WCibOw7GEtcsFBEpwoEfhhnGmOejL9eISGm0vhTApvQ0kdKFeQ0n5jV/xDMLRQBMAVBhjJnkqZoNYFQ0HgUgvg0pKCcwr+HEvOaXeIZQTgFwBYCPROTrj5DvAjARwN9EZCyAtQAuS08TKU2Y13BiXvNIox24MeZNABJQPTi1zQmXI+f5xvdiTCvM9Gk9Yc2r/9Sdyde7/VSr96tUXa6OayYjU3ntPqpSlS999Vwnvrdspqq7oY37fb/hJp2DAk9T65HYeHiB73831vtsq9cn61y64odO3OyHu1Rd3eaPE2pPJnElJhGRpdiBExFZirsRppH/YAhMbvg6gIc9JGPeniZO/NiYi1Vd0YJFThzGIZNsMXv3qvLu092Bv+tOvFbVVV1f5MRTT5mm6gYV66l7ifBPafRq8a/DVLlthW53kzc+cGIbfz74BE5EZCl24ERElmIHTkRkKY6BZ9DItac5sX83Qq9V9w1Q5QztTphTCuZ/qMrnd/p2XF8nyK1Dc/PSex+pYq+Rbnw3jlN1d6fgdt2wJAXvYic+gRMRWYodOBGRpTiEkkHeg4tHvn2aqlvwTl8nzschEyI6dHwCJyKyFDtwIiJLsQMnIrIUx8CzxDseDgA9wXFvIjo0fAInIrIUO3AiIkuxAycishQ7cCIiS7EDJyKyFDtwIiJLiTGJHSSa0M1ENuPAidjtAWzJ2I1jy8e2HGWMOSJVb8a8Nop5TZ18bUuDuc1oB+7cVGSRMaZ/xm/cALYldXKp/WxL6uRS+9kWjUMoRESWYgdORGSpbHXg5Vm6b0PYltTJpfazLamTS+1nWzyyMgZORETJ4xAKEZGl2IETEVkqox24iAwRkUoRWSUiEzJ57+j9p4rIJhFZ6nmtrYi8JiJV0T/bZKAdXUTkdRFZLiLLROSmbLUlFZhX1ZbQ5JZ5VW3JybxmrAMXkQiAyQDOAdAXwAgR6Rv7q1JuGoAhvtcmAJhrjOkFYG60nG61AG4zxvQFMADA+Oj3IhttSQrzepBQ5JZ5PUhu5tUYk5H/AAwE8IqnfCeAOzN1f899ywAs9ZQrAZRG41IAlVlo04sAzsqFtjCvzC3zak9eMzmE0gnAOk95ffS1bCsxxlRH440ASjJ5cxEpA9APwLvZbkuCmNcAlueWeQ2QS3nlh5ge5sBfoxmbVykiLQDMBHCzMUadsZbptoRZNr6XzG36Ma+Z7cA/A9DFU+4cfS3bakSkFACif27KxE1FpAgHfhBmGGOez2ZbksS8+oQkt8yrTy7mNZMd+EIAvUSkm4g0ATAcwOwM3j/IbACjovEoHBjbSisREQBTAFQYYyZlsy0pwLx6hCi3zKtHzuY1wwP/5wJYCWA1gJ9l4YOHpwFUA9iPA2N6YwG0w4FPj6sAzAHQNgPtOBUH/qm1BMDi6H/nZqMtzCtzy7zam1cupScishQ/xCQishQ7cCIiSyXVgWd7qS2lB/MaXsxtuCQ8Bh5darsSB1YjrceBT61HGGOWB31NE2lqinFYQvej1NmDndhn9kpDdcyrvWLlFTj03DKvuWMHtm4xDZyJWZjEe54IYJUx5mMAEJFnAAwFEPiLXozDcJIMTuKWlArvmrmxqplXSzWSV+AQc8u85o455rm1Db2ezBBKXEttRWSciCwSkUX7sTeJ21GGMK/h1WhumVe7pP1DTGNMuTGmvzGmfxGapvt2lCHMazgxr3ZJpgPP1aW2lBzmNbyY25BJpgPP1aW2lBzmNbyY25BJ+ENMY0ytiFwP4BUAEQBTjTHLUtYyygrmNbyY2/BJZhYKjDEvA3g5RW2hHMG8hle+57aw05Gq/I+F7rdiyIrzVF39L9xZe7JgcXobliCuxCQishQ7cCIiS7EDJyKyVFJj4EREuWzHsAGq/N07F6hynal34h93flPVPb7THROvR27iEzgRkaXYgRMRWYpDKEQUWt/6yYeq/JsOH6lynWcz1t8+crmq67j4rbS1K1X4BE5EZCl24ERElmIHTkRkKY6BJ0ma6i03I23bxPV1FT8/SpVNQXwnI/WarvdoLqz4xInrvtwW13sQhVn1rSc78Qul9/tqm6jSbrPPiZt+kdjpZNnEJ3AiIkuxAycishSHUBoi+lzYguOPceJPhrZWdUcOWq/Kfz/mhfS1CwAu0MXybWVO/OjU76u6I+/J/WlQRIko7NLZiasn64OX/9XvHiduKsWqbkvdTlW+6KZbnbjN82+nsokZwSdwIiJLsQMnIrIUO3AiIkvl7Ri4/2SOj+9v58TtDtfjZK9/88m0t2db/R4nvnvToITe43sj3lHlpfcEXBgyBc2bO/Hqnx8f99c9Mqzcic8o3q/qIqKfbb6zbKgTf7ZQ/+z0+LW7XLt+zx5QBhh3yt/v+s5SVS0Liv1XO8q3fkuVmz//bmrblWF8AicishQ7cCIiS4VuCCXSsqUTbz2vr6rbdslXTjzuGL15+/jWqxO63y839VPlJds6OfGyFV1UXctK99v99m16hdh//P0WJ+593XsJtSVfbBk3UJXL73zAiY9rMj+h9/Rv2F9v6lT51b7PuwX9Y4XzT3KHV1Z+XKrqItsjTty7fIuqq6tcdegNzVOF3ctUuXJ8Ryc+qelW39XuEMqj2/SK5wUX9PFd+2kKWpc9fAInIrIUO3AiIkuxAycislToxsBNd3eJ7bx7Jsf9dY9vd8erH1xxhqrrOKkpghRV6DG0ui3VTtwb1apuzdPuFLciiai65utDl4qkFJZ1VeW6qe6Y9NM99PzIboXB08a8hq0eosrb9jVLsHXa80f/1YmbH90k8LoxAwer8uaTAy6kg+z7s/5MovLohz2l4PxPfeB8VW7/iX3L5WPhEzgRkaUa7cBFZKqIbBKRpZ7X2orIayJSFf0zvk2wKWcwr+HF3OaPeP7dPg3AQwCme16bAGCuMWaiiEyIlu9IffMy580vezpxp+F6SqHZu9d/uaMusAaomnySKi8d9KATH/1/V6u6Tstq42hlSk1DjuXVu8Ncqxk7VN0TZXM8Jf1P5uq63U58xfhbEaT53KWqXLhrcwKtPNiWte5PQdcYv1Hnt/u3Kj/Z5RQnrl233n95MqYhx3KbiEiv7k48svO/4v66uz9353l2eFJ/z/3TRW3X6BO4MWYegC98Lw8F8EQ0fgLAhSluF6UZ8xpezG3+SPSTsxJjzNef0G0EUBJ0oYiMAzAOAIrRPOgyyg3Ma3jFlVvm1S5Jf4hpjDEAAg+TM8aUG2P6G2P6FyF4NgflFuY1vGLllnm1S6JP4DUiUmqMqRaRUgCbUtmoZBRs/tKJj533Y1X31qnu1KPDC/R0ryldX3fiYa/p6WZf/L7MiYtf+VDVmXr9e1D1p/5OXHHhQ7ptcKcO9rpyuX6fGOPsGZTVvK662p3KubTsocDres++VpU7vO1+X1v/PXiaWFLjn55Tmj67XS/lbx+Jb2pa8wKdY3N4Rp9wc/Z39mv1g/S2FE8+5f4MtCuIf8rnW5d+w33PXeHeriDRJ/DZAEZF41EAXkxNcyjLmNfwYm5DKJ5phE8DeBtAHxFZLyJjAUwEcJaIVAE4M1omizCv4cXc5o9Gh1CMMSMCqgYHvJ5VtZ9tcOJuIzaousvhTtuqelBP8Zt67p+d+OkeL6u6gnL377lLVp2n6qp3tFTlyn7eFWL678fzV7i71kndxoaanzG5mNcnL/+TpxT8bNFukV7F2np6+lfXRdq3d+IPb3zQVxu8+tLrln+MVOWey98JuDI5uZjbeJiIPky8Q+SwgCu1nm+MVuUelYtT1aScx5WYRESWYgdORGQpduBERJbK2y3wet2gDzO9+4bjnLhquj74tHKwOz4+s+c/4r7HNetOV+WCC7c5cX1txpfOUxIqfn9U4xc14ohFKWhIyERKOjjxqqH6s4Q6Ezzxc/E+9/en9293q7pY21v4SaHbBUa6dg68bnfP9qpcPdBta/fH9NYbtRtrDqEFyeETOBGRpdiBExFZKm+HUGIpWpeaJcRLJ39TlVvvCNdm8qn2k5WXOfHcbzwXeN3nA/TwU7spaWtS0vq8cJ0T93pqYRZbkpsqfuMOTa069+EYV2oTRru7eRYs+zDGldqG2/UpGsWnuQdNv9Pvmbjfx2vm5Xpn3kevv9SJm7y+RNWZ/fsSukcQPoETEVmKHTgRkaXYgRMRWSpvx8AjvXuocuU1RzjxR5c94L86sZtI45eQxyNuDlbdr3fu61nkfi7x0Tl6KfvzK9zpX/c9eJmq6/CIZ7poffAEs52X6K0VtvbWOf+/wX/wlIJ3xuvz3HhdvsNd1l0f4/75qmhzfF3Q6lo9VTCyc78T+/fFNaec4MSbb9+j6j76Dz3OHmuqYrwuOWyrLj9e7sQXnHqRqqtdszbp+3nxCZyIyFLswImILMUOnIjIUnk7Br7iRr00tvKiyZ5S8Jj3St88ztX726nyOc3d09S3f/8rVdd6OiiGZrPec+KRbW9TdVP+8z4nPqaoSNWNONxdujziLj0+/vurj3XiOhP8vPKj1veqctdC/zh3fCfC9JnypSrX79kTcGV+8p+686fLpsb1dZdO+qkqd6751ImrZ/dRdXM9Wzq3LCj2vZP+Gdhe7+bn9g1n6fdZ4b5v1Zl/iaudAND3zdFO3L0mvScC8QmciMhS7MCJiCwV6iGUwm56B7m+M91/dv2tw/2+q4NPVfGepFM4Uk87qry1qyqfM3wyKHltp+ptB25Z607PqxmvhyWm9pvmxP2a6GeSO9oti+t+M3Z0V+UTmq5T5WObBP+qvO+Z8Si7U7tUOmzWf1cPRZ3VbHfAlT6+Kbkbz3EPwF74bf/vnH/YxDXpC53nl279rvtV71Xpd7ne8z5nBjft0W26n+l5q7s8v3bXruAvTAE+gRMRWYodOBGRpdiBExFZyv4xcNGDYxt+OtCJ375xkqprKt7pZ8Fj3kNXfl/f4mz3BPla30k6nV73neIx3A0Xn6ynSF1cdon7Pp98Copf4dz3nbjTXF034cxrnHhr78S2Aj7yn5+p8iOnXqLKb058KPBrL3/rSifuWRX/1qb5qPuTG/QL4+L7uovHvKHK0xae3PCFjfjLX4eocu2P3am+N93/saq7ptXrge8zf4/bdU569TxV13vrEv/lacMncCIiS7EDJyKylPVDKJG+vVX5g5u8K/H0ij2v6jo9fenMGe5Kr+6/+kDVmRgHEO9tFfx3YAH/fsyIojnu8EqHOYm9hz/D7Yqs/9XISWbHTlV+ckdHJ77i8I3+yx0/b79Ul89ZGnBlbMuvTWw3wgV79e/yr6/9sRP3fPUdVZf8/obxYw9DRGSpRjtwEekiIq+LyHIRWSYiN0Vfbysir4lIVfTPNo29F+UO5jWcmNf8Es8TeC2A24wxfQEMADBeRPoCmABgrjGmF4C50TLZg3kNJ+Y1jzQ60GeMqQZQHY13iEgFgE4AhgI4I3rZEwDeAHBHWloZw5pfBU8H9Juxo9SJH56op4l1m+Yu3Y6U6eXxxjMeWnVViap78OLg3dTGfvodVa7f/HncbU23XM8rJSbX81q3ebMqz7janYJ3xVNTMt2cmL6/8nwn3vubUlVXNHdRppvToEP6pEZEygD0A/AugJLoDwsAbARQEvA14xCd7VmM5om2k9KIeQ0n5jX84v4QU0RaAJgJ4GZjzHZvnTHG4OCj6b6uKzfG9DfG9C9CYossKH2Y13BiXvNDXE/gIlKEAz8MM4wxz0dfrhGRUmNMtYiUAtiUrkbGUnHKk6q8v8EfywOW7erkxB3HrNGVY9zpTOd30P88ahdxV2td4DvA1O/Dfe4kok9+f7Sqa7bzPf/lWZXLeaXE2ZTXgvnuqsXzz71c1f3ob6868fAWeugllrs2fcuJn11wUowrgc5z3A6jxRuVqq5+p7urYOF+3wrSHBHPLBQBMAVAhTHGuzZ9NoBR0XgUgBdT3zxKF+Y1nJjX/BLPE/gpAK4A8JGILI6+dheAiQD+JiJjAawFcFl6mkhpwryGE/OaR+KZhfImDtpO3TE4tc2hTGFew4l5zS/Wrxcu33akKo9puS7gSuB3JZ6x7QY/g2/cmlp9GszZc25S5T4PuUv0m32YW2PeFL/6Nfrn6Li3RjvxkpOnqbpju1Q7cW3rVqqu7sttKW9bqNTXueG/K1TV9D7uqTvT0QWJ6IV34762rvFLcg6X0hMRWYodOBGRpawfQnnkoQtVecxdDwZcGb9vLhitynW1ESdu/5I+MLX3U3onshizGMkiZr8+nHjPjuA50TN7/sOJL+iqp8KBQyiURnwCJyKyFDtwIiJLsQMnIrKU9WPgHR7R04QunHlu0u95VI3vtA/Dke18V7gl+HQnrz1HHq7KTTJ3vi3lIT6BExFZih04EZGlrB9C8a7kAoDajTVZagiFWc9f/duJq4frA7FLI82cePeNerfKJv+b3nZRfuMTOBGRpdiBExFZih04EZGl7B8DJ8qA+l27nHjk1beour03fuHEu+Z0UHWtsCq9DaO8xidwIiJLsQMnIrIUh1CIDlHTfy70ld24JVZnuDWUz/gETkRkKXbgRESWYgdORGQpMRncaU9ENgNYC6A9gC0Zu3Fs+diWo4wxR6TqzZjXRjGvqZOvbWkwtxntwJ2biiwyxvTP+I0bwLakTi61n21JnVxqP9uicQiFiMhS7MCJiCyVrQ68PEv3bQjbkjq51H62JXVyqf1si0dWxsCJiCh5HEIhIrIUO3AiIktltAMXkSEiUikiq0RkQibvHb3/VBHZJCJLPa+1FZHXRKQq+mebDLSji4i8LiLLRWSZiNyUrbakAvOq2hKa3DKvqi05mdeMdeAiEgEwGcA5APoCGCEifTN1/6hpAIb4XpsAYK4xpheAudFyutUCuM0Y0xfAAADjo9+LbLQlKczrQUKRW+b1ILmZV2NMRv4DMBDAK57ynQDuzNT9PfctA7DUU64EUBqNSwFUZqFNLwI4Kxfawrwyt8yrPXnN5BBKJwDrPOX10deyrcQYUx2NNwIoyeTNRaQMQD8A72a7LQliXgNYnlvmNUAu5ZUfYnqYA3+NZmxepYi0ADATwM3GmO3ZbEuYZeN7ydymH/Oa2Q78MwBdPOXO0deyrUZESgEg+uemTNxURIpw4AdhhjHm+Wy2JUnMq3Se7KsAAACySURBVE9Icsu8+uRiXjPZgS8E0EtEuolIEwDDAczO4P2DzAYwKhqPwoGxrbQSEQEwBUCFMWZSNtuSAsyrR4hyy7x65GxeMzzwfy6AlQBWA/hZFj54eBpANYD9ODCmNxZAOxz49LgKwBwAbTPQjlNx4J9aSwAsjv53bjbawrwyt8yrvXnlUnoiIkvxQ0wiIkuxAycishQ7cCIiS7EDJyKyFDtwIiJLsQMnIrIUO3AiIkv9P4jp5isAxoAiAAAAAElFTkSuQmCC"/>


```python
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    out = self.linear1(x)
    out = self.relu1(out)
    out = self.linear2(out)
    return out
```


```python
model = NeuralNet(input_size, hidden_size, output_size)
```


```python
#loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


```python
n_total_steps = len(train_loader)


for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    # 100, 1, 28, 28
    # 100, 784
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print(f"epoch = {epoch+1} / {num_epochs}, step{i+1}/{n_total_steps}, loss={loss.item():.4f}")

  
```

<pre>
epoch = 1 / 2, step100/600, loss=0.3824
epoch = 1 / 2, step200/600, loss=0.1703
epoch = 1 / 2, step300/600, loss=0.2758
epoch = 1 / 2, step400/600, loss=0.3444
epoch = 1 / 2, step500/600, loss=0.2894
epoch = 1 / 2, step600/600, loss=0.2160
epoch = 2 / 2, step100/600, loss=0.1872
epoch = 2 / 2, step200/600, loss=0.3154
epoch = 2 / 2, step300/600, loss=0.1302
epoch = 2 / 2, step400/600, loss=0.2134
epoch = 2 / 2, step500/600, loss=0.2352
epoch = 2 / 2, step600/600, loss=0.1436
</pre>

```python
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)

    _, pred = torch.max(outputs,1)
    n_samples += labels.shape[0]
    n_correct += (pred == labels).sum().item()

  acc = 100.0 * n_correct / n_samples
  print(f"accuracy = {acc}")
```

<pre>
accuracy = 95.28
</pre>

```python

```
