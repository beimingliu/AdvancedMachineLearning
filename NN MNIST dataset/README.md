
# Neural Networks for MNIST dataset


```python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
```


```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```

## Loading MNIST
Here we load the dataset and create data loaders.


```python
train_ds = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_ds = datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
```


```python
batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
```

## Feed Forward Neural Network


```python
# for the number of neurons in the hidden unit
def get_model(M = 300):
    net = nn.Sequential(nn.Linear(28*28, M),
                        nn.ReLU(),
                        nn.Linear(M, 10))
    return net.cuda()
```


```python
def train_model(train_loader, test_loader, num_epochs, model, optimizer):
    sum_loss = 0.0
    total = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):  
            batch = images.shape[0] # size of the batch
            # Convert torch tensor to Variable, change shape of the input
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total += batch
            sum_loss += batch * loss.data[0]

        train_loss = sum_loss/total
        print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, train_loss))
        val_acc, val_loss = model_accuracy_loss(model, test_loader)
        print('Epoch [%d/%d], Valid Accuracy: %.4f, Valid Loss: %.4f' %(epoch+1, num_epochs, val_acc, val_loss))
    return val_acc, val_loss, train_loss
```


```python
def model_accuracy_loss(model, test_loader):
    model.eval()
    correct = 0
    sum_loss = 0.0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        loss = F.cross_entropy(outputs, labels)
        sum_loss += labels.size(0)*loss.data[0]
        total += labels.size(0)
        correct += pred.eq(labels.data).cpu().sum()
    return 100 * correct / total, sum_loss/ total
```

## Training

Learning Rate tuning


```python
%%time
learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
validation_accuracy0 = []

for r in learning_rates:
    net = get_model()
    optimizer = optim.Adam(net.parameters(), lr=r)
    model_accuracy_loss(net, test_loader)
    val_acc, _, _ = train_model(train_loader, test_loader, num_epochs=10, model=net, optimizer=optimizer)
    validation_accuracy0.append(val_acc)
```

    Epoch [1/10], Loss: 82.9692
    Epoch [1/10], Valid Accuracy: 10.0900, Valid Loss: 2.7915
    Epoch [2/10], Loss: 42.6954
    Epoch [2/10], Valid Accuracy: 10.6600, Valid Loss: 2.7205
    Epoch [3/10], Loss: 29.3076
    Epoch [3/10], Valid Accuracy: 8.9400, Valid Loss: 2.6788
    ......
    Epoch [8/10], Loss: 0.5196
    Epoch [8/10], Valid Accuracy: 92.3300, Valid Loss: 0.2747
    Epoch [9/10], Loss: 0.4927
    Epoch [9/10], Valid Accuracy: 92.6100, Valid Loss: 0.2634
    Epoch [10/10], Loss: 0.4702
    Epoch [10/10], Valid Accuracy: 92.7600, Valid Loss: 0.2548
    CPU times: user 2min 17s, sys: 26.2 s, total: 2min 44s
    Wall time: 8min 28s



```python
pd.DataFrame(data=[learning_rates, validation_accuracy0],index=['Learning Rate','Validation Accuracy'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Learning Rate</th>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.01</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>Validation Accuracy</th>
      <td>9.81</td>
      <td>9.86</td>
      <td>95.27</td>
      <td>97.980</td>
      <td>97.6100</td>
      <td>92.76000</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%time
learning_rates = np.linspace(0.0001, 0.001, 5)
validation_accuracy0 = []

for r in learning_rates:
    net = get_model()
    optimizer = optim.Adam(net.parameters(), lr=r)
    model_accuracy_loss(net, test_loader)
    val_acc, _, _ = train_model(train_loader, test_loader, num_epochs=10, model=net, optimizer=optimizer)
    validation_accuracy0.append(val_acc)

```

    Epoch [1/10], Loss: 0.4896
    Epoch [1/10], Valid Accuracy: 92.4800, Valid Loss: 0.2697
    Epoch [2/10], Loss: 0.3653
    Epoch [2/10], Valid Accuracy: 94.2100, Valid Loss: 0.2028
    Epoch [3/10], Loss: 0.3053
    Epoch [3/10], Valid Accuracy: 95.2300, Valid Loss: 0.1615
    ......
    Epoch [7/10], Loss: 0.0717
    Epoch [7/10], Valid Accuracy: 97.2000, Valid Loss: 0.1066
    Epoch [8/10], Loss: 0.0654
    Epoch [8/10], Valid Accuracy: 97.8000, Valid Loss: 0.0867
    Epoch [9/10], Loss: 0.0601
    Epoch [9/10], Valid Accuracy: 97.8900, Valid Loss: 0.0873
    Epoch [10/10], Loss: 0.0558
    Epoch [10/10], Valid Accuracy: 97.8800, Valid Loss: 0.0899
    CPU times: user 1min 48s, sys: 23 s, total: 2min 11s
    Wall time: 6min 52s



```python
pd.DataFrame(data=[learning_rates, validation_accuracy0],index=['Learning Rate','Validation Accuracy'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Learning Rate</th>
      <td>0.0001</td>
      <td>0.000325</td>
      <td>0.00055</td>
      <td>0.000775</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>Validation Accuracy</th>
      <td>97.4800</td>
      <td>97.970000</td>
      <td>97.67000</td>
      <td>97.820000</td>
      <td>97.880</td>
    </tr>
  </tbody>
</table>
</div>



## Number of neurons M in the hidden layer


```python
%%time
M = [10, 50, 100, 300, 1000, 2000]
validation_accuracy = []

for m in M:
    net = get_model(m)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    model_accuracy_loss(net, test_loader)
    val_acc, val_loss, train_loss = train_model(train_loader, test_loader, num_epochs=10, model=net, optimizer=optimizer)
    validation_accuracy.append(val_acc)


```

    Epoch [1/10], Loss: 0.4706
    Epoch [1/10], Valid Accuracy: 89.2600, Valid Loss: 0.3844
    Epoch [2/10], Loss: 0.4162
    Epoch [2/10], Valid Accuracy: 90.1400, Valid Loss: 0.3476
    Epoch [3/10], Loss: 0.3939
    Epoch [3/10], Valid Accuracy: 89.6400, Valid Loss: 0.3680
    ......
    Epoch [8/10], Loss: 0.2053
    Epoch [8/10], Valid Accuracy: 95.8000, Valid Loss: 0.2235
    Epoch [9/10], Loss: 0.1999
    Epoch [9/10], Valid Accuracy: 95.5400, Valid Loss: 0.2776
    Epoch [10/10], Loss: 0.1958
    Epoch [10/10], Valid Accuracy: 95.1900, Valid Loss: 0.2737
    CPU times: user 2min 25s, sys: 28.3 s, total: 2min 54s
    Wall time: 8min 29s



```python
pd.DataFrame(data=[M, validation_accuracy],index=['M','Validation Accuracy'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>10.00</td>
      <td>50.00</td>
      <td>100.00</td>
      <td>300.00</td>
      <td>1000.00</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>Validation Accuracy</th>
      <td>90.12</td>
      <td>94.95</td>
      <td>95.12</td>
      <td>94.99</td>
      <td>95.53</td>
      <td>95.19</td>
    </tr>
  </tbody>
</table>
</div>



If we look at the end of 10 epoches, M = 1000 seems to be the best with a Validation Accuracy of 95.53, however we do see that most of the models overfit: i.e. loss decreases while the validation accuracy increases.



## Models with L2 regularization
To add L2 regularization use the `weight_decay` argument on the optimizer


```python
%%time
weight_decay = [0, 0.0001, 0.001, 0.01, 0.1, 0.3]
validation_accuracy2 = []
Train_loss = []
Validation_loss = []

for decay_r in weight_decay:
    net = get_model(300)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay = decay_r)
    model_accuracy_loss(net, test_loader)
    val_acc, train_loss, val_loss = train_model(train_loader, test_loader, num_epochs=20, model=net, optimizer=optimizer)
    print(val_acc, train_loss, val_loss)
    validation_accuracy2.append(val_acc)
    Train_loss.append(round(train_loss,4))
    Validation_loss.append(round(val_loss,4))
```

    Epoch [1/20], Loss: 0.2240
    Epoch [1/20], Valid Accuracy: 95.7200, Valid Loss: 0.1341
    Epoch [2/20], Loss: 0.1587
    Epoch [2/20], Valid Accuracy: 97.1700, Valid Loss: 0.0930
    Epoch [3/20], Loss: 0.1268
    Epoch [3/20], Valid Accuracy: 97.6300, Valid Loss: 0.0735
    ......
    Epoch [17/20], Loss: 0.8197
    Epoch [17/20], Valid Accuracy: 86.0500, Valid Loss: 0.7873
    Epoch [18/20], Loss: 0.8191
    Epoch [18/20], Valid Accuracy: 84.2400, Valid Loss: 0.8025
    Epoch [19/20], Loss: 0.8186
    Epoch [19/20], Valid Accuracy: 86.2400, Valid Loss: 0.7820
    Epoch [20/20], Loss: 0.8181
    Epoch [20/20], Valid Accuracy: 85.7900, Valid Loss: 0.7806
    85.79 0.7806492678642273 0.8181052619043986
    CPU times: user 4min 30s, sys: 55.4 s, total: 5min 25s
    Wall time: 16min 26s



```python
weight_decay = [0, 0.0001, 0.001, 0.01, 0.1, 0.3]
pd.DataFrame(data=[weight_decay, validation_accuracy2,Train_loss,Validation_loss],
             index=['Weight decay ','Validation Accuracy','Train Loss','Validation Loss'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Weight decay</th>
      <td>0.0000</td>
      <td>0.0001</td>
      <td>0.0010</td>
      <td>0.0100</td>
      <td>0.1000</td>
      <td>0.3000</td>
    </tr>
    <tr>
      <th>Validation Accuracy</th>
      <td>97.5400</td>
      <td>97.8500</td>
      <td>97.7100</td>
      <td>96.2200</td>
      <td>89.6800</td>
      <td>85.7900</td>
    </tr>
    <tr>
      <th>Train Loss</th>
      <td>0.1475</td>
      <td>0.0836</td>
      <td>0.0777</td>
      <td>0.1439</td>
      <td>0.4420</td>
      <td>0.7806</td>
    </tr>
    <tr>
      <th>Validation Loss</th>
      <td>0.0342</td>
      <td>0.0402</td>
      <td>0.0712</td>
      <td>0.1692</td>
      <td>0.4718</td>
      <td>0.8181</td>
    </tr>
  </tbody>
</table>
</div>



## Models with Dropout


```python
def get_model_v2(M = 300, p=0):
    modules = []
    modules.append(nn.Linear(28*28, M))
    modules.append(nn.ReLU())
    if p > 0:
        modules.append(nn.Dropout(p))
    modules.append(nn.Linear(M, 10))
    return nn.Sequential(*modules).cuda()
```


```python
%%time
dropout = [0.1, 0.3, 0.5, 0.7, 0.9]
validation_accuracy3 = []
Train_loss3 = []
Validation_loss3 = []

for p1 in dropout:
    net2 = get_model_v2(M = 300, p=p1)
    optimizer = optim.Adam(net2.parameters(), lr=0.001)
    model_accuracy_loss(net2, test_loader)
    val_acc, train_loss, val_loss = train_model(train_loader, test_loader,
                                                num_epochs=20, model=net2,
                                                optimizer=optimizer)
    print(val_acc, train_loss, val_loss)
    validation_accuracy3.append(val_acc)
    Train_loss3.append(round(train_loss,4))
    Validation_loss3.append(round(val_loss,4))
```

    Epoch [1/20], Loss: 0.2344
    Epoch [1/20], Valid Accuracy: 96.6700, Valid Loss: 0.1097
    Epoch [2/20], Loss: 0.1678
    Epoch [2/20], Valid Accuracy: 97.3500, Valid Loss: 0.0874
    Epoch [3/20], Loss: 0.1364
    Epoch [3/20], Valid Accuracy: 97.5800, Valid Loss: 0.0796
    ......
    Epoch [17/20], Loss: 0.5545
    Epoch [17/20], Valid Accuracy: 95.5500, Valid Loss: 0.1655
    Epoch [18/20], Loss: 0.5507
    Epoch [18/20], Valid Accuracy: 95.3200, Valid Loss: 0.1716
    Epoch [19/20], Loss: 0.5478
    Epoch [19/20], Valid Accuracy: 95.5600, Valid Loss: 0.1668
    Epoch [20/20], Loss: 0.5446
    Epoch [20/20], Valid Accuracy: 95.5600, Valid Loss: 0.1641
    95.56 0.16408598736524582 0.5445897932958603
    CPU times: user 3min 49s, sys: 45.5 s, total: 4min 34s
    Wall time: 13min 41s



```python
pd.DataFrame(data=[dropout, validation_accuracy3, Train_loss3, Validation_loss3],
             index=['Dropout rate','Validation Accuracy','Train Loss','Validation Loss'])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dropout rate</th>
      <td>0.1000</td>
      <td>0.3000</td>
      <td>0.5000</td>
      <td>0.7000</td>
      <td>0.9000</td>
    </tr>
    <tr>
      <th>Validation Accuracy</th>
      <td>98.1100</td>
      <td>98.2200</td>
      <td>98.1300</td>
      <td>97.9300</td>
      <td>95.5600</td>
    </tr>
    <tr>
      <th>Train Loss</th>
      <td>0.1046</td>
      <td>0.0852</td>
      <td>0.0798</td>
      <td>0.0815</td>
      <td>0.1641</td>
    </tr>
    <tr>
      <th>Validation Loss</th>
      <td>0.0431</td>
      <td>0.0634</td>
      <td>0.0998</td>
      <td>0.1824</td>
      <td>0.5446</td>
    </tr>
  </tbody>
</table>
</div>



## Models with 3 layer


```python
def get_model_v3(M = 500, p=0):
    modules = []
    modules.append(nn.Linear(28*28, M))
    modules.append(nn.ReLU())
    if p > 0:
        modules.append(nn.Dropout(p))
    modules.append(nn.Linear(M, int(.3*M)))
    modules.append(nn.ReLU())
    modules.append(nn.Linear(int(.3*M), 10))

    return nn.Sequential(*modules).cuda()
```


```python
%%time
M = [300, 500, 800]
dropout = [0.3, 0.5]
weight_decay = [0.0001, 0.001]

best_validation_accuracy = 0
best_para = None

for m in M:
    for p1 in dropout:
        for decay_r in weight_decay:

            net3 = get_model_v3(M = m, p=p1)
            optimizer = optim.Adam(net3.parameters(), lr=0.001, weight_decay = decay_r)
            model_accuracy_loss(net, test_loader)
            val_acc, train_loss, val_loss = train_model(train_loader, test_loader,
                                                        num_epochs=16, model=net3,
                                                        optimizer=optimizer)
            if val_acc > best_validation_accuracy:
                best_validation_accuracy = val_acc
                best_para = [decay_r, p1, m]
print('best parameters:', best_para)
print('best validation accuracy:', best_validation_accuracy)
```

    Epoch [1/16], Loss: 0.2781
    Epoch [1/16], Valid Accuracy: 96.0900, Valid Loss: 0.1316
    Epoch [2/16], Loss: 0.2051
    Epoch [2/16], Valid Accuracy: 96.9600, Valid Loss: 0.0991
    Epoch [3/16], Loss: 0.1717
    Epoch [3/16], Valid Accuracy: 97.5500, Valid Loss: 0.0782
    ......
    Epoch [14/16], Loss: 0.1504
    Epoch [14/16], Valid Accuracy: 97.3200, Valid Loss: 0.0909
    Epoch [15/16], Loss: 0.1490
    Epoch [15/16], Valid Accuracy: 97.6200, Valid Loss: 0.0790
    Epoch [16/16], Loss: 0.1477
    Epoch [16/16], Valid Accuracy: 97.4600, Valid Loss: 0.0858
    best parameters: [0.0001, 0.5, 800]
    best validation accuracy: 98.29
    CPU times: user 8min 59s, sys: 1min 37s, total: 10min 36s
    Wall time: 26min 33s



```python
print('best parameters:', best_para)
print('best validation accuracy:', best_validation_accuracy)
```

    best parameters: [0.0001, 0.5, 800]
    best validation accuracy: 98.29



```python
net4 = get_model_v3(M = 800, p=.5)
optimizer = optim.Adam(net4.parameters(), lr=0.001, weight_decay = 0.0001)
model_accuracy_loss(net4, test_loader)
val_acc, train_loss, val_loss = train_model(train_loader, test_loader,
                                            num_epochs=20, model=net4,
                                            optimizer=optimizer)
```

    Epoch [1/20], Loss: 0.2645
    Epoch [1/20], Valid Accuracy: 96.2800, Valid Loss: 0.1194
    Epoch [2/20], Loss: 0.2052
    Epoch [2/20], Valid Accuracy: 96.9500, Valid Loss: 0.0927
    Epoch [3/20], Loss: 0.1779
    Epoch [3/20], Valid Accuracy: 97.2000, Valid Loss: 0.0883
    ......
    Epoch [18/20], Loss: 0.0991
    Epoch [18/20], Valid Accuracy: 97.9700, Valid Loss: 0.0658
    Epoch [19/20], Loss: 0.0978
    Epoch [19/20], Valid Accuracy: 98.0600, Valid Loss: 0.0633
    Epoch [20/20], Loss: 0.0963
    Epoch [20/20], Valid Accuracy: 98.2200, Valid Loss: 0.0572


The 3-layer NN turned out to be doing similar to 2-layer NN with dropout rate of 0.3, the validation accuract are both 98.22
