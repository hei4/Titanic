# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_util import TableDataset
from torch_net import MLP


print('---- initial info ----')
df_train = pd.read_csv('data_root/train.csv')
df_test = pd.read_csv("data_root/test.csv")

print(df_train.head())
print()
print(df_train.tail())
print()
print(df_train.describe())
print()

print('---- fill NA ----')
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

print(df_train.describe())
print()

print('---- get dummy ----')
df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked'], drop_first=True)

print(df_train.describe())
print()

print('---- independent variable ----')
y_train = df_train.loc[:, 'Survived']
X_train = df_train.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                          'Sex_male', 'Embarked_Q', 'Embarked_S']]
X_test = df_test.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                        'Sex_male', 'Embarked_Q', 'Embarked_S']]

print(X_train.describe())
print()

print('---- standard scaling ----')
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print(pd.DataFrame(X_train_std).describe())
print()

X_train_std, X_valid_std, y_train, y_valid = train_test_split(X_train_std, y_train, test_size=0.2, random_state=0)

X_train_std = X_train_std.astype(np.float32).reshape(len(X_train_std), -1)
y_train = y_train.as_matrix().astype(np.int64).ravel().reshape(len(y_train), )

X_valid_std = X_valid_std.astype(np.float32).reshape(len(X_valid_std), -1)
y_valid = y_valid.as_matrix().astype(np.int64).ravel().reshape(len(y_valid), )

X_test_std = X_test_std.astype(np.float32).reshape(len(X_test_std), -1)

print(X_train_std.shape)
print(y_train.shape)
print(X_valid_std.shape)
print(y_valid.shape)
print(X_test.shape)

batch_size = 32
epoch_size = 200

trainset = TableDataset(X_train_std, y_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

validset = TableDataset(X_valid_std, y_valid)
validloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

net = MLP()
print(net)
print()

net.to(device)  # for GPU

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

epoch_list = []
train_acc_list = []
valid_acc_list = []
for epoch in range(epoch_size):  # loop over the dataset multiple times

    train_true = []
    train_pred = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        X, y_true = data

        train_true.extend(y_true.tolist())

        X, y_true = X.to(device), y_true.to(device)  # for GPU

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_logit = net(X)
        loss = criterion(y_logit, y_true)
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(y_logit.data, 1)
        train_pred.extend(y_pred.tolist())

        # print statistics
        print('[epochs: {}, mini-batches: {}, images: {}] loss: {:.3f}'.format(
            epoch + 1, i + 1, (i + 1) * batch_size, loss.item() / len(X)))

    valid_true = []
    valid_pred = []
    with torch.no_grad():
        for data in validloader:
            X, y_true = data
            valid_true.extend(y_true.tolist())
            X, y_true = X.to(device), y_true.to(device)  # for GPU

            y_logit = net(X)
            _, y_pred = torch.max(y_logit.data, 1)
            valid_pred.extend(y_pred.tolist())

    train_acc = accuracy_score(train_true, train_pred)
    valid_acc = accuracy_score(valid_true, valid_pred)
    print('    epocs: {}, train acc.: {:.3f}, valid acc.: {:.3f}'.format(epoch + 1, train_acc, valid_acc))
    print()

    epoch_list.append(epoch + 1)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print('Finished Training')

print('Save Network')
torch.save(net.state_dict(), 'model.pth')

df_log = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list,
                       'valid/accuracy': valid_acc_list})

print('Save Training Log')
df_log.to_csv('train.log', index=False)


print('---- predict ----')
y_logit = net(torch.from_numpy(X_test_std).to(device))
_, y_pred = torch.max(y_logit.data, 1)

df_pred_test = pd.DataFrame(y_pred.tolist(), columns=['Survived'])
df_submit = pd.concat([df_test['PassengerId'], df_pred_test['Survived']], axis=1)
df_submit.to_csv('submit.csv', index=False)

print('---- show plot ----')
plt.scatter(df_log['epoch'], df_log['train/accuracy'], label='train/accuracy')
plt.scatter(df_log['epoch'], df_log['valid/accuracy'], label='valid/accuracy')
plt.legend(loc='lower right')
plt.show()
