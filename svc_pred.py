# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

print('---- initial info ----')
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv("data/test.csv")

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

# sns.pairplot(df_train, size=1.5)
# plt.show()

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
X_test_std = stdsc.fit_transform(X_test)

print(pd.DataFrame(X_train_std).describe())
print()

print('---- predict ----')
clf = svm.SVC(random_state=0)
clf.fit(X_train_std, y_train)

pred_train = clf.predict(X_train_std)
pred_test = clf.predict(X_test_std)

print('---- report ----')
print(classification_report(y_train, pred_train))
print(accuracy_score(y_train, pred_train))

df_pred_test = pd.DataFrame(pred_test, columns=['Survived'])
df_submit = pd.concat([df_test['PassengerId'], df_pred_test['Survived']], axis=1)
df_submit.to_csv('submit.csv', index=False)

# default
# 0.8462401795735129

# C=100., gamma=0.01
# 0.8406285072951739