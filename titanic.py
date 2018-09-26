# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv("data/test.csv")

print(df_train.head())
print()

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

print(df_train.head())
print()

df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked'])
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked'])

print(df_train.head())
print()

y_train = df_train.loc[:, 'Survived']
X_train = df_train.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                          'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
X_test = df_test.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                        'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

print(X_train.head())
print()

age_mean = X_train['Age'].mean()
age_std = X_train['Age'].std()
fare_mean = X_train['Fare'].mean()
fare_std = X_train['Fare'].std()

print(age_mean, age_std)
print(fare_mean, fare_std)
print()

X_train['Age'] = (X_train['Age'] - age_mean) / age_std
X_train['Fare'] = (X_train['Fare'] - fare_mean) / fare_std

X_test['Age'] = (X_test['Age'] - age_mean) / age_std
X_test['Fare'] = (X_test['Fare'] - fare_mean) / fare_std

print(X_train.head())
print()
print(X_test.head())
print()

clf = svm.SVC()
clf.fit(X_train, y_train)

pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

print(classification_report(y_train, pred_train))
print(accuracy_score(y_train, pred_train))

print(pred_test)

df_pred_test = pd.DataFrame(pred_test, columns=['Survived'])
df_submit = pd.concat([df_test['PassengerId'], df_pred_test['Survived']], axis=1)
df_submit.to_csv('submit.csv', index=False)
