# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv("data/test.csv")

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked'], drop_first=True)

y_train = df_train.loc[:, 'Survived']
X_train = df_train.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                          'Sex_male', 'Embarked_Q', 'Embarked_S']]
X_test = df_test.ix[:, ['Pclass','Age', 'SibSp', 'Parch', 'Fare',
                        'Sex_male', 'Embarked_Q', 'Embarked_S']]

pipe_svc = make_pipeline(StandardScaler(), svm.SVC(random_state=0))

param_range = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]

param_grid = [{'svc__C': param_range,
               'svc__gamma': param_range}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=2)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)

print(gs.best_params_)

# 0.8282828282828283
# {'svc__C': 100.0, 'svc__gamma': 0.01}