import numpy as np
import cython
from matplotlib import pyplot as plt
import pandas as pd

df=pd.read_csv('HR_comma_sep.csv')
#print(df.head())

left=df[df.left==1]
#print(left.shape)

retain=df[df.left==0]
#print(retain.shape)

subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','left']]
print(subdf.head())

salary_dummies=pd.get_dummies(subdf.salary,prefix='salary')
df=pd.concat([subdf,salary_dummies],axis='columns')

df['salary_low'] = df['salary_low'].replace({False: 0, True: 1})
df['salary_high'] = df['salary_high'].replace({False: 0, True: 1})
df['salary_medium'] = df['salary_medium'].replace({False: 0, True: 1})
print(df.head())

df.drop('salary',axis='columns',inplace=True)
print(df.head())

print(df.groupby('left').mean())

X=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary_high','salary_medium','salary_low']]
print(X.head())
y=df.left
print(y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train, y_train)
print(model.predict(X_test))

accuracy=model.score(X_test,y_test)
print(accuracy*100)
