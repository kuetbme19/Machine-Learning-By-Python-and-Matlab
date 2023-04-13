import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('D4.csv')
#print(df.head())
plt.scatter(df.age,df.insurance,marker='*',color='red')
#plt.show()
print(df.shape)

X_train, X_test, y_train,y_test=train_test_split(df[['age']],df.insurance,test_size=0.1)
print(X_test)

model=LogisticRegression()
model.fit(X_train, y_train)
res=model.predict(X_test)
print(res)

sco=model.score(X_test,y_test)
print(sco)
print(model.predict_proba(X_test))

