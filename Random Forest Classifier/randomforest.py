import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_digits

digit=load_digits()
print(digit)
print(dir(digit))

plt.gray()
for i in range(1):
    plt.matshow(digit.images[i])
    #plt.show()

df=pd.DataFrame(digit.data)
print(df.head())
print(df.shape)

df['target']=digit.target
print(df)

X=df.drop('target',axis='columns')
print(X)
y=df.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2)

print(len(X_train),len(X_test),len(y_train),len(y_test))

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)*100)

y_pred=model.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))