import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris=load_iris()
#print(diabetes)
print(iris.target_names)
print(iris.feature_names)

df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

df['target']=iris.target
print(df.head())

df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

plt.xlabel('petal length in cm')
plt.ylabel('petal width in cm')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='*')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='*')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red',marker='*')
plt.show()

plt.xlabel('sepal length in cm')
plt.ylabel('sepal width in cm')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red',marker='+')
plt.show()

from sklearn.model_selection import train_test_split
X=df.drop(['target'],axis='columns')
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.4,random_state=1)

print(len(X_train),len(X_test),len(y_train),len(y_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)
accuracy_result=knn.score(X_test,y_test)
print(accuracy_result)

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('true')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
