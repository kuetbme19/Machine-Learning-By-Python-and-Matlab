import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
breast=load_breast_cancer()

print(breast)

print(breast.feature_names)
print(breast.target_names)
df=pd.DataFrame(breast.data,columns=breast.feature_names)
print(df.head())

df['target']=breast.target
print(df.head())

print(df[df.target==1])
print(df.shape)

name= df['Disease_name']=df.target.apply(lambda x:breast.target_names[x])
print(df)

from sklearn.model_selection import train_test_split
X=df.drop(['Disease_name','target'],axis='columns')
y=df.target

X_train, X_test, y_train,y_test=train_test_split(X,y,train_size=0.7)
print(len(X_train),len(X_test),len(y_train),len(y_test))

from sklearn.svm import SVC
model=SVC()
model.fit(X_train, y_train)
accuracy=model.score(X_test,y_test)
print(accuracy)




