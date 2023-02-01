#library importing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data set including

heart_data=pd.read_csv('data.csv')
print(heart_data)
data_head=heart_data.head()
print(data_head)
data_tail=heart_data.tail()
print(data_tail)

#size of data set
print(heart_data.shape)

#data info and missing data finding
print(heart_data.info())
print(heart_data.isnull().sum())

#statistical data

data_stat=heart_data.describe()
print(data_stat)

#value checking 
data_check=heart_data['target'].value_counts()
print(data_check)

#data spliting into train and test

X=heart_data.drop(columns='target',axis=1)
print(X)
Y=heart_data['target']
print(Y)

#spliting
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)
print(X.shape,X_train.shape,X_test.shape)

#regression
model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy

X_train_predict=model.predict(X_train)
training_data_accuracy1=accuracy_score(X_train_predict,Y_train)
print(training_data_accuracy1)

X_test_predict=model.predict(X_test)
training_data_accuracy2=accuracy_score(X_test_predict,Y_test)
print(training_data_accuracy2)

#predicting
input_data=(48,1,0,140,298,0,1,122,1,4.2,1,3,3)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)

if prediction==1:
    print("The Patient maybe has heart disease")
else:
    print("The Patient has no heart disease")

#plotting
plt.barh(heart_data['age'],heart_data['target'])
#plt.show()