import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import  matplotlib.colors as mcolors
import random
import math
import time
import datetime
import operator

from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#loading three dataset

confirm=pd.read_csv('confirm.csv')
death=pd.read_csv('death.csv')
recover=pd.read_csv('recover.csv')

print(confirm.head())
print(death.head())
print(recover.head())

#extracting all the columns

cols=confirm.keys()
print(cols)

#extracting only the date columns

confirmed =confirm.loc[:,cols[4]:cols[-1]]
deathed =death.loc[:,cols[4]:cols[-1]]
recovered =recover.loc[:,cols[4]:cols[-1]]

print(confirmed.head())

#finding cases anatomy
dates= confirmed.keys()
print(dates)
world_cases=[]
total_deaths=[]
mortality_rate=[]
total_recovered=[]

for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deathed[i].sum()
    recovered_sum=recovered[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)

print(confirmed_sum)
print(death_sum)
print(recovered_sum)
print(world_cases)
print(total_deaths)
print(mortality_rate)
print(total_recovered)

#creating numpy library with numpy array

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
print(days_since_1_22)
world_cases=np.array(world_cases).reshape(-1,1)
print(world_cases)
total_deaths=np.array(total_deaths).reshape(-1,1)
print(total_deaths)
total_recovered=np.array(total_recovered).reshape(-1,1)
print(total_recovered)


#future forecasting for the next 10 days

days_in_future=10
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forecast[:-10]
print(future_forecast)
print(adjusted_dates)

#convert all the integers into datetime for better visualization

start='1/22/2020'
start_date=datetime.datetime.strptime(start,'%m/%d/%Y')
print(start_date)
future_forecast_dates=[]

for i in range (len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    print(future_forecast_dates)

#for visualization with the latest data of 15th of march

latest_confirmed=confirm[dates[-1]]
latest_deaths=death[dates[-1]]
latest_recovered=recover[dates[-1]]

print(latest_confirmed)
print(latest_deaths)
print(latest_recovered)

#find the list of unique countries
unique_countries =list(confirm['Country/Region'].unique())
print(unique_countries)

#total case of each country

country_confirmed_case=[]
no_case=[]

for i in unique_countries:
    cases=latest_confirmed[confirm['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_case.append(cases)
    else:
        no_case.append(i)

print(country_confirmed_case)

for i in no_case:
     unique_countries.remove(i)


unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_case),key=operator.itemgetter(1),reverse=True)]
print(unique_countries)

for i in range(len(unique_countries)):
    country_confirmed_case[i]=latest_confirmed[confirm['Country/Region']==unique_countries[i]].sum()

for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}:{country_confirmed_case[i]} cases')

#unique provience 

unique_provience=list(confirm['Province/State'].unique())
print(unique_provience)

outliers=['United Kingdom','Denmark','France']

for i in outliers:
    print(unique_provience.remove(i))

provience_confirmed_case=[]
no_case=[]

for i in unique_provience:
    cases=latest_confirmed[confirm['Province/State']==i].sum()

    if cases>0:
        provience_confirmed_case.append(cases)
    else:
        no_case.append(i)
print(provience_confirmed_case)
print(no_case)

for i in no_case:
    unique_provience.remove(i)


for i in range(len(unique_provience)):
    print(f'{unique_provience[i]} : {provience_confirmed_case[i]} cases')

nan_indices=[]

for i in range(len(unique_provience)):
    if type(unique_provience[i])==float:
        nan_indices.append(i)
unique_provience=list(unique_provience)
provience_confirmed_case=list(provience_confirmed_case)

for i in nan_indices:
    unique_provience.pop(i)
    provience_confirmed_case.pop(i)
    
#plotting a bar graph 

plt.figure(figsize=(5,5))
#plt.barh(unique_countries,country_confirmed_case)
#plt.show()

#plotting plot comparing between chaina mainland and outside mainland

china_confirmed=latest_confirmed[confirm['Country/Region']=='China'].sum()
print(china_confirmed)
outside_china=np.sum(country_confirmed_case)-china_confirmed
print(outside_china)
#plt.barh('Mainland China',china_confirmed)
#plt.barh('Outside China',outside_china)
#plt.show()

#printing result

print('outside china',(outside_china))
print('inside china',(china_confirmed))
print('total case ',outside_china+china_confirmed)

#10 countries
visiual_unique_countries=[]
visiual_confirmed_cases=[]
others=np.sum(country_confirmed_case[10:])
print(others)

for i in range(len(country_confirmed_case[:10])):
    visiual_unique_countries.append(unique_countries[i])
    visiual_confirmed_cases.append(country_confirmed_case[i])
    print(visiual_unique_countries)

visiual_unique_countries.append('Others')
visiual_confirmed_cases.append(others)

plt.barh(visiual_unique_countries,visiual_confirmed_cases)
plt.show()

#creating pie chart

c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.pie(visiual_confirmed_cases,colors=c)
plt.legend(visiual_unique_countries,loc='best')
plt.show()

#10 countries outside china

c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.pie(visiual_confirmed_cases[1:],colors=c)
plt.legend(visiual_unique_countries[1:],loc='best')
plt.show()

#spliting data set
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)

#builing  the SVM model

kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
svm_search.fit(X_train_confirmed, y_train_confirmed)

svm_search.best_params_

svm_confirmed = svm_search.best_estimator_
svm_pred = svm_confirmed.predict(future_forecast)

print(svm_confirmed)
print(svm_pred)

svm_test_pred=svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)

plt.show()

print('MAE=',mean_absolute_error(svm_test_pred,y_test_confirmed))
print('MSE=',mean_squared_error(svm_test_pred,y_test_confirmed))


# total number of corona virus over time

plt.plot(adjusted_dates,world_cases)
plt.xlabel('Days Since 1/22/2020',size=30)
plt.ylabel('Number of Cases ',size=30)
plt.show()

#confirmed vs predicted cases

plt.plot(adjusted_dates,world_cases)
plt.plot(future_forecast,svm_pred,linestyle='dashed',color='purple')
plt.xlabel('Days since 1/22/2020',size=30)
plt.ylabel('Number of Cases',size=30)
plt.show()

#predictions for the next 10 days using SVM

print('SVM future predictions :')
print(set(zip(future_forecast_dates[-10:],svm_pred[-10:])))

#using Linear regression model to make predictions

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(X_test_confirmed)
linear_pred = linear_model.predict(future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.show()

plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forecast, linear_pred, linestyle='dashed', color='orange')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

model=LogisticRegression()
model.fit(X_train_confirmed,y_train_confirmed)

#accuracy

X_train_predict=model.predict(X_train_confirmed)
training_data_accuracy1=accuracy_score(X_train_predict,y_train_confirmed)
print(training_data_accuracy1)

X_test_predict=model.predict(X_test_confirmed)
training_data_accuracy2=accuracy_score(X_test_predict,y_test_confirmed)
print(training_data_accuracy2)

mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])
plt.xlabel('Time', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# Coronavirus Deaths vs Recoveries

plt.figure(figsize=(20, 12))
plt.plot(total_recovered, total_deaths)
plt.title('Coronavirus Deaths vs Coronavirus Recoveries', size=30)
plt.xlabel('Total number of Coronavirus Recoveries', size=30)
plt.ylabel('Total number of Coronavirus Deaths', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()