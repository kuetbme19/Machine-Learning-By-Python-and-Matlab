import matplotlib.pyplot as plt
import numpy as np

x=np.array([6,8,12,14,18])
y=np.array([350,775,1150,1395,1675])
xy=np.empty(5)
x2=np.empty(5)
y_reg=np.empty(5)

sum_1=0
sum_2=0
sum_3=0
sum_4=0
for i in range(0,len(x)):
    sum_1=sum_1+x[i]
    sum_2=sum_2+y[i]
    xy[i]=x[i]*y[i]
    sum_3=sum_3+xy[i]
    x2[i]=x[i]*x[i]
    sum_4=sum_4+x2[i]

x_bar=sum_1/len(x)
y_bar=sum_2/len(y)
print(y_bar)
xy_bar=sum_3/len(y)
x2_bar=sum_4/len(x)

m=((x_bar*y_bar)-xy_bar)/((x_bar*x_bar)-x2_bar)
print('Gradient',m)
c=y_bar-m*x_bar
print('Y axis cut',c)

for j in range(0,len(x)):
    y_reg[j]=m*x[j]+c

num=0
denum=0
for k in range(0,len(x)):
    num=num+(y_reg[k]-y_bar)*(y_reg[k]-y_bar)
    denum=denum+(y[k]-y_bar)*(y[k]-y_bar)
    r_sqr=num/denum

print('The R_squared_value',r_sqr)

plt.scatter(x,y)
plt.plot(x,y_reg)
plt.show()
