import numpy as np

def gradient_descent(x,y,iterations,learning_rate):
    m_curr=0
    b_curr=0
    n=len(x)

    for i in range(iterations):
        y_predict=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predict)])
        md=-(2/n)*sum(x*(y-y_predict))
        bd=-(2/n)*sum(y-y_predict)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("interation {} ,m {} ,b {} ,cost {} ".format(i,m_curr,b_curr,cost))
    


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y,100000,0.01)


    