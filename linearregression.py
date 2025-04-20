import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/levienbang/Downloads/Student Performance Multiple Linear Regression.csv')
df_unique = df.groupby('Hours Studied').apply(lambda x: x.sample(1)).reset_index(drop=True)


def lossfunction(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i]['Hours Studied']
        y=points.iloc[i]['Performance Index']
        total_error+=(y-(m*x+b))**2
    total_error/=float(len(points))
    return total_error
def gradient_descent(m_now,b_now,points,L):
    m_gradient=0
    b_gradient=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i]['Hours Studied']
        y=points.iloc[i]['Performance Index']
        m_gradient+= -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient+= -(2/n)*(y-(m_now*x+b_now))
    m=m_now-m_gradient*L
    b=b_now-b_gradient*L
    return m,b
m=0
b=0
L=0.01
Epoch=100000
for i in range(Epoch):
    if i%50==0:
        print(f"Epoch{i}")
    m,b=gradient_descent(m,b,df_unique,L)
    if i%50==0:
        loss = lossfunction(m, b, df_unique)
        print(m,b,loss)
    print(m,b)
    
plt.scatter(df_unique['Hours Studied'],df_unique['Performance Index'],color='black')
plt.plot(list(range(0,10)),[m*x+b for x in range(0,10)],color='red')
plt.show()