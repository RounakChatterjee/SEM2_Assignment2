'''
RUNGE KUTTA(RK) METHOD OF SOLVING SIMULTANEOUS EQUATIONS 
===============================================================
Author : Rounak Chatterjee
Date : 11/04/2020
===============================================================
In this Program we will solve the set of simultaneous first order 
Differential Equations by 4th Order RK method .

Given equations:

u1' = u1 + 2u2 - 2u3 + exp(-t)
u2' = u2 + u3 - 2exp(-t)
u3' = u1 + 2u2 + exp(-t)

thus we have the vector equation as 

r' = f(r,t), where r = (u1,u2,u3), r' = (u1',u2',u3') and r(0) = (3,-1,1)
while f(r,t) = (u1 + 2u2 - 2u3 + exp(-t),u2 + u3 - 2exp(-t),u1 + 2u2 + exp(-t))
where a = 0 and b = 1

We know the 4th Order RK is Given by the Method

if h is the step size and a<=x<=b then n = (b-a)/h with 
t_i = a+ih. Thus if w_i is solution the vector for t_i, then

k1 = h*f(w_i,t_i)
k2 = h*f(w_i+k1/2,t_i+h/2)
k3 = h*f(w_i+k2/2,t_i+h/2)
k4 = h*f(w_i+k3,t_i+h)

w_(i+1) = w_i+1/6(k1+2k2+2k3+k4)

where k1,k2,k3,k4 are intermediately computed vectors.
we'll plot all the computed  solution together.
'''

import numpy as np
import matplotlib.pyplot as plt
h = 0.01 # step value
a = 0.0 #Initial Value
b = 1.0 #Final Value


def f(r,t): #The Vector function 
	return np.array([r[0]+2.0*r[1]-2.0*r[2]+np.exp(-t),r[1]+r[2]-2.0*np.exp(-t),r[0]+2.0*r[1]+np.exp(-t)])

def next_value(wi,xi): # This Executes The RK Algorithm
	k1 = h*f(wi,xi)
	k2 = h*f(wi+k1/2.0,xi+h/2.0)
	k3 = h*f(wi+k2/2,xi+h/2.0)
	k4 = h*f(wi+k3,xi+h) 
	return (wi+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4))

n = int((b-a)/h)
x = np.arange(a,b,h)

w = np.zeros(shape=(n,3),dtype=np.float64)
w[0]=np.array([3.0,-1.0,1.0]) # Initial values

for i in range(n-1):
	w[i+1] = next_value(w[i],a+i*h)

plt.title("Plot of Solutions",size = 15)
plt.plot(x,w[:,0],color = 'blue',label = 'RK4 computed u$_1$(t)') 
plt.plot(x,w[:,1],color = 'red',label = 'RK4 computed u$_2$(t)')
plt.plot(x,w[:,2],color = 'green',label = 'RK4 computed u$_3$(t)')
plt.legend()
plt.xlabel("x",size = 13)
plt.ylabel("f(x)",size = 13)
plt.grid()
plt.show()
