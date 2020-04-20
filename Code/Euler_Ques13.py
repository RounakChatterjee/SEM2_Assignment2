'''
SOLUTION USING EULER METHOD
==============================================================
Author: Rounak Chatterjee
Date : 08/04/2020
==============================================================

This Program Does The Euler Integration on a Given First Order Initial 
Value problem

The equation is:

t^2y'' - 2ty'+2y = t^3ln(t)

we re-write it as:

y''-(2/t)y' + (2/t^2)y = tln(t)

with  1(=a)<=t<=2(=b) and y(1.0) = 1.0 y'(1.0) = 0

we can vectorize the equation as:
 u = y and v = y'
 then the set of functions we have are
u' = v 								----------------------(1)
v' = (2/t)v - (2/t^2)u + tln(t)		----------------------(2)

thus the solution vector is

r(t) = (u,v) r'(t) = (u',v') and f(r,t) = (v,(2/t)v - (2/t^2)u + tln(t))

r(1) = (u(1) = y(1) = 1.0,v(1) = y'(1) = 0.0)
We know the Euler Scheme is given by:

w_(j+1) = w_j + h*f(w_j,t_j)

where w_i is a 2d vector with w_0 = (1.0,0.0) and h is the step size such that
t_j = a+hj
'''
import numpy as np
import matplotlib.pyplot as plt


def y(t):#True Solution
	return 7.0*t/4.0 + (t**3.0/2.0)*np.log(t)-(3.0/4.0)*t**3.0
def f(r,t):
	return np.array([r[1],(2.0/t)*r[1] - (2.0/t**2.0)*r[0] + t*np.log(t)])

h = 0.001 # The step size we want to use (Made Global)
a = 1.0 #Lower Bound
b = 2.0 #Upper Bound
n = int((b-a)/h) # Steps
x = np.arange(a,b,h)
w = np.zeros(shape = (n,2),dtype = np.float64)
w[0] = [1.0,0.0] #Initual Value
 
for i in range(n-1):
	w[i+1] = w[i] + h*f(w[i],a+i*h)

plt.plot(x,w[:,0],lw = 5,color = '#00FF00',label = 'numerical solution')
plt.plot(x,y(x),color = 'black',lw = 2,label = 'true solution')
plt.legend(frameon = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.show()