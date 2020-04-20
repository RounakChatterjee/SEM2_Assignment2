'''
RK4 WITH ADAPTIVE STEP SIZE
=====================================================================
Author : Rounak Chatterjee
Date : 11/04/2020
=====================================================================
This Program solves the given differential equation with the help of 
an adaptive step rk4 scheme.

Given first order equation:
y' = (y^2+y)/t = f(y,t)
with 1<=t<=3 and y(1) = -2 

Adaptive step size relies on the test condition:
if e is the permisible error and and if y1 and y2 are the values of the 
fuction at some t+2h such that for y1 it was obtained via 
t --> t+h --> t+2h and for y2 it was obtained via t ---> t+2h, where h is 
the current step size. Then the optimal stepsize to obtain an error bound e is

h' = (30.0*h*e/(|y1-y2|))^0.25 * h

thus for any necessary step, this is the optimal step size.
We had already executed RK4 Schemes in previous sections and will be
using the same algorithms to solve these.
'''

import numpy as np
import matplotlib.pyplot as plt

a = 1.0 #Initial Value
b = 3.0 #Final Value
def y(t): # Analytical solution
	return t/(0.5-t)

def f(y,t):  
	return (y**2.0+y)/t

def RK4_value(w,x,h): # This Executes The RK Algorithm at specified t and h
	k1 = h*f(w,x)
	k2 = h*f(w+k1/2.0,x+h/2.0)
	k3 = h*f(w+k2/2,x+h/2.0)
	k4 = h*f(w+k3,x+h) 
	return (w+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4))

def h_approipate(w,x,h,error): # finds the Appropiate value of h at every step
	y1 = RK4_value(RK4_value(w,x,h),x+h,h)
	y2 = RK4_value(w,x,2.0*h)
	r = (30.0*h*error)/np.abs(y1-y2)
	return r**0.25*h

x = np.zeros(shape = 1,dtype = np.float64)
w = np.zeros(shape= 1,dtype=np.float64)
k = 0
x[k] = a
w[k] = -2.0 # Initial values
h = 0.01 # initial step 
error = 1.0e-4 # error bound
flag = False
while(flag == False):
	h = h_approipate(w[k],x[k],h,error)
	if(x[k]+h>b):
		h = (b-x[k])
		flag =True
	w = np.append(w,RK4_value(w[k],x[k],h))
	x = np.append(x,x[k]+h)
	k = k+1

plt.title("Solution using Adaptive RK4",size = 15)
plt.plot(np.linspace(1.0,3.0,500),y(np.linspace(1.0,3.0,500)),color = 'blue',label = 'Analytical Solution')
plt.scatter(x,w,color = 'red',label = 'Adaptive RK4')
plt.legend()
plt.xlabel("t",size = 13)
plt.ylabel("y(t)",size = 13)
plt.grid()
plt.show()