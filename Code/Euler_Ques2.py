'''
SOLUTION USING EULER METHOD
==============================================================
Author: Rounak Chatterjee
Date : 30/03/2020
==============================================================

This Program Does The Euler Integration on a Given First Order Initial 
Value problem

The equation is:

dy/dt = y/t - (y/t)^2 = f(y,t)

with 1<=t<=2 and y(1) = 1.0

We know the Euler Scheme is given by:

w_(j+1) = w_j + h*f(w_j,t_j)

where w_0 = y(1) and h is the step size such that
t_j = a+hj
'''
import numpy as np
import matplotlib.pyplot as plt


def y(t):#True Solution
	return t/(1.0+np.log(t))
def f(y,t):
	return (y/t) - (y/t)**2.0
def error(true,evaluated,er_type):
	if(er_type == True):
		return np.abs(true - evaluated)
	elif(er_type == False):
		return np.abs(true - evaluated)/float(true)

h = 0.1 # The step size we want to use (Made Global)
a = 1.0 #Lower Bound
b = 2.0 #Upper Bound
n = int((b-a)/h) # Steps
x = np.arange(a,b,h)
w = np.zeros(shape = n,dtype = np.float64)
abs_err = np.zeros(n,dtype = np.float64)
r_error = np.zeros(n,dtype = np.float64)
w[0] = 1.0 #Initual Value
abs_err[0] = 0.0
r_error[0] = 0.0
y 
for i in range(n-1):
	w[i+1] = w[i] + h*f(w[i],a+i*h)
	abs_err[i+1] = np.round(error(y(a+(i+1)*h),w[i+1],True),decimals = 4)
	r_error[i+1] = np.round(error(y(a+(i+1)*h),w[i+1],False),decimals =4)
print("Data Table:")
print("evaluated 	True 	Absolute error 		relative error")
for i in range(n):
	print(np.round(w[i],decimals = 4),"	",np.round(y(a+i*h),decimals = 4),"		",abs_err[i],"		",r_error[i])

plt.plot(x,w,color = 'blue',label = 'numerical solution')
plt.plot(x,y(x),color = 'red',label = 'true solution')
plt.legend(frameon = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.show()