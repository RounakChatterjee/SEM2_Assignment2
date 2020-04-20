'''
BACKWARD EULER INTEGRATION
======================================================================================================
Author: Rounak Chatterjee
Date : 30/03/2020
=======================================================================================================
To solve the equation using backward or Explicit Euler method, what we can do is use the Newton Raphson 
root finding scheme to find the w_(j+1) value. But what should be our initial guess for w_(j+1). We can expect that w_(j+1)
obtained from an explicit euler might serve as a good initial guess to thw w_(j+1) we want to find.

thus our scheme stands as :
Identify equation and dy/dx = f(y,x) with t belonging to (a,b) and y(a) = aplha

thus if h is the step size then w_0 = aplha 
w_guess_(j+1) = w_j + h.f(w_j,x_j)

w_true_(j+1) = root computed using Newton Raphson using guess value w_guess_(j+1)

so w_(j+1) = w_true_(j+1)

So we can make an array of the numerically computed solution
and plot it against the true solution

Since Doing Newton Raphson requires explicit knowledge of the derivative 
We won't generalize this scheme in this program but will code it to
problem specife.

We have dy/dx = -20(y-x)^2+2x with 0<=x<=1 and y(0) = 1/3

so the Newton Raphson equation we're after to solve is of the form

w_j+h.f(y,x_(j+1)) - y = 0
for our case it is

w_j + -20h(y-x_(j+1))^2+2x_(j+1)- y = 0 = g(y)(say)

g'(y) = -40h(y-x_(j+1))-1.0

we use g(y) and g'(y) = Dg(y) in a newton Raphson scheme

'''

import numpy as np
import matplotlib.pyplot as plt
h = 0.001 # The step size we want to use (Made Global)
a = 0.0 #Lower Bound
b = 1.0 #Upper Bound
n = int((b-a)/h) # Steps

def f(y,x):
	return -20.0*(y-x)**2+2.0*x

def g(y,wj,xj1):
	return wj-20.0*h*(y-xj1)**2.0+2.0*xj1- y

def Dg(y,xj1):
	return -40.0*h*(y-xj1)-1.0

def Find_Root_NR(y_guess,wj,xj1):
	error = 1.0e-4 # Error Threshold
	y = y_guess #initial start
	while(np.abs(g(y,wj,xj1))>error):
	 	y = y-(g(y,wj,xj1)/Dg(y,xj1))
	return y
x = np.arange(a,b,h)
w = np.zeros(shape = n,dtype = np.float64)
w[0] = 1.0/3.0 #Initual Value
for i in range(n-1):
	w[i+1] = w[i] + h*f(w[i],a+i*h) # w_guess_(j+1)
	w[i+1] = Find_Root_NR(w[i+1],w[i],a+(i+1)*h)
plt.plot(x,w,color = 'blue',label = 'numerical solution')
#plt.plot(x,y(x),color = 'red',label = 'true solution')
plt.legend(frameon = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.show()





