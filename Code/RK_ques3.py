'''
RUNGE KUTTA(RK) METHOD OF SOLVING SECOND ORDER ODE
===============================================================
Author : Rounak Chatterjee
Date : 31/03/2020
===============================================================
In this Program we will solve the second order Differential Equation by converting it into a set of first order differential eqaution 
and then using the vector form of 4th Order RK method to solve for it.

Given equation:

y''-2y'+y = xe^x-x, with 0<=x<=1 and y(0) = 0 and y'(0) = 0

we can convert it into a vector equation as

let u = y and v = y', then
u' = v and v' = 2v-u+xe^x-x with u(0) = v(0) = 0

thus we have the vector equation as 

r' = f(r,x), where r = (u,v), r' = (u',v') and r(0) = (0,0)
while f(r,x) = (v,2v-u+xe^x-x)

We know the 4th Order RK is Given by the Method

if h is the step size and a<=x<=b then n = (b-a)/h with 
x_i = a+ih. Thus if w_i is solution the vector for x_i, then

k1 = h*f(w_i,x_i)
k2 = h*f(w_i+k1/2,x_i+h/2)
k3 = h*f(w_i+k2/2,x_i+h/2)
k4 = h*f(w_i+k3,x_i+h)

w_(i+1) = w_i+1/6(k1+2k2+2k3+k4)

where k1,k2,k3,k4 are intermediately computed vectors.

Once we have solution, then u = y is our true soution to the equation.

Mathematica suggests the true solution to the IVP as 

-2-x+ 1/6 e^x(12-6x+x^3)

we'll plot it against the computed solution.
'''

import numpy as np
import matplotlib.pyplot as plt
h = 0.01 # step value
a = 0.0 #Initial Value
b = 1.0 #Final Value

def y(x): #True Solution
	return (-2.0-x+1.0/6.0*np.exp(x)*(12.0-6.0*x+x**3.0))

def f(r,x): #The Vector function 
	return np.array([r[1],2.0*r[1]-r[0]+x*np.exp(x)-x])

def next_value(wi,xi): # This Executes The RK Algorithm
	k1 = h*f(wi,xi)
	k2 = h*f(wi+k1/2.0,xi+h/2.0)
	k3 = h*f(wi+k2/2,xi+h/2.0)
	k4 = h*f(wi+k3,xi+h) 
	return (wi+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4))

n = int((b-a)/h)
x = np.arange(a,b,h)

w = np.zeros(shape=(n,2),dtype=np.float64)
w[0,:]=[0.0,0.0] # Initial values

for i in range(n-1):
	w[i+1] = next_value(w[i],a+i*h)

plt.title("Comparison of NUmerical Solution by RK with True solution")
plt.scatter(x,w[:,0],marker = 'x',color = 'blue',label = 'RK computed y(x)') 
# The 0th element is the variable u = y
plt.plot(x,y(x),color = 'red',label = 'True Solution y(x)')
plt.legend()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid()
plt.show()
