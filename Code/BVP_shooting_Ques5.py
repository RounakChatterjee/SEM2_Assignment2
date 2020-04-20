'''
BVP SOLVING BY SHOOTING METHOD
====================================================================
Author: Rounak Chatterjee
Date : 13/4/2020
====================================================================
This Program solves the given Boundary value problem using 
shooting method which relies on a Guess of the first derivative
of the solution at the initial point and solving an Initial value problem
using it and then repeating this guess systematically to obtain an 
initial value of first derivative of solution at initial point that
would yield a solution matching with the boundary value at the other extreme
point.
 
Schematically if we have an equation

y'' = f(x,y,y') with boundary values y(x1) = y1 and y(x2) = y2

we consider the initial value problem vectorized as:

r' = f(r,x)

where r(x) = (y,y'), r'(x) = (y',y'') f(r,x) =(f1(r,x),f2(r,x))

with inital guess r(0) = (y1,si) where si is the initial guess

now obviously the solution y(x) will depend on s values

if y(x;s) is the solution to the IVP for some value s then we're trying 
find an s = sf which serves as a zero of the equation

y(x2,s) - y2 = 0

we will use the bisection scheme with an error bound of 1e-8 
to compute the solution.

Given equation

x''(t) = -g with boundary values 

x(0) = x(t1) = 0

thus the vectorized equation for IVP  is:
let u = x v = x'
r' = f(r,t)

where r(t) = (u,v), r'(t) = (u',v') f(r,t) =(v,-g)

with initial guess r(0) = (0,si)

si being the iniial guess 

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
g = 9.8 #Original Gravity value
ti = 0
tf = 20 # Taken higher tha t1 to compare solution.
t1 = 10
x_ti = 0.0 # initial position
x_tf = 0.0
error = 1e-8
def x(t):
	return 0.5*g*(t*t1 - t**2.0)
def f(r,t): # The vector Function
	return np.array([r[1],-g])
def Euler_Solve(s):
	h = 0.01
	t = np.arange(ti,tf,h,dtype = np.float64)
	r = np.zeros(shape = (len(t),2),dtype = np.float64)
	r[0] = np.array([x_ti,s])
	k = 0
	for i in range(len(t)-1):
		r[i+1] = r[i] + h*f(r[i],t[i])
	t_coord = int((t1-ti)/h)
	return [r[t_coord,0],[t[0:1+np.argmin(np.abs(r[1:len(t)-1,0]))],r[0:1+np.argmin(np.abs(r[1:len(t)-1,0])),0]]]

def func(s): # the function on which root solving is to be done
	return Euler_Solve(s)[0] - x_tf

root = op.newton(func,2.0)

p = np.arange(root-5.0,root+5.0,0.5,dtype = np.float64)
plt.title("Obtaining solution by shooting method", size = 15)
plt.xlabel("Time(t)")
plt.ylabel("x(t)")
plt.grid()
for i in range(len(p)):
	ar = Euler_Solve(p[i])[1]
	if(np.abs(p[i]-root)<=0.001):
		plt.plot(ar[0],ar[1],color = '#00FF00',lw = 5,label = 'True Numerical solution')
	elif(i==0):
		plt.plot(ar[0],ar[1],':',color = 'blue',label = 'Trail solutions')
	else:	
		plt.plot(ar[0],ar[1],':',color = 'blue')
plt.plot(np.linspace(0,t1,1000),x(np.linspace(0,t1,1000)),color = '#000000',label = "Analytical solution")
plt.legend()
plt.show()
