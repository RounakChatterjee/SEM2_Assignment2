'''
SOLVING EQUATION WITH INIFINTE LIMIT USING RK4
===================================================================
Author: Rounak Chatterjee
Date: 15/04/2020
===================================================================
To Solve the given equation:

dx/dt = 1/(x^2+t^2) for 0<t<inf and x(0) = 1

we will use an ADAPTIVE 4th ORDER RK method, one similar to queston 9.
Since the question wants us to find the value of solution at 
t = 3.5 x 10^6, we will explicitly make arrangement for the Adaptive 
scheme to choose this point while computing the solution over a range
0 to 10^9 which should be sufficinetly enough to give the asymptotic 
approximation of the solution as t --> inf.

This Program solves the given differential equation with the help of 
an adaptive step rk4 scheme.

Given first order equation:
x' = 1/(x^2+t^2) = f(x,t)
with 1<=t<inf and x(0) = 1.0 

Adaptive step size relies on the test condition:
if e is the permisible error and and if y1 and y2 are the values of the 
fuction at some t+2h such that for y1 it was obtained via 
t --> t+h --> t+2h and for y2 it was obtained via t ---> t+2h, where h is 
the current step size. Then the optimal stepsize to obtain an error bound e is

h' = (30.0*h*e/(|y1-y2|))^0.25 * h

thus for any necessary step, this is the optimal step size.
We had already executed RK4 Schemes in previous sections and will be
using the same algorithms to solve these.

While solving the problem we find that the solution very quickly 
saturates to a constant value almost as early as t = 30
so we make 
'''

import numpy as np
import matplotlib.pyplot as plt

a = 0.0 #Initial Value
b = 1.0e7 #Final Value, Asymptotic of t = inf

def f(x,t):  
	return 1.0/(x**2.0+t**2.0)

def RK4_value(w,x,h): #This Executes The RK Algorithm at specified t and h
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
x_point_calculate = 3.500e6
index_of_x = 0
t = np.zeros(shape = 1,dtype = np.float64)
w = np.zeros(shape= 1,dtype=np.float64)
k = 0
t[k] = a
w[k] = 1.0 # Initial values
h = 0.1 # initial step 
error = 1.0e-4 # error bound
flag = False
point_flag = False
while(flag == False):
	h = h_approipate(w[k],t[k],h,error)
	if(t[k]+h>x_point_calculate and point_flag == False):
		h = (x_point_calculate-t[k])
		point_flag = True
		index_of_x = k+1
	elif(t[k]+h>b):
		h = (b-t[k])
		flag =True
	w = np.append(w,RK4_value(w[k],t[k],h))
	t = np.append(t,t[k]+h)
	k = k+1
print("Value at t = ",x_point_calculate," is = ",w[index_of_x])

'''
From the analysis of the solution we can find that the solution saturtes after it crosses 
30, to make the trend more visible we have made two plots.
The first plot plots till 1.0e7 with poing 3.5e6 marked with its value quoted
the second graph plots till t = 30 which brings out the essential feature
in the solution
'''
fig = plt.figure(constrained_layout=False)
fig.suptitle("Solution of the Problem with ADAPTIVE RK4",size = 15)
spec = fig.add_gridspec(1,2)
plot1 = fig.add_subplot(spec[0,0])
plot1.set_title("Full Scope of Solution",size = 13)
plot1.set_xlabel("t")
plot1.set_ylabel("x(t)")
plot1.scatter(t,w,color = 'red',label = 'Adaptive RK4')
plot1.scatter([t[index_of_x]],[w[index_of_x]],color = 'blue',label = "The point at 3.5 x 10$^6$")
plot1.legend()
plot1.grid()

plot2 = fig.add_subplot(spec[0,1])
plot2.plot(t,w,'ro-',color = 'red',label = 'Adaptive RK4')
plot2.set_xlim([-0.50,40.0])
plot2.set_title("Region of Essential change",size = 13)
plot2.set_xlabel("t")
plot2.set_ylabel("x(t)")
plot2.legend()
plot2.grid()

fig.show()
plt.show()
