'''
SOLVING EQUATION WITH scipy.integrate.solve_ivp
====================================================================
Author : Rounak Chatterjee
Date : 31/03/2020
====================================================================
This Program solves various equation with the package from scipy 
scipy.integrate.solve_ivp()

To justify its correctness we plot the solution against their true analytical
solution computed using Mathematica

The Four IVP equations and their respective analyytical solutions are:

1) y' = te^(3t) - 2y with 0<=t<=1 and y(0) = 0

soln: 1/25 e^(-2t)(1 + e^(5t)(-1+5t))

2) y' = 1 - (t - y)^2 with 2<=t<=3 and y(2) = 0

soln: (2-5t+2t^2)/(-5+2t)

3) y' = 1 + y/t with 1<=t<=2 and y(1) = 2

soln: t(2+ln(t))

4) y' = cos(2t) + sin(3t) with 0<=t<=1 and y(0) = 1

soln: 1/6(8-2cos(3t) + 3sin(2t))

'''
import numpy as np
import scipy.integrate as it
import matplotlib.pyplot as plt

def func1(t,y):
	return t*np.exp(3.0*t) - 2.0*y

def y1(t):
	return 1.0/25.0*np.exp(-2.0*t)*(1.0 + np.exp(5.0*t)*(-1.0+5.0*t))

def func2(t,y):
	return (1.0-(t-y)**2.0)

def y2(t):
	return (2.0-5.0*t+2.0*t**2.0)/(-5.0+2.0*t)

def func3(t,y):
	return (1.0 + y/t)

def y3(t):
	return t*(2.0+np.log(t))

def func4(t,y):
	return (np.cos(2.0*t) + np.sin(3.0*t))

def y4(t):
	return (1.0/6.0)*(8.0-2.0*np.cos(3.0*t) + 3.0*np.sin(2.0*t))
sol1 = it.solve_ivp(func1,np.array([0.0,1.0]),[0.0])
sol2 = it.solve_ivp(func2,np.array([2.0,3.0]),[1.0])
sol3 = it.solve_ivp(func3,np.array([1.0,2.0]),[2.0])
sol4 = it.solve_ivp(func4,np.array([0.0,1.0]),[1.0])

fig = plt.figure(constrained_layout=False)
fig.suptitle("Plots of computed functions vs original solution",size = 15)
spec = fig.add_gridspec(2,2)
plot1 = fig.add_subplot(spec[0,0])
plot1.set_title("Ques (i)")
plot1.set_ylabel("y(t)",size = 13)
plot1.plot(sol1.t,sol1.y[0],'ro',color = 'red',label = 'computed')
plot1.plot(np.linspace(0.0,1.0,1000),y1(np.linspace(0.0,1.0,1000)),color = 'blue',label = 'Analytical')
plot1.legend()
plot1.grid()

plot2 = fig.add_subplot(spec[0,1])
plot2.set_ylabel("y(t)",size = 13)
plot2.set_title("Ques (ii)")
plot2.plot(sol2.t,sol2.y[0],'ro',color = 'red',label = 'computed')
plot2.plot(np.linspace(2.0,3.0,1000),y2(np.linspace(2.0,3.0,1000)),color = 'blue',label = 'Analytical')
plot2.legend()
plot2.grid()

plot3 = fig.add_subplot(spec[1,0])
plot3.set_ylabel("y(t)",size = 13)
plot3.set_xlabel("t",size = 14)
plot3.set_title("Ques (ii)")
plot3.plot(sol3.t,sol3.y[0],'ro',color = 'red',label = 'computed')
plot3.plot(np.linspace(1.0,2.0,1000),y3(np.linspace(1.0,2.0,1000)),color = 'blue',label = 'Analytical')
plot3.legend()
plot3.grid()

plot4 = fig.add_subplot(spec[1,1])
plot4.set_title("Ques (ii)")
plot4.set_ylabel("y(t)",size = 13)
plot4.set_xlabel("t",size = 14)
plot4.plot(sol4.t,sol4.y[0],'ro',color = 'red',label = 'computed')
plot4.plot(np.linspace(0.0,1.0,1000),y4(np.linspace(0.0,1.0,1000)),color = 'blue',label = 'Analytical')
plot4.legend()
plot4.grid()

fig.show()
plt.show()

'''
Obviously solve_ivp() uses an adaptive stepsize to solve the given problems
'''








