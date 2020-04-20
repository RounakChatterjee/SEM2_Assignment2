'''
SOLVING BVP BY USING SCIPY'S SOLVE_BVP
=====================================================================
Author : Rounak Chatterjee
Date : 20/4/2020
=====================================================================
In this program we aim to solve the given boundary value problems using
the scipy.integrate.solve_bvp module.

We solve the Boundary Value Problems on Mathematica to obtai analytical form
which we quote here. We Solve them using solve_bvp() and compare the analytical 
to computed solution. But Most Equations here are non-linear Equations
which are most generally solved Numerically. Softwares like Mathematica cannot also
come up with symbolic solution for these equations.

to solve these equations we must vectorie them, we do them as:
for every case we take u = y and v = y'
(i)
Eqn:
y'' = -exp(-2y) with 1<=x<=2, y(1) = 0, and y(2) = ln(2)
Vector Eqn:
u' = v, v' = -exp(-2u)


(ii)
Eqn:
y'' = y'cos(x)- yln(y) with 0<=x<=2, y(0) = 1, and y(Pi/2) = e
Vector Eqn:

u' = v, v' = vcos(x)-uln(u)

(iii)
Eqn:
y''= -(2(y')^3 + y^2y')sec(x) with Pi/4<=x<=Pi/3, y(Pi/4) = 2^(-1/4), and y(Pi/3) =12^(1/4)/2
Vector Eqn:
u'=v, v' = -(2(v)^3 + u^2v)sec(x)

(iv)
Eqn:
y'' = 1/2-(y')^2/2-ysin(x/2) with 0<=x<=Pi, y(0) = 2, and y(Pi) = 2
Vector Eqn:
u'= v, v'= 1/2-(v)^2/2-usin(x/2)

'''

import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
a=np.array([1.0,0.0,np.pi/4.0,0.0])
b=np.array([2.0,np.pi/2.0,np.pi/3,np.pi])
y_a = np.array([0.0,1.0,1.0/2**(0.25),2.0])
y_b	= np.array([np.log(2.0),np.exp(1.0),(12.0**0.25)/2.0,2.0])
def fun1(x,y): 
    return np.vstack((y[1],-np.exp(-2*y[0])))

def fun2(x,y):
	return np.vstack((y[1],y[1]*np.cos(x)-y[0]*np.log(y[0])))

def fun3(x,y):
	return np.vstack((y[1],-(2.0*y[1]**3.0 + y[0]**2.0*y[1])*(1/np.cos(x))))	

def fun4(x,y):
	return np.vstack((y[1],0.5-y[1]**2.0/2.0-y[0]*np.sin(x/2.0)))

def bc1(ya,yb):        
	return np.array([ya[0]-y_a[0],yb[0]-y_b[0]])

def bc2(ya,yb):
	return np.array([ya[0]-y_a[1],yb[0]-y_b[1]])

def bc3(ya,yb):
	return np.array([ya[0]-y_a[2],yb[0]-y_b[2]])

def bc4(ya,yb):
	return np.array([ya[0]-y_a[3],yb[0]-y_b[3]])

x1 = np.linspace(a[0],b[0],200)
x2 = np.linspace(a[1],b[1],200)
x3 = np.linspace(a[2],b[2],200)
x4 = np.linspace(a[3],b[3],200)

y1=np.zeros((2,x1.size))
y2=np.zeros((2,x2.size))
y3=np.zeros((2,x3.size))
y4=np.zeros((2,x4.size))

y1[0]=1.0 #initial guess of the solution
y2[0]=1.0
y3[0]=1.0
y4[0]=1.0

sol1=solve_bvp(fun1,bc1,x1,y1)
sol2=solve_bvp(fun2,bc2,x2,y2)
sol3=solve_bvp(fun3,bc3,x3,y3)
sol4=solve_bvp(fun4,bc4,x4,y4)

fig = plt.figure(constrained_layout=False)
fig.suptitle("Computed Solutions",size = 15)
spec = fig.add_gridspec(2,2)

p1 = fig.add_subplot(spec[0,0])
p1.set_title("Ques (i)",size = 13)
p1.plot(x1,sol1.sol(x1)[0],label="Numerical solution")
p1.legend()
p1.grid()
p1.set_ylabel("y(x)",size = 13)

p2 = fig.add_subplot(spec[0,1])
p2.set_title("Ques (ii)",size = 13)
p2.plot(x2,sol2.sol(x2)[0],label="Numerical solution")
p2.legend()
p2.grid()
p2.set_ylabel("y(x)",size = 13)

p3 = fig.add_subplot(spec[1,0])
p3.set_title("Ques (iii)",size = 13)
p3.plot(x3,sol3.sol(x3)[0],label="Numerical solution")
p3.legend()
p3.grid()
p3.set_xlabel("x",size = 13)
p3.set_ylabel("y(x)",size = 13)

p4 = fig.add_subplot(spec[1,1])
p4.set_title("Ques (iv)",size = 13)
p4.plot(x4,sol4.sol(x4)[0],label="Numerical solution")
p4.legend()
p4.grid()
p4.set_xlabel("x",size = 13)
p4.set_ylabel("y(x)",size = 13)

fig.show()
plt.show()

