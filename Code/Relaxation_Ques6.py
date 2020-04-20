'''
BVP SOLVING USING RELAXATION METHOD OR FINITE DIFFERENCE METHOD
=================================================================
Author : Rounak Chatterjee
Date : 17/04/2020
=================================================================
We analyse a very restricted class boundary value problems given by:

-y'' + q(x)y = g(x)				--------------------------------(1)
 where a<=x<=b and y(a) = ya y(b) = yb

we follow the method similar to one given in Bulrish and stoer 
section 7.4 Difference Equation where we discretize the equation as :

There it says that a unique solution to this equation exists only if 
q(x) for all x>= 0

y'' = (y_(i+1) - 2y_i + y_(i-1))/h^2 where we have divided the interval into n+1 equal
sub intervals i.e h = (b-a)/(n+1) and x_0 = a while x_(n+1) = b.

so we'll get the intermeidate interval points as y_1 to y_n with the 
boundary values as y_0 = ya and y_n+1 = yb.

So if we plug these points in the differential equation we get a set of
n linear equations that can be written in matrix vector form as:

y = Ak where

y = (y_1,...,y_n) k = (g(x_1) + ya/h^2, g_2,...,g_(n-1),g_n+yb/h^2)

Now the matrix A is a Tridiagonal matrix which has the form with 1/h^2
as a common factor, the super and sub diaginal have negative ones while 
the diaginal has form A_(ii) = 2+h^2*q(x_i)

Thus if we solve this set of linear simultaneous equations we can get a set of solution points
for the differential equation

Given Equation in the problem 

x'' = -g

if we convert this equation in the above form

-x'' = g with 0<=t<=10 and x(0) = x(10) = 0

hence we can write the functions as :

q(x) = 0 and g(x) = g

To Solve this problem we take the help of numpy.linalg.solve() function since for this case
the Matrix is a Trigiagonal matrix and the speed of solution will be considerably fast.
To get a set of candidate solution, we vary the in between points randomly
to obtain curves that satisfy the boundary condition but not the differential equation. 

Even though in other methods where the differential equation is 
not linear, the best way to solve it is by using root finding schemes via itterative techniques.
But in this case since the form of the Matrix is simplier, we just use simple
Solving using matrix Algebra  like in this case we use LU decomposition.

To generate other candidate solution the best Idea is to change the form of the function g(x), in this case e can change the value of
g_value to obtain other solutions, but that would be just for cosmetic purpouse, the true solution is solution from the Matrix equation solving
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
a = 0.0
b = 10.0
g_val = 10.0
x_a = x_b = 0.0 # Boundary Values
n = 99 
h = (b-a)/(n+1)
t = np.delete(np.arange(a,b,h),0) # this creates the time mesh points between a and b
def x(t): # True Solution
    return 0.5*g_val*(t*b - t**2.0)

def q(x): #This step is unnecessary but upholds the generality of the problem at hand.
    return 0.0
def g(x):
    return g_val

def Create_A(t):
    sup = -1.0*np.ones(len(t)-1,dtype = np.float64)
    sub = -1.0*np.ones(len(t)-1,dtype = np.float64)
    dig = np.zeros(len(t),dtype = np.float64)
    dig[:] = 2.0+q(t[:])*h**2.0
    A = diags((sub,dig,sup),offsets=(-1,0,1),shape=(len(t),len(t)),dtype=np.float64)
    A = A.toarray()*(1/h**2.0)
    return A
def create_k(t):
    k = np.zeros(len(t),dtype = np.float64)
    k[1:len(t)-2] = g(t[1:len(t)-2])
    k[0] = g(t[0])+x_a/h**2.0
    k[len(t)-1] = g(t[len(t)-1]) + x_b/h**2.0
    return k
# Creating the Two matrices
A = Create_A(t)
k = create_k(t)
true_sol = np.linalg.solve(A,k)
t_full = np.insert(t,0,a)
true_sol = np.insert(true_sol,0,x_a)
t_full = np.insert(t_full,len(t_full),b)
true_sol = np.insert(true_sol,len(true_sol),x_b)
x_cand_sol = np.zeros(shape = len(true_sol),dtype = np.float64)
x_cand_sol[0] = x_a
x_cand_sol[len(true_sol)-1] = x_b
val = [9.0,11.0,12.5,8.5,7.0,13.0,9.5,14.2]
plt.title("Solution of BVP by Relaxation method")
plt.plot(t_full,true_sol,color = '#00FF00',lw = 4,label = "Obtained Numerical Solution")
plt.plot(np.linspace(a,b,1000),x(np.linspace(a,b,1000)),color = '#000000',label = "Analytical solution")
for i in range(len(val)):
    g_val = val[i]
    A = Create_A(t)
    k = create_k(t)
    x_cand_sol[1:n+1] = np.copy(np.linalg.solve(A,k))
    if(i==0):
        plt.plot(t_full,x_cand_sol,':',color = 'red',label = "Candidate Solutions")
    else:
         plt.plot(t_full,x_cand_sol,':',color = 'red')       
    g_val = 10.0
plt.grid()
plt.xlabel("x(t)",size = 13)
plt.ylabel("t",size = 13)
plt.legend()
plt.show()




