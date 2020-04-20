/*
EULER INTEGRATION SCHEME USING C
========================================================================================
Author :  Rounak Chatterjee
Date : 19/04/2020
========================================================================================
This code executes the Euler integration scheme but written in C language.
We know the Euler Integration scheme as:

w_(j+1) = w_j + h*f(w_j,t_j)

where a<=t<= b and w_0 = y(a) and h is the step size such that
t_j = a+hj

where the 1st order ODE is of form y'(t) = f(y,t)

Given Differential Equation

y' = y-t^2+1 = f(y,t) with 0<=t<=2 and y(0) = 0.5

The exact solution to this ODE is

y(t) = (t+1)^2 - 0.5exp(t)

Now to find the error bound we know that it is given by

for each step i

|y_i-w_i|<= hM/2L(exp(L(t_i-a))-1)

where L = the bound of |delf/dely| which can be found to be = 1
and M is the bound of |y''| and since we know the analytical form of y(t)
we can find this as M = (0.5e^2 - 2)

since a = 0

we can write the bound as :

|y_i-w_i| <= h/2*(0.5e^2 - 2)*(exp(t_i)-1) for each step i

in the program we take h = 0.2 and evaluate the original error and the error
bound at each step.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
float y(float t)
{
	return (t+1.0)*(t+1.0) - 0.5*exp(t);
}
float error_bound(float t,float h)
{
	return (h/2.0)*(0.5*exp(2.0) - 2.0)*(exp(t)-1.0);
}
float f(float y, float t)
{
	return y-t*t+1.0;
}
int main()
{
float h = 0.2; //The step size we want to use (Made Global)
float a = 0.0; //Lower Bound
float b = 2.0; //Upper Bound
int n = (int)(ceil((b-a)/h));
float* w = NULL;
float* t = NULL;
t = (float*)calloc(n+1,sizeof(float));
w = (float*)calloc(n+1,sizeof(float));
t[0] = a;
w[0] = 0.5; // initial value

for(int i = 0;i<n+1;i=i+1)
{	
	t[i+1] = t[i] + h;
	w[i+1] = w[i] + h*f(w[i],t[i]);

}
printf("Time\ty(t)\tEuler\tTrue Error\tError Bound\n");
for(int i = 0;i<n+1;i=i+1)
{
	printf("%0.3f\t%0.3f\t%0.3f\t%0.3f\t\t%0.3f\t\n",t[i],y(t[i]),w[i],fabs(y(t[i])-w[i]),error_bound(t[i],h));
}
}