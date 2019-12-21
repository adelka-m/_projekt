# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:46:57 2019

@author: Barbara
"""


import numpy as np
import matplotlib.pyplot as plt

u0 = lambda x,y: np.sin(np.pi*x) + np.sin(np.pi*y)+3
#u0 = lambda x,y: np.exp(x*y)
#u0 = lambda x,y: 1+0*x+0*y

xs_help = np.linspace(0.1,1,10)
h = xs_help[1]-xs_help[0]
xs = np.concatenate((-xs_help[::-1], np.array([0]), xs_help))

Xs, Ys = np.meshgrid(xs,xs)

Us = u0(Xs,Ys)
for i in np.arange(len(xs)):                   ### BC and L shape
    for j in np.arange(len(xs)):
        if xs[i] >= 0 and xs[j] >= 0:
            Us[i][j] = 0
            Us[j][i] = 0
        if np.abs(xs[i]) == 1 or np.abs(xs[j]) == 1:
            Us[i][j] = 0
        
plt.scatter(Xs,Ys,c=Us)
plt.colorbar()
plt.show()


plt.imshow(Us,origin='center')
plt.show()
plt.imshow(Us,interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])
plt.show()
#############################################


f = lambda x,y: 1+0*x+0*y
#plt.imshow(f(Xs,Ys), interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])
#plt.colorbar()
#plt.show()


u = Us[1]    ### 1st column!! not row
for i in np.arange(1,len(Us)):
    u = np.concatenate((u,Us[i]))
    
    
### FEM-Method:
A = np.diag(4*np.ones(len(Xs)*len(Ys))) + np.diag(-np.ones(len(Xs)*len(Ys)-1),1) + np.diag(-np.ones(len(Xs)*len(Ys)-1),-1) + np.diag(-np.ones(len(Xs)*len(Ys)-len(xs)-1), len(xs)+1) + np.diag(-np.ones(len(Xs)*len(Ys)-len(xs)-1), -len(xs)-1)

for v in np.arange(len(u)):
    if u[v] == 0:
         A[:,v] = np.zeros_like(A[:,v])
         A[v,:] = np.zeros_like(A[v,:])
         A[v,v] = 1
         
### A is somehow wrong. The solution in the end is not right...??
        

b = np.ones_like(u)*h**2
for v in np.arange(len(u)):
    if u[v] == 0:
        b[v] = 0
        
        
sol = np.linalg.solve(1/h**2*A,b)

Ugrid = sol.reshape((len(Xs),len(Ys)))

plt.imshow(Ugrid, interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])
plt.colorbar()
plt.show()
#4*u[i,i]-u[i,i+1]-u[i,i-1]-u[i+1,i]-u[i-1,i]