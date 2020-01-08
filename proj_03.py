# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipl


"""
#u0 = lambda x,y: np.sin(np.pi*x) + np.sin(np.pi*y)+3
#u0 = lambda x,y: np.exp(x*y)
u0 = lambda x,y: 1+0*x+0*y

n=5
xs_help = np.linspace(1/(3*n),0.25,3*n)
xs2_help = np.linspace(0.25+1/n,1,n)
h = xs_help[1]-xs_help[0]
h2 = xs2_help[1]-xs2_help[0]
xs = np.concatenate((-xs2_help[::-1],-xs_help[::-1], np.array([0]), xs_help, xs2_help))

Xs,Ys = np.meshgrid(xs,xs)
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
u = Us[:,0]    
for i in np.arange(1,len(Us)):
    u = np.concatenate((u,Us[:,i]))
    
    
### FEM-Method:
A = np.diag(4*np.ones(len(Xs)*len(Ys))) + np.diag(-np.ones(len(Xs)*len(Ys)-1),1) + np.diag(-np.ones(len(Xs)*len(Ys)-1),-1) + np.diag(-np.ones(len(Xs)*len(Ys)-len(xs)), len(xs)) + np.diag(-np.ones(len(Xs)*len(Ys)-len(xs)), -len(xs))
### Fixed: the two outer diagonals where at the wrong place.
for v in np.arange(len(u)):
    if u[v] == 0:
         A[:,v] = np.zeros_like(A[:,v])
         A[v,:] = np.zeros_like(A[v,:])
         A[v,v] = 1
         8
                
b = np.ones_like(u)*h**2
for v in np.arange(len(u)):
    if u[v] == 0:
        b[v] = 0
        8
        
sol = np.linalg.solve(1/h**2*A,b)
Ugrid = sol.reshape((len(Xs),len(Ys)))
plt.imshow(Ugrid, interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])
plt.colorbar()
plt.show()
#4*u[i,i]-u[i,i+1]-u[i,i-1]-u[i+1,i]-u[i-1,i]
"""

##############################################################
##############################################################

def fem_poisL(n):                                 ### L shape
    xs_help = np.linspace(1/(2*n),0.25,2*n)
    xs2_help = np.linspace(0.25+1/n,1,n)
    h = xs_help[1]-xs_help[0]
    h2 = xs2_help[1]-xs2_help[0]
    xs = np.concatenate((-xs2_help[::-1],-xs_help[::-1], np.array([0]), xs_help, xs2_help))

    Xs, Ys = np.meshgrid(xs,xs)
    
    Us = np.ones((len(Xs),len(Ys)))
    for i in np.arange(len(xs)):                   ### BC and L shape
        for j in np.arange(len(xs)):
            if xs[i] >= 0 and xs[j] >= 0:
                Us[i][j] = 0
                Us[j][i] = 0
            if np.abs(xs[i]) == 1 or np.abs(xs[j]) == 1:
                Us[i][j] = 0
                u = Us[:,0]    

    for i in np.arange(1,len(Us)):                ### as a vector
        u = np.concatenate((u,Us[:,i]))
    
    A = np.diag(4*np.ones(len(u))) + np.diag(-np.ones(len(u)-1),1) + np.diag(-np.ones(len(u)-1),-1) + np.diag(-np.ones(len(u)-len(xs)), len(xs)) + np.diag(-np.ones(len(u)-len(xs)), -len(xs))
### Fixed: the two outer diagonals where at the wrong place.

    for v in np.arange(len(u)):                  ### BC and L shape
        if u[v] == 0:
            A[:,v] = np.zeros_like(A[:,v])
            A[v,:] = np.zeros_like(A[v,:])
            A[v,v] = 1
            8
                
    b = np.ones_like(u)*h**2
    for v in np.arange(len(u)):
        if u[v] == 0:
            b[v] = 0
            8
        
    sol = np.linalg.solve(1/h2**2*A,b)
    
    return(sol,xs)

#plt.imshow(fem_poisL(5)[0].reshape((11,11)), interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])
#plt.colorbar()
#plt.show()


def fem_poisQ(n):                         ### Square
    xs_help = np.linspace(1/(2*n),0.25,2*n)
    xs2_help = np.linspace(0.25+1/n,1,n)
    h = xs_help[1]-xs_help[0]
    h2 = xs2_help[1]-xs2_help[0]
    xs = np.concatenate((-xs2_help[::-1],-xs_help[::-1], np.array([0]), xs_help, xs2_help))

    Xs, Ys = np.meshgrid(xs,xs)
    
    
    Us = np.ones((len(Xs),len(Ys)))
    for i in np.arange(len(xs)):                   ### BC
        for j in np.arange(len(xs)):
            if np.abs(xs[i]) == 1 or np.abs(xs[j]) == 1:
                Us[i][j] = 0
                u = Us[:,0]    

    for i in np.arange(1,len(Us)):                ### as a vector
        u = np.concatenate((u,Us[:,i]))
    
    A = np.diag(4*np.ones(len(u))) + np.diag(-np.ones(len(u)-1),1) + np.diag(-np.ones(len(u)-1),-1) + np.diag(-np.ones(len(u)-len(xs)), len(xs)) + np.diag(-np.ones(len(u)-len(xs)), -len(xs))
### Fixed: the two outer diagonals where at the wrong place.

    for v in np.arange(len(u)):                  ### BC
        if u[v] == 0:
            A[:,v] = np.zeros_like(A[:,v])
            A[v,:] = np.zeros_like(A[v,:])
            A[v,v] = 1
            8
                
    b = np.ones_like(u)*h**2
    for v in np.arange(len(u)):
        if u[v] == 0:
            b[v] = 0
            8
        
    sol = np.linalg.solve(1/h2**2*A,b)
    
    return(sol,xs)



### Error plots:

ap = []
ap2 = []
hs = []
xxs_help = np.linspace(1/70,1,70)
xxs = np.concatenate((-xxs_help[::-1], np.array([0]), xxs_help))

for n in [5,7,10]:
    sol,xs = fem_poisL(n)
    hs.append(xs[1]-xs[0])
    ap.append(scipl.RectBivariateSpline(xs,xs,sol.reshape((len(xs),len(xs))),kx = 2)(xxs,xxs))
    #print(n)
    
    ### square for comparison
    sol2,xs = fem_poisQ(n)
    ap2.append(scipl.RectBivariateSpline(xs,xs,sol2.reshape((len(xs),len(xs))))(xxs,xxs))
   
    
err = []
err2 = []

v = len(ap)-1

for l in np.arange(v):
    err.append(np.max(np.abs(ap[l]-ap[v])))
    err2.append(np.max(np.abs(ap2[l]-ap2[v])))
    
plt.loglog(hs[:-1],err,label='Lshape')
plt.loglog(hs[:-1],err2,label='Square')
plt.loglog(hs[:-1],np.array(hs[:-1])**2,label='2nd order')
#plt.loglog(hs[:-1],np.array(hs[:-1])**4,label='4th order')
plt.legend()
plt.show()

### Lshape isn't "worse" than the square?? 
### Mistake in the interpolation and error estimation (e.g. maximal error)???
### Or something before that?
