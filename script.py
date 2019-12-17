'''
This project is going to investigate the impact of nonsmooth border (in particular L shape) on the smoothness of a solution.
'''

import numpy as np
import matplotlib.pyplot as plt



u0 = lambda x,y: np.sin(np.pi*x) + np.sin(np.pi*y)

xs_help = np.linspace(0.1,1,10)
xs = np.concatenate((-xs_help[::-1], np.array([0]), xs_help))

Xs, Ys = np.meshgrid(xs,xs)

Us = u0(Xs,Ys)
for i in np.arange(len(xs)):
    for j in np.arange(len(xs)):
        if xs[i] > 0 and xs[j] > 0:
            Us[i][j] = 0
            Us[j][i] = 0                      ### BC: u = 0
        
        
##Visualization        
plt.scatter(Xs,Ys,c=Us)
plt.colorbar()
plt.show()

plt.imshow(Us,origin='center')
plt.show()

plt.imshow(Us,interpolation='Gaussian',origin='center',extent=[-1,1,-1,1])


