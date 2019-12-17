'''
This project is going to investigate the impact of nonsmooth border (in particular L shape) on the smoothness of a solution.
'''

import numpy as np
import matplotlib.pyplot as plt



xs_help = np.linspace(0.1,1,10)
xs = np.concatenate((-xs_help[::-1], np.array([0]), xs_help))

Xs, Ys = np.meshgrid(xs,xs)      ### creating an equidistant grid for a 'symmetric' L shape
for i in np.arange(len(xs)):
    for j in np.arange(len(xs)):
        if xs[i] > 0 and xs[j] > 0:
            Xs[i][j] = 0
            Ys[j][i] = 0

"""
#Visualizing the L
Zs = np.ones((len(xs),len(xs)))
for i in np.arange(len(xs)):
    for j in np.arange(len(xs)):
        if xs[i] > 0.01 and xs[j] > 0.01:
            Zs[i,j] = 0
        
plt.scatter(Xs,Ys,c=Zs)     ### Is there a problem at (0,0) or is this just the Scatterplot marking the origin?
"""
