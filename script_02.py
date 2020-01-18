'''
This project is going to investigate the impact of nonsmooth border (in particular L shape) on the smoothness of a solution.
'''
'''            ---u_0 = 0---
              |           |
              |           |
              |           |
              |           | 
              n         (0,0) ----------
              |                        |
              |                      u_0 = 0
              |                        |
              |                        |
              -u_0 = 0---- n ------(1,-1)  
 '''
import numpy as np
import matplotlib.pyplot as plt
import math




def init_array_of_grid_points(n):
    ''' This function is going to make an array of grid points, so we can refine grid in easy manner'''
    '''  array =  | value P1   | , | value P2  | , ...  0
                  |neighbor1 1 | , |neighbor2 1| ,...   1
                  |neighbor1 2 | , |neighbor2 2| ,...   2
                  |neighbor1 3 | , |neighbor2 3| ,...   3
                  |neighbor1 4 | , |neighbor2 4| ,...   4
                  |   h 1 1    | , |   h 2 1   | ,...   5
                  |   h 1 2    | , |   h 2 2   | ,...   6
                  |   h 1 3    | , |   h 2 3   | ,...   7
                  |   h 1 4    | , |   h 2 4   | ,...   8
    '''

    ''' Number of grid points is now 3*n**2. '''
    h = 1/(n)
    A = np.empty((3*n**2, 9))
    for i in range(3*n**2):

        

        boundary = {e for e in np.arange(0,2*n)}
        boundary.update( e for e in np.arange(    0,    2*n**2-n,     2*n ) )   #ok
        boundary.update( e for e in np.arange(2*n-1,    2*n**2    ,     2*n) ) #ok
        boundary.update( e for e in np.arange(2*n**2,   3*n**2-n    ,  n) ) #ok
        boundary.update( e for e in np.arange(2*n**2+n-1,   3*n**2    ,  n) )# ok
        boundary.update( e for e in np.arange(3*n**2-n, 3*n**2) ) #ok
        boundary.update( e for e in np.arange(2*n**2-n-1, 2*n**2) ) #ok
        

        if i < 2*n**2:        # lower part of L, rectangle
            if i not in boundary:
                A[i][0] = 1     # initial condition
                A[i][1] = i-1   # neighboring grid points
                A[i][2] = i+1
                A[i][3] = i+2*n
                A[i][4] = i-2*n
                A[i][5] = h     # distance between grid point and its neighbor
                A[i][6] = h
                A[i][7] = h
                A[i][8] = h

            else:           #grid points on boundary
                A[i][0] = 0     # initial condition
                if i in np.arange(0,2*n):
                    A[i][1] = i-1
                    A[i][2] = i+1
                    A[i][3] = i+2*n
                    A[i][4] = -1
                    A[i][5] = h
                    A[i][6] = h
                    A[i][7] = h
                    A[i][8] = 0
                    if i == 0:
                        A[i][1] = -1
                        A[i][5] = 0
                    if i == 2*n-1:
                        A[i][2] = -1
                        A[i][6] = 0
                elif i in np.arange(0,    2*n**2-n,     2*n ):
                    A[i][1] = -1
                    A[i][2] = i+1
                    A[i][3] = i+2*n
                    A[i][4] = i-2*n
                    A[i][5] = 0
                    A[i][6] = h
                    A[i][7] = h
                    A[i][8] = h
                    if i == 0:
                        A[i][4] = -1
                        A[i][8] = 0
                elif i in np.arange(2*n-1,    2*n**2-1    ,     2*n):
                    A[i][1] = i-1
                    A[i][2] = -1
                    A[i][3] = i+2*n
                    A[i][4] = i-2*n
                    A[i][5] = h
                    A[i][6] = 0
                    A[i][7] = h
                    A[i][8] = h
                    if i == 2*n-1:
                        A[i][4] = -1
                        A[i][8] = 0
                elif i in np.arange(2*n**2-n-1, 2*n**2):
                    A[i][1] = i-1
                    A[i][2] = i+1
                    A[i][3] = -1
                    A[i][4] = i-2*n
                    A[i][5] = h
                    A[i][6] = h
                    A[i][7] = 0
                    A[i][8] = h
                    if i == 2*n**2:
                        A[i][2] = -1
                        A[i][6] = 0


                    

        else:              #upper part of L, square
            if i not in boundary:
                A[i][0] = 1           # initial condition

                A[i][1] = i-1
                A[i][2] = i+1
                A[i][3] = i+n
                if i < 2*n**2 + n:
                    A[i][4] = i-2*n
                else:
                    A[i][4] = i-n
                A[i][5] = h
                A[i][6] = h
                A[i][7] = h
                A[i][8] = h 
            else:
                A[i][0] = 0    # initial condition
                if i in np.arange(2*n**2+n-1,   3*n**2-1    ,  n):
                    A[i][1] = i-1
                    A[i][2] = -1
                    A[i][3] = i+n
                    if i == 2*n**2+n-1: 
                        A[i][4] = i-2*n
                    else:
                        A[i][4] = i-n
                    A[i][5] = h
                    A[i][6] = 0
                    A[i][7] = h
                    A[i][8] = h

                elif i in np.arange(2*n**2,   3*n**2-n    ,  n):
                    A[i][1] = -1
                    A[i][2] = i+1
                    A[i][3] = i+n
                    if i == 2*n**2: 
                        A[i][4] = i-2*n
                    else:
                        A[i][4] = i-n
                    A[i][5] = 0
                    A[i][6] = h
                    A[i][7] = h
                    A[i][8] = h
                elif i in np.arange(3*n**2-n, 3*n**2):
                    A[i][1] = i-1
                    A[i][2] = i+1
                    A[i][3] = -1
                    A[i][4] = i-n
                    if i == 3*n**2-n: 
                        A[i][1] = -1
                    if i ==3*n**2-1:
                        A[i][2] = -1
                    A[i][5] = h
                    A[i][6] = h
                    A[i][7] = 0
                    A[i][8] = h
                
    return A, boundary

def scalar_multiplication(S,E):
    ''' Implementation of some kind of a scalar multiplication of two matrices S, E of the dimension n*n. '''
    x = 0
    for i in range(len(S)):
            x = x + S[i]*E[i]

    return x

def conjugated_gradients(n):
    ''' Function for performing CG for our problem. 
           u_xx + u_yy = 1    -->  A u = b    ---> A symmetric, pos.semidef.   ---> CG  '''

    A, boundary = init_array_of_grid_points(n)
    p = A[:,0]
    print('A O ', A[:,0])
    
    resold = np.dot(p,p)
    res = p
    resnew = 0

    k=1
    while True:
        for i in range(len(A)):
            if i not in boundary:
                print(1/A[i,5]**2)
               # A[i,0] = (1/A[i,5]**2 + 1/A[i,6]**2 + 1/A[i,7]**2 + 1/A[i,8]**2) * p[i] - ( 1/A[i,5]**2*p[int(A[i,1])] + 1/A[i,6]**2 *p[int(A[i,2])] +  1/A[i,7]**2*p[int(A[i,3])] +  1/A[i,8]**2*p[int(A[i,4])] )
                A[i,0] = (A[i,5]**2 + A[i,6]**2 + A[i,7]**2 + A[i,8]**2) * p[i] - ( A[i,5]**2*p[int(A[i,1])] + A[i,6]**2 *p[int(A[i,2])] +  A[i,7]**2*p[int(A[i,3])] +  A[i,8]**2*p[int(A[i,4])] )

        
        
        res = res - resold / np.dot(p,A[:,0])* A[:,0] 
        print(res)
        resnew = np.dot(res,res)
        print(resnew)

        #if 1/n*resnew < 1e-5:  # mean of the norm^2 of residuum
        #print( np.absolute(res[(n)//2][(n)//2]) )
        #print( "number of steps: ", k, ", for n = ",n )
        #break
      
        if k > 2:
            break
        

        k = k+1
        p = res + (resnew / resold) * p
        resold = resnew

    return math.sqrt(1/(3*n**2)*resnew)

 

res = []
N = np.arange(4,10,10)
for n in N:
    print(n)
    res.append(conjugated_gradients(n))


xx = []
for n in N:
    xx.append( 1/ math.sqrt((2*n+1)**2*0.75) * res[0] * math.sqrt((2*N[0]+1)**2*0.75) )



plt.semilogy( 2*N+1, xx , linestyle='dashed', label = 'order of -1/2')
plt.semilogy( 2*N+1, res, linestyle='dotted',marker='o',label = 'discrete L2 norm of residuum')

plt.title("Value of residual error")
plt.xlabel("Number of points") 
plt.ylabel("Error (log scale)")
plt.legend()
plt.show()