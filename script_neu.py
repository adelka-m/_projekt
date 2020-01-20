

'''
This project is going to investigate the impact of nonsmooth border (in particular L shape) on the smoothness of a solution.
'''
'''            ---u_0 = 0---
              |           |
              |           |
              |           |
              |           | 
              |         (0,0) ----------
              |                        |
              |                      u_0 = 0
              |                        |
              |                        |
              ------u_0 = 0----------(1,-1)  
 '''
import numpy as np
import matplotlib.pyplot as plt



def scalar_multiplication(S,E,n):
    ''' Implementation of some kind of a scalar multiplication of two matrices S, E of the dimension n*n. '''
    x = 0
    for i in range(n):
    	for j in range(n):
    		x = x + S[i][j]*E[i][j]

    return x

def adaptive_grid():
	''' From our residuum matrix, we conclude, if we refine grid at that point. "Big" residuum at x = refine grid at x. '''

	return 0

def conjugated_gradients(n):
    ''' Function for performing CG for our problem. 
           u_xx + u_yy = 1    -->  A u = b    ---> A symmetric, pos.semidef.   ---> CG  '''
    b = np.zeros((n,n))     
    u = np.zeros((n,n))
    Ap = np.zeros((n,n))
    res = np.zeros((n,n))
    p = np.zeros((n,n))

    h = 2/n
    #for i in range(1,(n+1)//2):
    #	for j in range(1,(n+1)//2):
    #		b[i][j] = h*h
    #		b[n-1-i][j] = h*h
    #		b[i][n-1-j] = h*h

    ### the square for comparison:
    for i in range(1,n-1):
    	for j in range(1,n-1):
    		b[i][j] = h*h
    		#b[n-1-i][j] = h*h
    		#b[i][n-1-j] = h*h


    res = b
    p = res
    #print(p)
    resold = scalar_multiplication(res,res,n)
    resnew = 0
    
    k=1
    while True:
        
        Ap2 = np.copy(Ap)
        
        #for i in range(1,(n+1)//2):
        #    for j in range(1,(n+1)//2):
        #        Ap2[i][j] = 4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])
        #        Ap2[n-1-i][j] = 4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1])
        #       Ap2[i][n-1-j] = 4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)])
        
        ###the square for comparison.
        for i in range(1,n-1):
            for j in range(1,n-1):
                Ap2[i][j] = 4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])
                #Ap2[n-1-i][j] = 4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1])
                #Ap2[i][n-1-j] = 4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)])
	       
        
        Ap = Ap2
        alpha = resold / scalar_multiplication(p,Ap,n)
        u = u + alpha * p
        res = res - alpha * Ap

    	#if k==1:
      		#print(Ap)
        resnew = scalar_multiplication(res,res,n)
        
        #if math.sqrt(0.75*(2/(n-1))**2*resnew)  < 1e-5:  # mean of the norm^2 of residuum
    	 	#print( np.absolute(res[(n)//2][(n)//2]) )
    	 	#print( "number of steps: ", k, ", for n = ",n )
            #break
    	
        if k > 0:
            break
	
        k = k+1
        p = res + (resnew / resold) * p
        resold = resnew
    
    print(np.shape(res), n)
    
    #return u    
    return np.sqrt(1/(0.75*n)*resnew)  ###???###

 

res = []
N = np.arange(4,110,10)
for n in N:
	#print(n)
	res.append(conjugated_gradients(2*n+1))

alpha = np.log(res[7]/res[6])/np.log(0.75*(2*N+1)[7]**2/(0.75*(2*N+1)[6]**2))
print(alpha)
#alpha=-0.5

xx = []
for n in N:
	xx.append( (0.75*(2*n+1)**2)**alpha * res[0] * ((0.75*(2*N[0]+1)**2)**(-alpha) ) )



plt.loglog( 0.75*(2*N+1)**2, xx , linestyle='dashed', label = 'order of -1/2')
plt.loglog( 0.75*(2*N+1)**2, res, linestyle='dotted',marker='o',label = 'discrete L2 norm of residuum')


plt.title("Value of residual error")
plt.xlabel("Number of points") 
plt.ylabel("Error (log scale)")
plt.legend()
plt.show()
# plt.imshow(res[0], interpolation = 'lanczos', origin = 'center', extent = [-1,1,-1,1])
# plt.colorbar()
# plt.show()