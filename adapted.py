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
import math


def scalar_multiplication(S,E,n):
    ''' Implementation of some kind of a scalar multiplication of two matrices S, E of the dimension n*n. '''
    x = 0
    for i in range(n):
    	for j in range(n):
    		x = x + S[i][j]*E[i][j]

    return x

def derivative_approx(p,q,h):
	return np.abs(p-q)/h

def adapted_grid_CG(n):
	''' CG for finer grid at kink. '''
	''' Function for performing CG for our problem. 
    u_xx + u_yy = 1    -->  A u = b    ---> A symmetric, pos.semidef.   ---> CG  '''
        
	u = np.zeros((n,n))
	Ap = np.zeros((n,n))
	res = np.zeros((n,n))
	p = np.zeros((n,n))


	h1 = 0.4/(n//3)       #h1 = 0.4/(2*(n/2)//3)
	h2 = 1.6/(n-n//3)     #h2 = 1.6/(n-2*(n/2)//3)
	half = n//2
	third = half//3
	for i in range(1,half):
		for j in range(1,half):
			res[i][j] = 1
			res[n-1-i][j] = 1
			res[i][n-1-j] = 1

    # print('kink point at ', n//2)
	#print(res)
    # print(p.size)
	p = res.copy()
	resold = scalar_multiplication(res,res,n)
	resnew = 0

	k=1
	while True:
		for i in range(1,2*third):
			for j in range(1,2*third):
				#print('first:', i,j)
				Ap[i][j] = 1/h2 **2 * (4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1]))
				Ap[n-1-i][j] = 1/h2 **2 * (4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1]))
				Ap[i][n-1-j] = 1/h2 **2 * (4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)]))
                
			for j in range(2*third,half):
				#print('second:', i,j)
				Ap[i][j] = (2/h1**2 + 2/h2**2) * p[i][j]  -        (1/h2**2)*(p[i+1][j]+p[i-1][j])      -        (1/h1**2) * (p[i][j+1]+p[i][j-1]) 
				Ap[n-1-i][j] = (2/h1**2 + 2/h2**2) * p[n-1-i][j] - (1/h2**2)*(p[n-1-(i+1)][j]+p[n-1-(i-1)][j]) - (1/h1**2) * (p[n-1-i][j+1]+p[n-1-i][j-1]) 
				Ap[i][n-1-j] = (2/h1**2 + 2/h2**2) * p[i][n-1-j] - (1/h2**2)*(p[i+1][n-1-j]+p[i-1][n-1-j])     - (1/h1**2) * (p[i][n-1-(j+1)]+p[i][n-1-(j-1)])
	
		for i in range(2*third,half):
			for j in range(2*third,half):
				#print('third:', i,j)
				Ap[i][j] = 1/h1 **2 * (4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1]))
				Ap[n-1-i][j] = 1/h1 **2 * (4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1]))
				Ap[i][n-1-j] = 1/h1 **2 * (4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)]))
                
			for j in range(1,2*third):
				#print('forth:', i,j)
				Ap[i][j] = (2/h1**2 + 2/h2**2) * p[i][j] - (1/h1**2)*(p[i+1][j]+p[i-1][j]) - (1/h2**2)*(p[i][j+1]+p[i][j-1])
				Ap[n-1-i][j] = (2/h1**2 + 2/h2**2)*p[n-1-i][j] - (1/h1**2)*(p[n-1-(i+1)][j]+p[n-1-(i-1)][j]) - (1/h2**2)*(p[n-1-i][j+1]+p[n-1-i][j-1]) 
				Ap[i][n-1-j] = (2/h1**2 + 2/h2**2)*p[i][n-1-j] - (1/h1**2)*(p[i+1][n-1-j]+p[i-1][n-1-j])  - (1/h2**2)*(p[i][n-1-(j+1)]+p[i][n-1-(j-1)])

		#print(Ap)
        
		alpha = resold / scalar_multiplication(p,Ap,n)
		u = u + alpha * p
		res = res - alpha * Ap

		#derivative = derivative_approx(u[n//2][n//2],u[n//2-1][n//2-1], math.sqrt(2)*h1)
		#print(derivative)
        
        
		resnew = scalar_multiplication(res,res,n)

		#print(resnew)
		if np.sqrt(1/(0.75*n**2)*resnew) < 1e-5:  # mean of the norm^2 of residuum
			#print( np.absolute(res[(n)//2][(n)//2]) )
			print( "number of steps: ", k, ", for n = ",n )
			#break
    	
		if k > n**2:
			print(k)
			break
		
		# plt.imshow(u, origin = 'center')
		# plt.show()

		k = k+1
		p = res + (resnew / resold) * p
		resold = resnew

	return np.sqrt(1/(0.75*n**2)*resnew), u, res[n//2-4:n//2+4,n//2-1:n//2+4]


def conjugated_gradients(n):
    ''' Function for performing CG for our problem. 
           u_xx + u_yy = 1    -->  A u = b    ---> A symmetric, pos.semidef.   ---> CG  '''
        
    u = np.zeros((n,n))
    Ap = np.zeros((n,n))
    res = np.zeros((n,n))
    p = np.zeros((n,n))

    h = 2/(n)
    for i in range(1,(n)//2):
    	for j in range(1,(n)//2):
    		res[i][j] = h*h
    		res[n-1-i][j] = h*h
    		res[i][n-1-j] = h*h

    # print('kink point at ', n//2)
    # print(res)
    # print(p.size)
    p = res.copy()
    resold = scalar_multiplication(res,res,n)
    resnew = 0

    k=1
    while True:
    	for i in range(1,(n)//2):
    		for j in range(1,(n)//2):
    			Ap[i][j] = 4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])

    			Ap[n-1-i][j] = 4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1])
    			Ap[i][n-1-j] = 4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)])
    	
    	alpha = resold / scalar_multiplication(p,Ap,n)
    	u = u + alpha * p
    	res = res - alpha * Ap


    	resnew = scalar_multiplication(res,res,n)


    	if np.sqrt(1/(0.75*n**2)*resnew) < 1e-5:  # mean of the norm^2 of residuum
    	 	#print( np.absolute(res[(n)//2][(n)//2]) )
    	 	print( "number of steps: ", k, ", for n = ",n )
    	 	break
    	
    	if k > n**2:
    	 	break
	

    	k = k+1
    	p = res + (resnew / resold) * p
    	resold = resnew
    	
    #print(k)
    derivativeD = derivative_approx(u[n//2][n//2],u[n//2-1][n//2-1],np.sqrt(2)*h)
    derivativeU = derivative_approx(u[n//2][n//2],u[n//2-1][n//2],h)
    print(derivativeD, derivativeU)
    resnew = scalar_multiplication(res,res,n)
    
    return u,derivativeD,derivativeU
    #return(np.sqrt(1/(0.75*n**2)*resnew))

def conjugated_gradients_square(n):
    ''' Function for performing CG for our problem. 
           u_xx + u_yy = 1    -->  A u = b    ---> A symmetric, pos.semidef.   ---> CG  '''
    b = np.zeros((n,n))     
    u = np.zeros((n,n))
    Ap = np.zeros((n,n))
    res = np.zeros((n,n))
    p = np.zeros((n,n))
    

    h = 2/n
    for i in range(1,n-1):
    	for j in range(1,n-1):
    		b[i][j] = h*h

    res = b
    p = res.copy()
    resold = scalar_multiplication(res,res,n)
    resnew = 0

    k=1
    while True:
    	for i in range(1,n-1):
    		for j in range(1,n-1):
    			Ap[i][j] = 4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])


    	alpha = resold / scalar_multiplication(p,Ap,n)
    	u = u + alpha * p
    	res = res - alpha * Ap


    	resnew = scalar_multiplication(res,res,n)


    	if 1/n*resnew < 1e-8:  # mean of the norm^2 of residuum
    	 	#print( np.absolute(res[(n)//2][(n)//2]) )
    	 	#print( "number of steps: ", k, ", for n = ",n )
    	 	break
    	
    	if k > n/2:
    	 	break


    	k = k+1
    	p = res + (resnew / resold) * p
    	resold = resnew
    print(k)
    
    return math.sqrt((1/n**2)*resnew)


"""
d = []
d2 =[]
N = np.arange(4,206,50)
for n in N:
    sol = conjugated_gradients(2*n)
    d.append(sol[1])
    d2.append(sol[2])
    #print(n)
    #plt.imshow(conjugated_gradients(2*n)[0], origin='center')
    #plt.colorbar()
    
    #plt.show()

#N = np.array([ 34,  84, 134, 184, 234, 284, 334, 384, 434, 484, 534, 584, 634, 684, 734, 784, 834, 884, 934, 984])
#d = [0.7478890428765244, 0.7972648312550097, 0.8302749080260768, 0.858049018418628, 0.8819413930727685, 0.9026620045916867, 0.9207270966804443, 0.936541229875971, 0.9504298114813072, 0.9626583739364073, 0.9734459140507935, 0.9829746819059307, 0.9913975695931275, 0.9988437798201382, 1.0054232311611386, 1.011230019537546, 1.0163451582193919, 1.020838772992197, 1.024771867537335, 1.0281977580046682]

plt.semilogx(N,d, marker='o', label='diagonal derivative at (0,0)')
plt.semilogx(N,d2, marker='o', label='vertical derivative at (0,0)')
plt.legend()
plt.savefig('derivates.png',dpi=300)
plt.show()

#d = [0.37106639453232465, 1.1988106064148063, 1.4959747574759767, 1.7075064792662098, 1.875347916486363]
#d2 = [0.3949523630406332, 1.1126710401062396, 1.3817345633907745, 1.5732594811369056, 1.7268959654084304]
### Ableitung in diagonale Richtung


"""


#u = conjugated_gradients(2*5)
#plt.imshow(u, origin = 'center')
#plt.scatter(12//2-1, 12//2-1)
#plt.show()


#res = []
#res_square = []
res_ad = []
N = np.arange(5,51,5)
for n in N:
	print(n)
	#res.append( math.sqrt(conjugated_gradients(2*n)/(3*(n-1)**2)) )
	res_ad.append( adapted_grid_CG(2*n)[0])
    #res_square.append(conjugated_gradients_square(2*n))

"""
alpha = np.log(res[-1]/res[-2])/np.log((3*(N[-1]-1)**2)/((3*(N[-2]-1)**2)))
print(alpha)
alpha = np.log(res_square[-1]/res_square[-2])/np.log((3*(N[-1]-1)**2)/((3*(N[-2]-1)**2)))
print(alpha)
xx = []
for n in N:
    xx.append(  ((3*(n-1)**2))**alpha * res[0] * ((3*(N[0]-1)**2))**(-alpha) )

slope_square = 0
slope_L = 0
for i in range(1,len(res_square)-1):
    slope_square = slope_square - 1/(len(res_square)-1) * (res_square[i] - res_square[i+1])
    slope_L = slope_L - 1/(len(res_square)-1) * (res[i] - res[i+1])

"""
#print('slope L:', slope_L,', slope square:', slope_square)
#plt.semilogy( 2*N+1, xx , linestyle='dashed', label = 'order of -1/2')
plt.semilogy( 2*N[:-3]+1, res_ad, linestyle='dotted',marker='o',label = 'discrete L2 norm of residuum')
#plt.semilogy( 2*N+1, res_square, linestyle='dotted',marker='*',label = 'discrete L2 norm of residuum square')

# plt.title("Value of residual error")
# plt.xlabel("Number of points") 
# plt.ylabel("Error (log scale)")
# plt.legend()
# plt.show()
# plt.imshow(res[0], interpolation = 'lanczos', origin = 'center', extent = [-1,1,-1,1])
# plt.colorbar()
# plt.show()
