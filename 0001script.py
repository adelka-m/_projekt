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


	number_of_refined_points = n//3
	
	refined_grid = {(0,0),(0,1),(1,0),(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1), (2,-1), (2,0), (0,2),(-1,2)}
	for i in range(number_of_refined_points+1):
		refined_grid.update([(i,-1), (i,0), (0,i),(-1,i)])



	
	h1 = 2/(2*n)
	h2 = (2-number_of_refined_points/n)/(n-number_of_refined_points)

	for i in range(1,(n)//2):
		for j in range(1,(n)//2):
			res[i][j] = 1
			res[n-1-i][j] = 1
			res[i][n-1-j] = 1


	p = res.copy()
	resold = scalar_multiplication(res,res,n)
	resnew = 0

	test = np.zeros((n,n))

	k=1
	while True:

		
		for i,j in refined_grid:
			i -= 1
			j -= 1	
			m=1
			Ap[(n)//2+m*i][(n)//2+m*j] = 1/h1**2 * (4*p[(n)//2 +m*i][(n)//2 +m*j]  - (p[(n)//2 + m*i+1][(n)//2 +m*j]+p[ (n)//2 + m*i-1][(n)//2 +m*j]+p[(n)//2 + m*i][(n)//2 + m*j+1]+p[(n)//2 +i][(n)//2 +m*j-1]))
			test[(n)//2+i][(n)//2+j] = 100
		for i in range(1,(n)//2):
			for j in range(1,(n)//2):
				if i < n//2 -2 and j < n//2 -2:  # region 1

					Ap[i][j]     = 1/h2 **2 * (4*p[i][j] -     (p[i+1][j]      +p[i-1][j]      +p[i][j+1]      +p[i][j-1]))
					Ap[n-1-i][j] = 1/h2 **2 * (4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]  +p[n-1-i][j-1]))
					Ap[i][n-1-j] = 1/h2 **2 * (4*p[i][n-1-j] - (p[i+1][n-1-j]  +p[i-1][n-1-j]  +p[i][n-1-(j+1)]+p[i][n-1-(j-1)]))
					test[i][j] = 200
					test[n-1-i][j] = 200
					test[i][n-1-j] = 200

				elif  i < n//2 -2 and j > n//2 -number_of_refined_points-1: # region 2
					# print('nnn:', i,j)
					# print('nnn1:', n//2 + i+1, j)
					# print('nn2', n-i-1, j)

					a = n//2 + i +1
					b = n-i-1
					Ap[i][j] =     (2/h1**2 + 2/h2**2) * p[i][j]  -      (1/h2**2)*(p[i+1][j]+p[i-1][j])      -        (1/h1**2) * (p[i][j+1]  +p[i][j-1]) 
					Ap[a][j] =     (2/h1**2 + 2/h2**2) * p[a][j]  -      (1/h2**2)*(p[a+1][j]+p[a-1][j]) -             (1/h1**2) * (p[a][j+1]  +p[a][j-1]) 
					Ap[b][j] =     (2/h1**2 + 2/h2**2) * p[b][j] -       (1/h2**2)*(p[b+1][j]+p[b-1][j])     -         (1/h1**2) * (p[b][(j+1)]+p[b][(j-1)])
					Ap[i][n-j-1] = (2/h1**2 + 2/h2**2) * p[i][n-j-1]  -  (1/h2**2)*(p[i+1][n-j-1]+p[i-1][n-j-1])      -(1/h1**2) * (p[i][n-j-1+1]  +p[i][n-j-1-1])
					test[i][j] = 300
					test[a][j] = 300
					test[b][j] = 300
					test[i][n-1-j] = 300

				elif  j < n//2 -2 and i > n//2 -number_of_refined_points-1:    # region 4
					# print('AAAA:', i,j)
					# print('AAAAAAA:', i, n//2+j+1)
				 # 	print('AAAA!:', n-i-1, j)

					a = n//2 +j +1
					Ap[i][j] =     (2/h1**2 + 2/h2**2) * p[i][j]   - (1/h1**2)*(p[i+1][j]      +p[i-1][j])       - (1/h2**2)*(p[i][j+1]    +p[i][j-1])
					Ap[n-1-i][j] = (2/h1**2 + 2/h2**2)*p[n-1-i][j] - (1/h1**2)*(p[n-1-(i+1)][j]+p[n-1-(i-1)][j]) - (1/h2**2)*(p[n-1-i][j+1]+p[n-1-i][j-1]) 
					Ap[i][a] =     (2/h1**2 + 2/h2**2)*p[i][a]     - (1/h1**2)*(p[i+1][a]      +p[i-1][a])       - (1/h2**2)*(p[i][a+1]    +p[i][a-1])
					test[i][j] = 400
					test[n-1-i][j] = 400
					test[i][a] = 400
		
		# plt.imshow(test)
		# plt.colorbar()
		# plt.show()
		alpha = resold / scalar_multiplication(p,Ap,n)
		u = u + alpha * p
		res = res - alpha * Ap

		derivative = derivative_approx(u[n//2][n//2],u[n//2-1][n//2-1], math.sqrt(2)*h1)
		
		resnew = scalar_multiplication(res,res,n)

		#print(resnew)
		#if 1/n*resnew < 1e-5:  # mean of the norm^2 of residuum
			#print( np.absolute(res[(n)//2][(n)//2]) )
			#print( "number of steps: ", k, ", for n = ",n )
			#break
    	
		if k > n:
			break
		

		k = k+1
		p = res + (resnew / resold) * p
		resold = resnew

	# plt.imshow(u)
	# plt.show()
	return np.sqrt(1/(n**2)*resnew)


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


    	if 1/n**2*resnew < 1e-15:  # mean of the norm^2 of residuum
    	 	#print( np.absolute(res[(n)//2][(n)//2]) )
    	 	#print( "number of steps: ", k, ", for n = ",n )
    	 	break
    	
    	if k > n:
    	 	break
	

    	k = k+1
    	p = res + (resnew / resold) * p
    	resold = resnew
    # plt.imshow(u)
    # plt.show()
    return np.sqrt(1/(n**2)*resnew)

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


    	if 1/n**2*resnew < 1e-15:  # mean of the norm^2 of residuum
    	 	#print( np.absolute(res[(n)//2][(n)//2]) )
    	 	#print( "number of steps: ", k, ", for n = ",n )
    	 	break
    	
    	if k > n:
    	 	break
	

    	k = k+1
    	p = res + (resnew / resold) * p
    	resold = resnew

    return math.sqrt((1/n**2)*resnew)




res_adapted = []
res_L = []
res_Q = []
N = np.arange(30,76,10)
for n in N:
	print(n)
	res_adapted.append(adapted_grid_CG(n))
	res_L.append(conjugated_gradients(n))
	res_Q.append(conjugated_gradients_square(n))

# plt.semilogy(N, res_adapted, linestyle='dotted',marker='o', label = 'adapted grid')
plt.semilogy( N, res_Q, linestyle='dotted',marker='^',label = 'square')
plt.semilogy( N, res_L, linestyle='dotted',marker='*',label = 'L shape')
plt.legend()
plt.show()

# xx = []
# for n in N:
# 	xx.append(  ((3*(n-1)**2))**alpha * res[0] * ((3*(N[0]-1)**2))**(-alpha) )

slope_square = np.log(res_Q[-1]/res_Q[-2])/np.log((N[-1])**2/(N[-2])**2)
slope_L = np.log(res_L[-1]/res_L[-2])/np.log((N[-1])**2/(N[-2])**2)
slope_adapted = np.log(res_adapted[-1]/res_adapted[-2])/np.log((N[-1])**2/(N[-2])**2)

print('slope L:', slope_L,', slope square:', slope_square, 'slope adapted:', slope_adapted)
# #plt.semilogy( 2*N+1, xx , linestyle='dashed', label = 'order of -1/2')


# plt.title("Value of residual error")
# plt.xlabel("Number of points") 
# plt.ylabel("Error (log scale)")
# plt.legend()
# plt.imshow(res[0], interpolation = 'lanczos', origin = 'center', extent = [-1,1,-1,1])
# plt.colorbar()
# plt.show()