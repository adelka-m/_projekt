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


def scalar_multiplication(S,E,n):
     ''' Implementation of some kind of a scalar multiplication of two matrices S, E of the dimension n*n. '''

     x = 0
    
     for i in range(n):
          for j in range(n):
               x = x + S[i][j]*E[i][j]

     return x


def conjugated_gradients(n):
     ''' Function for performing CG for our problem. '''

     b = np.zeros((n,n))
     u = np.zeros((n,n))
     Ap = np.zeros((n,n))


     h = 1/n
     for i in range(1,(n+1)//2):
          for j in range(1,(n+1)//2):
               b[i][j] = h*h
               b[n-1-i][j] = h*h
               b[i][n-1-j] = h*h
  
     res = b
     p = res
     
     resold = scalar_multiplication(res,res,n-1)
     resnew = resold
     
     k=1
     while True:
       
          for i in range(1,(n+1)//2):
               for j in range(1,(n+1)//2):
                    Ap[i][j] = 4*p[i][j] - (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])
                    Ap[n-1-i][j] = 4*p[n-1-i][j] - (p[n-1-(i+1)][j]+p[n-1-(i-1)][j]+p[n-1-i][j+1]+p[n-1-i][j-1])
                    Ap[i][n-1-j] = 4*p[i][n-1-j] - (p[i+1][n-1-j]+p[i-1][n-1-j]+p[i][n-1-(j+1)]+p[i][n-1-(j-1)])

       
          alpha = resold / scalar_multiplication(p,Ap,n-1)
          u = u + alpha * p
          res = res - alpha * Ap
          resnew = scalar_multiplication(res,res,n-1)
        
          if math.sqrt(resnew) < 1e-4:
               print(resnew)
               print(k)
               break

          p = res + (resnew / resold) * p
          resold = resnew
       
     
     return u
 

n = 100 

plt.imshow(conjugated_gradients(2*n+1), interpolation = 'none', origin = 'center', extent = [-1,1,-1,1])
plt.colorbar()
plt.show()